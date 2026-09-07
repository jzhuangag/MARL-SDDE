"""Outcome-free cache and topology accounting audit on Pursuit states."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .async_pistonball_ctde import (
    OwnerGradientQueue,
    PolicyCacheBank,
    apply_refresh_action,
    changed_parameter_groups,
    clone_parameter_groups,
)
from .audit_pursuit_local_factorization import bounded_local_graph
from .composite_cache_lyapunov import (
    cache_mismatch_energy,
    topology_motion_increment,
)


class PursuitInterfaceActor(nn.Module):
    """Small distinct actor used only to exercise Pursuit cache semantics."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 3, 16),
            nn.Tanh(),
            nn.Linear(16, 5),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation.float())


def _pursuit_model(environment: Any) -> Any:
    current = environment
    for _ in range(12):
        if hasattr(current, "pursuer_layer"):
            return current
        if hasattr(current, "aec_env"):
            current = current.aec_env
        elif hasattr(current, "env"):
            current = current.env
        else:
            break
    raise RuntimeError("unable to locate the pinned Pursuit model")


def _parameter_vector(module: nn.Module) -> np.ndarray:
    return np.concatenate(
        [parameter.detach().cpu().numpy().ravel() for parameter in module.parameters()]
    )


def _fixed_universe_state(
    actors: Sequence[nn.Module], caches: PolicyCacheBank
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    current = {donor: _parameter_vector(actor) for donor, actor in enumerate(actors)}
    cached = {
        (donor, recipient): _parameter_vector(caches.cached_actor(recipient, donor))
        for recipient in range(len(actors))
        for donor in range(len(actors))
        if donor != recipient
    }
    return current, cached


def _active_weights(
    *,
    universe: Sequence[tuple[int, int]],
    owner_donor_edges: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    active = {(donor, owner) for owner, donor in owner_donor_edges}
    return {edge: float(edge in active) for edge in universe}


def run_audit(
    *,
    seeds: Sequence[int],
    action_seed: int,
    cycles: int = 40,
    n_pursuers: int = 8,
    n_evaders: int = 30,
    observation_range: int = 7,
    max_neighbors: int = 4,
) -> dict[str, object]:
    """Check launch, receipt, fixed-universe, and topology-motion invariants."""

    from pettingzoo.sisl import pursuit_v4
    from pettingzoo.sisl.pursuit import pursuit, pursuit_base

    torch.manual_seed(78123)
    actors = tuple(PursuitInterfaceActor() for _ in range(n_pursuers))
    caches = PolicyCacheBank(actors)
    universe = tuple(
        (donor, recipient)
        for recipient in range(n_pursuers)
        for donor in range(n_pursuers)
        if donor != recipient
    )
    fixed_weights = {edge: 1.0 for edge in universe}
    rng = np.random.default_rng(int(action_seed))
    refresh_count = 0
    charged_bytes = 0
    zero_weight_receipts = 0
    topology_changes = 0
    positive_topology_jumps = 0
    maximum_reset_error = 0.0
    maximum_topology_error = 0.0

    for seed in seeds:
        environment = pursuit_v4.parallel_env(
            n_pursuers=n_pursuers,
            n_evaders=n_evaders,
            max_cycles=cycles,
            obs_range=observation_range,
            n_catch=2,
            shared_reward=True,
        )
        observations, _ = environment.reset(seed=int(seed))
        try:
            for event in range(cycles):
                model = _pursuit_model(environment)
                positions = tuple(
                    tuple(int(value) for value in model.pursuer_layer.get_position(i))
                    for i in range(n_pursuers)
                )
                graph, _degrees = bounded_local_graph(
                    positions,
                    radius=observation_range // 2,
                    max_neighbors=max_neighbors,
                )
                owner = event % n_pursuers
                eligible = tuple(donor for edge_owner, donor in graph if edge_owner == owner)
                if eligible:
                    donor = eligible[0]
                    with torch.no_grad():
                        next(actors[donor].parameters()).add_(0.0001 * (event + 1))
                    caches.mark_owner_update(donor)
                    current, cached = _fixed_universe_state(actors, caches)
                    before = cache_mismatch_energy(current, cached, fixed_weights)
                    reset = 0.5 * float(
                        np.sum((current[donor] - cached[(donor, owner)]) ** 2)
                    )
                    action = apply_refresh_action(
                        recipient=owner,
                        donors=(donor,),
                        actors=actors,
                        caches=caches,
                    )
                    next_current, next_cached = _fixed_universe_state(actors, caches)
                    after = cache_mismatch_energy(
                        next_current, next_cached, fixed_weights
                    )
                    maximum_reset_error = max(
                        maximum_reset_error, abs((before - after) - reset)
                    )
                    if action.optional_policy_bytes <= 0:
                        raise AssertionError("a non-null refresh must charge actor bytes")
                    charged_bytes += action.optional_policy_bytes
                    refresh_count += 1

                    refreshed = tuple(
                        parameter.detach().clone()
                        for parameter in caches.cached_actor(owner, donor).parameters()
                    )
                    before_receipt = clone_parameter_groups(actors)
                    queue = OwnerGradientQueue()
                    queue.launch(
                        owner=owner,
                        launch_event=event,
                        delay=0,
                        gradients=tuple(
                            torch.ones_like(parameter)
                            for parameter in actors[owner].parameters()
                        ),
                        packet_weight=0.0,
                    )
                    received = queue.apply_due(
                        event=event,
                        actors=actors,
                        caches=caches,
                        step=0.01,
                    )
                    if len(received) != 1 or changed_parameter_groups(
                        before_receipt, actors
                    ):
                        raise AssertionError("zero packet weight changed an actor")
                    if not all(
                        torch.equal(saved, current_parameter.detach())
                        for saved, current_parameter in zip(
                            refreshed, caches.cached_actor(owner, donor).parameters()
                        )
                    ):
                        raise AssertionError("receipt erased a persistent refresh")
                    zero_weight_receipts += 1

                actions = {
                    agent: int(rng.integers(0, 5)) for agent in environment.agents
                }
                observations, _rewards, terms, truncations, _infos = environment.step(
                    actions
                )
                if not environment.agents:
                    break
                next_model = _pursuit_model(environment)
                next_positions = tuple(
                    tuple(
                        int(value)
                        for value in next_model.pursuer_layer.get_position(index)
                    )
                    for index in range(n_pursuers)
                )
                next_graph, _next_degrees = bounded_local_graph(
                    next_positions,
                    radius=observation_range // 2,
                    max_neighbors=max_neighbors,
                )
                current, cached = _fixed_universe_state(actors, caches)
                old_active = _active_weights(
                    universe=universe, owner_donor_edges=graph
                )
                new_active = _active_weights(
                    universe=universe, owner_donor_edges=next_graph
                )
                direct = cache_mismatch_energy(current, cached, new_active) - cache_mismatch_energy(
                    current, cached, old_active
                )
                exact = topology_motion_increment(
                    current, cached, old_active, new_active
                )
                maximum_topology_error = max(
                    maximum_topology_error, abs(direct - exact)
                )
                topology_changes += int(graph != next_graph)
                positive_topology_jumps += int(exact > 1e-15)
                if any(terms.values()) or any(truncations.values()):
                    break
        finally:
            environment.close()

    gates = {
        "C1_nonnull_refreshes_exercised": refresh_count > 0,
        "C2_policy_bytes_charged": charged_bytes > 0,
        "C3_exact_fixed_universe_reset": maximum_reset_error <= 1e-10,
        "C4_zero_weight_cache_persistence": zero_weight_receipts == refresh_count,
        "C5_dynamic_support_exercised": topology_changes > 0,
        "C6_exact_topology_motion": maximum_topology_error <= 1e-10,
        "C7_positive_topology_jump_observed": positive_topology_jumps > 0,
    }
    return {
        "audit": "PURSUIT-CACHE-SEMANTICS-QUALIFICATION",
        "scientific_efficacy_evidence": False,
        "reward_values_read": False,
        "passed": all(gates.values()),
        "gates": gates,
        "configuration": {
            "seeds": list(map(int, seeds)),
            "action_seed": int(action_seed),
            "cycles": int(cycles),
            "n_pursuers": int(n_pursuers),
            "n_evaders": int(n_evaders),
            "observation_range": int(observation_range),
            "max_neighbors": int(max_neighbors),
        },
        "refresh_count": refresh_count,
        "charged_policy_bytes": charged_bytes,
        "zero_weight_receipts": zero_weight_receipts,
        "topology_changes": topology_changes,
        "positive_topology_jumps": positive_topology_jumps,
        "maximum_reset_identity_error": maximum_reset_error,
        "maximum_topology_identity_error": maximum_topology_error,
        "source_sha256": {
            "pursuit.py": hashlib.sha256(
                Path(inspect.getfile(pursuit)).read_bytes()
            ).hexdigest(),
            "pursuit_base.py": hashlib.sha256(
                Path(inspect.getfile(pursuit_base)).read_bytes()
            ).hexdigest(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(
        seeds=tuple(range(92000, 92008)),
        action_seed=9981,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
