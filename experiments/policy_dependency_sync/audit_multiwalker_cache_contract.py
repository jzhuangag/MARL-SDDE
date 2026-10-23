"""Outcome-free Multiwalker policy-cache and receipt contract."""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .async_pistonball_ctde import PolicyCacheBank, apply_refresh_action
from .neural_signed_cache import parameter_bytes


class MultiwalkerContractActor(nn.Module):
    def __init__(self, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.network = nn.Sequential(
            nn.Linear(31, 32),
            nn.Tanh(),
            nn.Linear(32, 4),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation.float())


@dataclass(frozen=True)
class MultiwalkerCacheContract:
    seeds: tuple[int, ...]
    walkers: int
    cycles: int
    launches: int
    receipts: int
    nonnull_refreshes: int
    optional_policy_bytes: int
    maximum_degree: int
    action_bound_violations: int
    nonfinite_observations: int
    exact_cache_copy_failures: int
    receipt_order_failures: int
    changed_cached_actions: int
    trace_sha256: str


def chain_neighbors(owner: int, walkers: int) -> tuple[int, ...]:
    if walkers <= 1 or not 0 <= owner < walkers:
        raise ValueError("invalid Multiwalker chain")
    return tuple(
        neighbor
        for neighbor in (owner - 1, owner + 1)
        if 0 <= neighbor < walkers
    )


def _actor_action(actor: nn.Module, observation: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        value = actor(
            torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        ).squeeze(0)
    return value.cpu().numpy().astype(np.float32)


def _same_parameters(left: nn.Module, right: nn.Module) -> bool:
    return all(
        torch.equal(first.detach(), second.detach())
        for first, second in zip(left.parameters(), right.parameters())
    )


def _drift_owner(actor: MultiwalkerContractActor, event: int) -> None:
    final = actor.network[-2]
    if not isinstance(final, nn.Linear):
        raise AssertionError("contract actor layout changed")
    with torch.no_grad():
        increment = 0.0025 * torch.tensor(
            [
                np.sin(event + 0.3),
                np.cos(event + 0.7),
                np.sin(0.5 * event + 1.1),
                np.cos(0.25 * event + 0.2),
            ],
            dtype=final.bias.dtype,
        )
        final.bias.add_(increment)


def run_multiwalker_cache_contract(
    *,
    seeds: Sequence[int],
    walkers: int = 5,
    cycles: int = 40,
    actor_seed: int = 95200,
    selection_seed: int = 95201,
    delay_seed: int = 95202,
    maximum_delay: int = 3,
) -> MultiwalkerCacheContract:
    """Exercise cache semantics without using rewards as an outcome."""

    from pettingzoo.sisl import multiwalker_v9

    if not seeds or walkers < 2 or cycles <= 0 or maximum_delay < 0:
        raise ValueError("invalid Multiwalker contract configuration")
    launches = 0
    receipts = 0
    nonnull_refreshes = 0
    optional_policy_bytes = 0
    maximum_degree = 0
    action_bound_violations = 0
    nonfinite_observations = 0
    exact_cache_copy_failures = 0
    receipt_order_failures = 0
    changed_cached_actions = 0
    trace = hashlib.sha256()
    packet_id = 0

    for seed in seeds:
        actors = tuple(
            MultiwalkerContractActor(seed=actor_seed + index)
            for index in range(walkers)
        )
        for actor in actors:
            actor.eval()
        caches = PolicyCacheBank(actors)
        selection_rng = np.random.default_rng(selection_seed + int(seed))
        delay_rng = np.random.default_rng(delay_seed + int(seed))
        pending: list[tuple[int, int, int]] = []
        environment = multiwalker_v9.parallel_env(
            n_walkers=walkers,
            position_noise=1e-3,
            angle_noise=1e-3,
            shared_reward=True,
            max_cycles=cycles,
        )
        observations, _ = environment.reset(seed=int(seed))
        try:
            for event in range(cycles):
                while pending and pending[0][0] <= event:
                    receipt_event, launch_event, _ = heapq.heappop(pending)
                    receipts += 1
                    receipt_order_failures += int(receipt_event <= launch_event)
                owner = event % walkers
                neighbors = chain_neighbors(owner, walkers)
                maximum_degree = max(maximum_degree, len(neighbors))
                candidates: tuple[int | None, ...] = (None,) + neighbors
                selected = candidates[
                    int(selection_rng.integers(0, len(candidates)))
                ]
                agent_names = environment.possible_agents
                before_cached_action: np.ndarray | None = None
                before_current_action: np.ndarray | None = None
                if selected is not None:
                    donor_observation = observations[agent_names[selected]]
                    before_cached_action = _actor_action(
                        caches.cached_actor(owner, selected), donor_observation
                    )
                    before_current_action = _actor_action(
                        actors[selected], donor_observation
                    )
                    refresh = apply_refresh_action(
                        recipient=owner,
                        donors=(selected,),
                        actors=actors,
                        caches=caches,
                    )
                    expected_bytes = parameter_bytes(
                        tuple(actors[selected].parameters())
                    )
                    exact_cache_copy_failures += int(
                        int(refresh.optional_policy_bytes) != expected_bytes
                        or not _same_parameters(
                            actors[selected], caches.cached_actor(owner, selected)
                        )
                    )
                    optional_policy_bytes += int(refresh.optional_policy_bytes)
                    nonnull_refreshes += 1
                    refreshed_action = _actor_action(
                        caches.cached_actor(owner, selected), donor_observation
                    )
                    changed_cached_actions += int(
                        not np.allclose(before_cached_action, before_current_action)
                        and np.allclose(refreshed_action, before_current_action)
                    )
                actions: dict[str, np.ndarray] = {}
                for donor, agent in enumerate(agent_names):
                    actor = (
                        actors[donor]
                        if donor == owner
                        else caches.cached_actor(owner, donor)
                    )
                    action = _actor_action(actor, observations[agent])
                    action_bound_violations += int(
                        np.any(action < -1.0) or np.any(action > 1.0)
                    )
                    actions[agent] = action
                    trace.update(action.tobytes())
                observations, _, terms, truncations, _ = environment.step(actions)
                for observation in observations.values():
                    nonfinite_observations += int(
                        not np.all(np.isfinite(observation))
                    )
                    trace.update(np.asarray(observation, dtype=np.float32).tobytes())
                receipt_event = event + int(
                    delay_rng.integers(0, maximum_delay + 1)
                ) + 1
                heapq.heappush(pending, (receipt_event, event, packet_id))
                packet_id += 1
                launches += 1
                _drift_owner(actors[owner], event)
                caches.mark_owner_update(owner)
                if not environment.agents or any(terms.values()) or any(
                    truncations.values()
                ):
                    break
            while pending:
                receipt_event, launch_event, _ = heapq.heappop(pending)
                receipts += 1
                receipt_order_failures += int(receipt_event <= launch_event)
        finally:
            environment.close()

    return MultiwalkerCacheContract(
        seeds=tuple(int(seed) for seed in seeds),
        walkers=int(walkers),
        cycles=int(cycles),
        launches=int(launches),
        receipts=int(receipts),
        nonnull_refreshes=int(nonnull_refreshes),
        optional_policy_bytes=int(optional_policy_bytes),
        maximum_degree=int(maximum_degree),
        action_bound_violations=int(action_bound_violations),
        nonfinite_observations=int(nonfinite_observations),
        exact_cache_copy_failures=int(exact_cache_copy_failures),
        receipt_order_failures=int(receipt_order_failures),
        changed_cached_actions=int(changed_cached_actions),
        trace_sha256=trace.hexdigest(),
    )


def write_contract(path: Path, contract: MultiwalkerCacheContract) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(contract), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
