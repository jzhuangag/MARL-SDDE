"""Outcome-free audit that drives Pursuit with owner-specific policy caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .async_pistonball_ctde import PolicyCacheBank, apply_refresh_action
from .audit_pursuit_cache_semantics import PursuitInterfaceActor, _pursuit_model
from .audit_pursuit_local_factorization import bounded_local_graph
from .pursuit_cached_rollout import pursuit_cached_profile_actions


def _force_current_action(actor: torch.nn.Module, action: int) -> None:
    final = actor.network[-1]
    if not isinstance(final, torch.nn.Linear):
        raise TypeError("the audit actor requires a final linear layer")
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(-10.0)
        final.bias[int(action)] = 10.0


def run_audit(
    *,
    seeds: Sequence[int],
    cycles: int = 40,
    n_pursuers: int = 8,
    n_evaders: int = 30,
    observation_range: int = 7,
    max_neighbors: int = 4,
) -> dict[str, object]:
    from pettingzoo.sisl import pursuit_v4

    torch.manual_seed(81425)
    actors = tuple(PursuitInterfaceActor() for _ in range(n_pursuers))
    caches = PolicyCacheBank(actors)
    profile_steps = 0
    refreshes = 0
    charged_bytes = 0
    exact_refreshed_actions = 0
    changed_refreshed_actions = 0
    graph_changes = 0

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
        previous_graph: tuple[tuple[int, int], ...] | None = None
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
                if previous_graph is not None:
                    graph_changes += int(graph != previous_graph)
                previous_graph = graph
                owner = event % n_pursuers
                before = pursuit_cached_profile_actions(
                    observations=observations,
                    possible_agents=environment.possible_agents,
                    owner=owner,
                    actors=actors,
                    caches=caches,
                    device=torch.device("cpu"),
                )
                eligible = tuple(donor for edge_owner, donor in graph if edge_owner == owner)
                if eligible:
                    donor = eligible[0]
                    forced_action = (event + donor + 1) % 5
                    _force_current_action(actors[donor], forced_action)
                    caches.mark_owner_update(donor)
                    action = apply_refresh_action(
                        recipient=owner,
                        donors=(donor,),
                        actors=actors,
                        caches=caches,
                    )
                    after = pursuit_cached_profile_actions(
                        observations=observations,
                        possible_agents=environment.possible_agents,
                        owner=owner,
                        actors=actors,
                        caches=caches,
                        device=torch.device("cpu"),
                    )
                    agent = environment.possible_agents[donor]
                    exact_refreshed_actions += int(after[agent] == forced_action)
                    changed_refreshed_actions += int(before[agent] != after[agent])
                    refreshes += 1
                    charged_bytes += action.optional_policy_bytes
                    launch_actions = after
                else:
                    launch_actions = before
                if set(launch_actions) != set(environment.agents):
                    raise AssertionError("cached profile did not cover all live agents")
                observations, _rewards, terms, truncations, _infos = environment.step(
                    launch_actions
                )
                profile_steps += 1
                if not environment.agents or any(terms.values()) or any(
                    truncations.values()
                ):
                    break
        finally:
            environment.close()

    gates = {
        "B1_cached_profile_drives_every_step": profile_steps > 0,
        "B2_refreshes_exercised_and_charged": refreshes > 0 and charged_bytes > 0,
        "B3_refreshed_action_matches_current": exact_refreshed_actions == refreshes,
        "B4_refresh_changes_realized_behavior": changed_refreshed_actions > 0,
        "B5_dynamic_graph_exercised": graph_changes > 0,
    }
    return {
        "audit": "PURSUIT-CACHED-ROLLOUT-QUALIFICATION",
        "scientific_efficacy_evidence": False,
        "reward_values_read": False,
        "passed": all(gates.values()),
        "gates": gates,
        "configuration": {
            "seeds": list(map(int, seeds)),
            "cycles": int(cycles),
            "n_pursuers": int(n_pursuers),
            "n_evaders": int(n_evaders),
            "observation_range": int(observation_range),
            "max_neighbors": int(max_neighbors),
        },
        "profile_steps": profile_steps,
        "refreshes": refreshes,
        "charged_policy_bytes": charged_bytes,
        "exact_refreshed_actions": exact_refreshed_actions,
        "changed_refreshed_actions": changed_refreshed_actions,
        "graph_changes": graph_changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(seeds=tuple(range(92200, 92208)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
