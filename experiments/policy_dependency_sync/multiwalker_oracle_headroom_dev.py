"""CPU development audit for Multiwalker policy-profile oracle headroom.

Counterfactual outcomes are available only to the privileged development
oracle and the explicitly labelled one-step oracle baseline.  No learned
controller consumes these labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .audit_multiwalker_cache_contract import (
    MultiwalkerContractActor,
    _actor_action,
    chain_neighbors,
)
from .neural_signed_cache import parameter_bytes


POLICIES = (
    "oracle_h8",
    "oracle_h1",
    "action_gap",
    "parameter_gap",
    "age",
    "round_robin",
    "random",
    "fixed_left",
    "fixed_right",
    "all_neighbors",
    "no_refresh",
)


@dataclass(frozen=True)
class ScenarioResult:
    seed: int
    drift_scale: float
    budget_rate: float
    policy: str
    selected_h_return: float
    no_action_h_return: float
    captured_immediate_gain: float
    spent_edges: int
    optional_policy_bytes: int
    positive_choices: int
    nonnull_choices: int
    launches: int
    replay_failures: int


def _environment(*, walkers: int, maximum_cycles: int):
    from pettingzoo.sisl import multiwalker_v9

    return multiwalker_v9.parallel_env(
        n_walkers=walkers,
        position_noise=1e-3,
        angle_noise=1e-3,
        shared_reward=True,
        max_cycles=maximum_cycles,
    )


def _drift_actor(actor: MultiwalkerContractActor, event: int, scale: float) -> None:
    final = actor.network[-2]
    if not isinstance(final, nn.Linear):
        raise AssertionError("development actor layout changed")
    with torch.no_grad():
        final.bias.add_(
            float(scale)
            * torch.tensor(
                [
                    np.sin(event + 0.3),
                    np.cos(event + 0.7),
                    np.sin(0.5 * event + 1.1),
                    np.cos(0.25 * event + 0.2),
                ],
                dtype=final.bias.dtype,
            )
        )


def _parameter_gap(current: nn.Module, cached: nn.Module) -> float:
    return float(
        math.sqrt(
            sum(
                float(torch.sum((left.detach() - right.detach()) ** 2))
                for left, right in zip(current.parameters(), cached.parameters())
            )
        )
    )


def _prefix_replay(
    *,
    seed: int,
    prefix_actions: Sequence[Mapping[str, np.ndarray]],
    walkers: int,
    maximum_cycles: int,
):
    environment = _environment(walkers=walkers, maximum_cycles=maximum_cycles)
    observations, _ = environment.reset(seed=int(seed))
    for actions in prefix_actions:
        if not environment.agents:
            break
        observations, _, _, _, _ = environment.step(
            {agent: actions[agent] for agent in environment.agents}
        )
    return environment, observations


def counterfactual_profile_return(
    *,
    seed: int,
    prefix_actions: Sequence[Mapping[str, np.ndarray]],
    actors: Sequence[MultiwalkerContractActor],
    recipient_cache: Sequence[MultiwalkerContractActor],
    owner: int,
    refresh_donors: Iterable[int],
    horizon: int,
    discount: float,
    walkers: int,
    maximum_cycles: int,
) -> tuple[float, float, str]:
    """Return H-step and one-step team values from an exactly replayed state."""

    if horizon <= 0 or not 0.0 < discount <= 1.0:
        raise ValueError("invalid counterfactual horizon")
    refreshed = frozenset(int(donor) for donor in refresh_donors)
    environment, observations = _prefix_replay(
        seed=seed,
        prefix_actions=prefix_actions,
        walkers=walkers,
        maximum_cycles=maximum_cycles,
    )
    trace = hashlib.sha256()
    total = 0.0
    first = 0.0
    try:
        for step in range(horizon):
            if not environment.agents:
                break
            actions: dict[str, np.ndarray] = {}
            for donor, agent in enumerate(environment.possible_agents):
                selected_actor = (
                    actors[donor]
                    if donor == owner or donor in refreshed
                    else recipient_cache[donor]
                )
                action = _actor_action(selected_actor, observations[agent])
                actions[agent] = action
                trace.update(action.tobytes())
            observations, rewards, terms, truncations, _ = environment.step(
                {agent: actions[agent] for agent in environment.agents}
            )
            team_reward = float(np.mean(tuple(rewards.values()))) if rewards else 0.0
            if step == 0:
                first = team_reward
            total += discount**step * team_reward
            for agent in sorted(observations):
                trace.update(np.asarray(observations[agent], dtype=np.float32).tobytes())
            if not environment.agents or any(terms.values()) or any(truncations.values()):
                break
    finally:
        environment.close()
    return float(total), float(first), trace.hexdigest()


def _candidate_key(candidate: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    return (len(candidate), candidate)


def _best_value_candidate(
    values: Mapping[tuple[int, ...], float],
    allowed: Sequence[tuple[int, ...]],
) -> tuple[int, ...]:
    return min(
        allowed,
        key=lambda candidate: (-float(values[candidate]), _candidate_key(candidate)),
    )


def _select_candidate(
    *,
    policy: str,
    owner: int,
    event: int,
    capacity: int,
    neighbors: tuple[int, ...],
    h_values: Mapping[tuple[int, ...], float],
    h1_values: Mapping[tuple[int, ...], float],
    actors: Sequence[MultiwalkerContractActor],
    cache: Sequence[Sequence[MultiwalkerContractActor]],
    observations: Mapping[str, np.ndarray],
    last_refresh: Mapping[tuple[int, int], int],
    random_generator: np.random.Generator,
) -> tuple[int, ...]:
    if policy not in POLICIES or capacity <= 0 or not neighbors:
        return ()
    one_edges = tuple((donor,) for donor in neighbors if capacity >= 1)
    allowed = ((),) + one_edges
    if policy == "oracle_h8":
        selected = _best_value_candidate(h_values, allowed)
        return selected if h_values[selected] > h_values[()] else ()
    if policy == "oracle_h1":
        selected = _best_value_candidate(h1_values, allowed)
        return selected if h1_values[selected] > h1_values[()] else ()
    if policy == "all_neighbors":
        candidate = tuple(neighbors)
        return candidate if len(candidate) <= capacity else ()
    if policy == "no_refresh":
        return ()
    if policy == "round_robin":
        return (neighbors[event % len(neighbors)],)
    if policy == "random":
        return (neighbors[int(random_generator.integers(0, len(neighbors)))],)
    if policy == "fixed_left":
        return (min(neighbors),)
    if policy == "fixed_right":
        return (max(neighbors),)
    if policy == "age":
        return (
            max(neighbors, key=lambda donor: (event - last_refresh[(owner, donor)], -donor)),
        )
    if policy == "parameter_gap":
        return (
            max(
                neighbors,
                key=lambda donor: (
                    _parameter_gap(actors[donor], cache[owner][donor]),
                    -donor,
                ),
            ),
        )
    if policy == "action_gap":
        names = tuple(sorted(observations, key=lambda name: int(name.split("_")[-1])))
        return (
            max(
                neighbors,
                key=lambda donor: (
                    float(
                        np.linalg.norm(
                            _actor_action(actors[donor], observations[names[donor]])
                            - _actor_action(
                                cache[owner][donor], observations[names[donor]]
                            )
                        )
                    ),
                    -donor,
                ),
            ),
        )
    raise AssertionError("unreachable policy")


def run_scenario(
    *,
    seed: int,
    drift_scale: float,
    budget_rate: float,
    policy: str,
    walkers: int = 5,
    events: int = 40,
    horizon: int = 8,
    discount: float = 0.99,
    actor_seed: int = 95200,
    random_seed: int = 95800,
) -> ScenarioResult:
    if policy not in POLICIES or events <= 0 or horizon <= 0:
        raise ValueError("invalid development scenario")
    maximum_cycles = events + horizon + 2
    actors = [
        MultiwalkerContractActor(seed=actor_seed + index) for index in range(walkers)
    ]
    cache = [copy.deepcopy(actors) for _ in range(walkers)]
    last_refresh = {
        (owner, donor): -1
        for owner in range(walkers)
        for donor in chain_neighbors(owner, walkers)
    }
    environment = _environment(walkers=walkers, maximum_cycles=maximum_cycles)
    observations, _ = environment.reset(seed=int(seed))
    prefix_actions: list[dict[str, np.ndarray]] = []
    random_generator = np.random.default_rng(
        int(random_seed) + 101 * int(seed) + POLICIES.index(policy)
    )
    actor_payload = parameter_bytes(tuple(actors[0].parameters()))
    selected_h_return = 0.0
    no_action_h_return = 0.0
    captured_immediate_gain = 0.0
    spent_edges = 0
    positive_choices = 0
    nonnull_choices = 0
    replay_failures = 0
    launches = 0
    try:
        for event in range(events):
            if not environment.agents:
                break
            owner = event % walkers
            neighbors = chain_neighbors(owner, walkers)
            candidates = ((),) + tuple((donor,) for donor in neighbors) + (
                (tuple(neighbors),) if len(neighbors) > 1 else ()
            )
            h_values: dict[tuple[int, ...], float] = {}
            h1_values: dict[tuple[int, ...], float] = {}
            trace_by_candidate: dict[tuple[int, ...], str] = {}
            for candidate in candidates:
                h_value, h1_value, branch_trace = counterfactual_profile_return(
                    seed=seed,
                    prefix_actions=prefix_actions,
                    actors=actors,
                    recipient_cache=cache[owner],
                    owner=owner,
                    refresh_donors=candidate,
                    horizon=horizon,
                    discount=discount,
                    walkers=walkers,
                    maximum_cycles=maximum_cycles,
                )
                h_values[candidate] = h_value
                h1_values[candidate] = h1_value
                trace_by_candidate[candidate] = branch_trace
            repeat_null = counterfactual_profile_return(
                seed=seed,
                prefix_actions=prefix_actions,
                actors=actors,
                recipient_cache=cache[owner],
                owner=owner,
                refresh_donors=(),
                horizon=horizon,
                discount=discount,
                walkers=walkers,
                maximum_cycles=maximum_cycles,
            )
            replay_failures += int(
                repeat_null[0] != h_values[()]
                or repeat_null[1] != h1_values[()]
                or repeat_null[2] != trace_by_candidate[()]
            )
            capacity = max(0, math.floor(budget_rate * (event + 1)) - spent_edges)
            selected = _select_candidate(
                policy=policy,
                owner=owner,
                event=event,
                capacity=capacity,
                neighbors=neighbors,
                h_values=h_values,
                h1_values=h1_values,
                actors=actors,
                cache=cache,
                observations=observations,
                last_refresh=last_refresh,
                random_generator=random_generator,
            )
            if len(selected) > capacity:
                raise AssertionError("policy exceeded its prefix edge budget")
            selected_h_return += h_values[selected]
            no_action_h_return += h_values[()]
            captured_immediate_gain += h_values[selected] - h_values[()]
            positive_choices += int(h_values[selected] > h_values[()])
            nonnull_choices += int(bool(selected))
            spent_edges += len(selected)
            for donor in selected:
                cache[owner][donor] = copy.deepcopy(actors[donor])
                last_refresh[(owner, donor)] = event

            behavior_actions = {
                agent: _actor_action(actors[index], observations[agent])
                for index, agent in enumerate(environment.possible_agents)
            }
            prefix_actions.append(behavior_actions)
            observations, _, _, _, _ = environment.step(
                {agent: behavior_actions[agent] for agent in environment.agents}
            )
            _drift_actor(actors[owner], event, drift_scale)
            launches += 1
    finally:
        environment.close()
    return ScenarioResult(
        seed=int(seed),
        drift_scale=float(drift_scale),
        budget_rate=float(budget_rate),
        policy=policy,
        selected_h_return=float(selected_h_return),
        no_action_h_return=float(no_action_h_return),
        captured_immediate_gain=float(captured_immediate_gain),
        spent_edges=int(spent_edges),
        optional_policy_bytes=int(spent_edges * actor_payload),
        positive_choices=int(positive_choices),
        nonnull_choices=int(nonnull_choices),
        launches=int(launches),
        replay_failures=int(replay_failures),
    )


def analyze(rows: Sequence[ScenarioResult]) -> dict[str, object]:
    if not rows:
        raise ValueError("empty Multiwalker headroom table")
    grouped: dict[tuple[int, float, float], dict[str, ScenarioResult]] = {}
    for row in rows:
        grouped.setdefault((row.seed, row.drift_scale, row.budget_rate), {})[
            row.policy
        ] = row
    cell_metrics: list[dict[str, object]] = []
    for key, policy_rows in sorted(grouped.items()):
        if set(policy_rows) != set(POLICIES):
            raise ValueError("incomplete policy family")
        oracle = policy_rows["oracle_h8"]
        no_refresh = policy_rows["no_refresh"]
        online_names = tuple(
            policy for policy in POLICIES if policy not in {"oracle_h8", "no_refresh"}
        )
        strong_name = max(
            online_names,
            key=lambda policy: (policy_rows[policy].selected_h_return, policy),
        )
        strong = policy_rows[strong_name]
        oracle_gain = oracle.selected_h_return - no_refresh.selected_h_return
        strong_gain = strong.selected_h_return - no_refresh.selected_h_return
        headroom = oracle.selected_h_return - strong.selected_h_return
        recovery_gap = (
            headroom / abs(oracle_gain) if abs(oracle_gain) > 1e-12 else 0.0
        )
        scale = max(abs(no_refresh.selected_h_return), 1e-12)
        cell_metrics.append(
            {
                "seed": key[0],
                "drift_scale": key[1],
                "budget_rate": key[2],
                "strong_online_policy": strong_name,
                "oracle_gain_over_no_refresh": oracle_gain,
                "strong_gain_over_no_refresh": strong_gain,
                "oracle_headroom_over_strong": headroom,
                "oracle_recovery_gap": recovery_gap,
                "normalized_absolute_headroom": headroom / scale,
                "oracle_spent_edges": oracle.spent_edges,
                "strong_spent_edges": strong.spent_edges,
            }
        )
    active = [metric for metric in cell_metrics if metric["drift_scale"] == 0.04]
    oracle_gain = sum(float(metric["oracle_gain_over_no_refresh"]) for metric in active)
    strong_gain = sum(float(metric["strong_gain_over_no_refresh"]) for metric in active)
    headroom = oracle_gain - strong_gain
    recovery_gap = headroom / abs(oracle_gain) if abs(oracle_gain) > 1e-12 else 0.0
    direction = float(
        np.mean(
            [float(metric["oracle_headroom_over_strong"]) > 0.0 for metric in active]
        )
    )
    normalized = float(
        np.median([float(metric["normalized_absolute_headroom"]) for metric in active])
    )
    gates = {
        "D1_complete_finite": all(
            math.isfinite(row.selected_h_return)
            and math.isfinite(row.no_action_h_return)
            for row in rows
        ),
        "D2_exact_replay": all(row.replay_failures == 0 for row in rows),
        "D3_prefix_budget": all(
            row.spent_edges <= math.floor(row.budget_rate * row.launches)
            for row in rows
        ),
        "D4_active_oracle_positive": oracle_gain > 0.0,
        "D5_active_recovery_gap": recovery_gap >= 0.10,
        "D6_active_direction": direction >= 0.75,
        "D7_active_absolute_effect": normalized >= 0.001,
    }
    return {
        "status": "development_only_counterfactual_oracle_audit",
        "policies": list(POLICIES),
        "rows": len(rows),
        "cells": len(cell_metrics),
        "active_drift_scale": 0.04,
        "active_oracle_gain_over_no_refresh": oracle_gain,
        "active_strong_gain_over_no_refresh": strong_gain,
        "active_oracle_headroom_over_strong": headroom,
        "active_oracle_recovery_gap": recovery_gap,
        "active_direction_rate": direction,
        "active_median_normalized_absolute_headroom": normalized,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "cell_metrics": cell_metrics,
    }


def run_development(
    *,
    seeds: Sequence[int],
    drift_scales: Sequence[float],
    budget_rates: Sequence[float],
    events: int,
    horizon: int,
) -> tuple[list[ScenarioResult], dict[str, object]]:
    rows = [
        run_scenario(
            seed=seed,
            drift_scale=drift_scale,
            budget_rate=budget_rate,
            policy=policy,
            events=events,
            horizon=horizon,
        )
        for seed in seeds
        for drift_scale in drift_scales
        for budget_rate in budget_rates
        for policy in POLICIES
    ]
    return rows, analyze(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(95700, 95708)))
    parser.add_argument("--events", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    rows, summary = run_development(
        seeds=args.seeds,
        drift_scales=(0.01, 0.04),
        budget_rates=(0.25, 0.5),
        events=args.events,
        horizon=args.horizon,
    )
    _write_json(args.output_dir / "rows.json", [asdict(row) for row in rows])
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "cell_metrics"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
