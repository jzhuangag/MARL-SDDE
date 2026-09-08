"""Exact finite-horizon budget oracle for the Multiwalker development audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from .audit_multiwalker_cache_contract import (
    MultiwalkerContractActor,
    _actor_action,
    chain_neighbors,
)
from .multiwalker_oracle_headroom_dev import (
    POLICIES,
    _drift_actor,
    _environment,
    counterfactual_profile_return,
)


ORIGINAL_ROWS_SHA256 = "AD79327D849C820C1062C88BB1502A2BB5756064D7EA23EBCFFE6203662B2D42"


@dataclass(frozen=True)
class LocalOption:
    owner: int
    cost_sequence: tuple[int, ...]
    action_sequence: tuple[tuple[int, ...], ...]
    total_h_return: float


@dataclass(frozen=True)
class ExactOracleRow:
    seed: int
    drift_scale: float
    budget_rate: float
    exact_oracle_h_return: float
    exact_oracle_spent_edges: int
    strong_online_policy: str
    strong_online_h_return: float
    no_refresh_h_return: float
    oracle_gain_over_no_refresh: float
    oracle_headroom_over_strong: float
    oracle_recovery_gap: float
    normalized_absolute_headroom: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_mip_gap: float
    selected_owner_count: int
    maximum_prefix_excess: int
    no_refresh_replay_error: float


def _public_trace(
    *, seed: int, drift_scale: float, walkers: int, events: int, horizon: int
) -> tuple[
    list[dict[str, np.ndarray]],
    list[list[MultiwalkerContractActor]],
    list[MultiwalkerContractActor],
]:
    actors = [
        MultiwalkerContractActor(seed=95200 + index) for index in range(walkers)
    ]
    initial = copy.deepcopy(actors)
    prefix: list[dict[str, np.ndarray]] = []
    snapshots: list[list[MultiwalkerContractActor]] = []
    environment = _environment(
        walkers=walkers, maximum_cycles=events + horizon + 2
    )
    observations, _ = environment.reset(seed=int(seed))
    try:
        for event in range(events):
            if not environment.agents:
                break
            snapshots.append(copy.deepcopy(actors))
            actions = {
                agent: _actor_action(actors[index], observations[agent])
                for index, agent in enumerate(environment.possible_agents)
            }
            prefix.append(actions)
            observations, _, _, _, _ = environment.step(
                {agent: actions[agent] for agent in environment.agents}
            )
            _drift_actor(actors[event % walkers], event, drift_scale)
    finally:
        environment.close()
    if len(prefix) != events or len(snapshots) != events:
        raise RuntimeError("public Multiwalker trace terminated before the frozen horizon")
    return prefix, snapshots, initial


def _cache_for_state(
    *,
    owner: int,
    neighbors: tuple[int, ...],
    state: tuple[int, ...],
    launch_events: Sequence[int],
    actor_snapshots: Sequence[Sequence[MultiwalkerContractActor]],
    initial_actors: Sequence[MultiwalkerContractActor],
) -> list[MultiwalkerContractActor]:
    cache = list(initial_actors)
    for position, local_refresh_index in enumerate(state):
        if local_refresh_index >= 0:
            global_event = launch_events[local_refresh_index]
            donor = neighbors[position]
            cache[donor] = actor_snapshots[global_event][donor]
    return cache


def local_owner_options(
    *,
    seed: int,
    drift_scale: float,
    owner: int,
    prefix_actions: Sequence[Mapping[str, np.ndarray]],
    actor_snapshots: Sequence[Sequence[MultiwalkerContractActor]],
    initial_actors: Sequence[MultiwalkerContractActor],
    walkers: int,
    events: int,
    horizon: int,
    discount: float,
) -> list[LocalOption]:
    neighbors = chain_neighbors(owner, walkers)
    launch_events = tuple(range(owner, events, walkers))
    actions = ((),) + tuple((donor,) for donor in neighbors)
    # State entries are local launch indices of the last refresh, or -1.
    dynamic: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[float, tuple[tuple[int, ...], ...]],
    ] = {(((-1,) * len(neighbors)), ()): (0.0, ())}
    maximum_cycles = events + horizon + 2
    for local_index, global_event in enumerate(launch_events):
        next_dynamic: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[float, tuple[tuple[int, ...], ...]],
        ] = {}
        value_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        for (state, cost_sequence), (total, action_sequence) in dynamic.items():
            recipient_cache = _cache_for_state(
                owner=owner,
                neighbors=neighbors,
                state=state,
                launch_events=launch_events,
                actor_snapshots=actor_snapshots,
                initial_actors=initial_actors,
            )
            for candidate in actions:
                query = (state, candidate)
                if query not in value_cache:
                    value_cache[query] = counterfactual_profile_return(
                        seed=seed,
                        prefix_actions=prefix_actions[:global_event],
                        actors=actor_snapshots[global_event],
                        recipient_cache=recipient_cache,
                        owner=owner,
                        refresh_donors=candidate,
                        horizon=horizon,
                        discount=discount,
                        walkers=walkers,
                        maximum_cycles=maximum_cycles,
                    )[0]
                next_state = list(state)
                for donor in candidate:
                    next_state[neighbors.index(donor)] = local_index
                next_cost = cost_sequence + (len(candidate),)
                key = (tuple(next_state), next_cost)
                proposal = (
                    total + value_cache[query],
                    action_sequence + (candidate,),
                )
                incumbent = next_dynamic.get(key)
                if incumbent is None or proposal[0] > incumbent[0]:
                    next_dynamic[key] = proposal
        dynamic = next_dynamic
    by_cost: dict[tuple[int, ...], LocalOption] = {}
    for (state, cost_sequence), (total, action_sequence) in dynamic.items():
        del state
        option = LocalOption(
            owner=owner,
            cost_sequence=cost_sequence,
            action_sequence=action_sequence,
            total_h_return=float(total),
        )
        incumbent = by_cost.get(cost_sequence)
        if incumbent is None or option.total_h_return > incumbent.total_h_return:
            by_cost[cost_sequence] = option
    return [by_cost[key] for key in sorted(by_cost)]


def solve_global_prefix_oracle(
    *,
    options_by_owner: Sequence[Sequence[LocalOption]],
    budget_rate: float,
    events: int,
    walkers: int,
) -> tuple[float, int, bool, int, float, tuple[LocalOption, ...]]:
    flat: list[LocalOption] = [option for options in options_by_owner for option in options]
    offsets = np.cumsum([0] + [len(options) for options in options_by_owner])
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower = np.full(walkers + events, -np.inf)
    upper = np.empty(walkers + events)
    for owner in range(walkers):
        lower[owner] = 1.0
        upper[owner] = 1.0
        for column in range(offsets[owner], offsets[owner + 1]):
            row_indices.append(owner)
            column_indices.append(column)
            values.append(1.0)
    for event in range(events):
        row = walkers + event
        upper[row] = math.floor(budget_rate * (event + 1))
        for column, option in enumerate(flat):
            launch_events = tuple(range(option.owner, events, walkers))
            cumulative = sum(
                cost
                for launch, cost in zip(launch_events, option.cost_sequence)
                if launch <= event
            )
            if cumulative:
                row_indices.append(row)
                column_indices.append(column)
                values.append(float(cumulative))
    constraint = LinearConstraint(
        csc_matrix(
            (values, (row_indices, column_indices)),
            shape=(walkers + events, len(flat)),
        ),
        lower,
        upper,
    )
    objective = -np.asarray([option.total_h_return for option in flat], dtype=float)
    result = milp(
        c=objective,
        integrality=np.ones(len(flat), dtype=np.int8),
        bounds=Bounds(np.zeros(len(flat)), np.ones(len(flat))),
        constraints=constraint,
        options={"presolve": True, "time_limit": 300.0, "mip_rel_gap": 0.0},
    )
    if result.x is None:
        return -math.inf, 0, False, int(result.status), math.inf, ()
    selected = tuple(flat[index] for index in np.flatnonzero(result.x > 0.5))
    total_return = sum(option.total_h_return for option in selected)
    spent = sum(sum(option.cost_sequence) for option in selected)
    gap_value = getattr(result, "mip_gap", None)
    return (
        float(total_return),
        int(spent),
        bool(result.success),
        int(result.status),
        float(gap_value) if gap_value is not None else math.inf,
        selected,
    )


def _load_original_rows(path: Path) -> list[dict[str, object]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if digest != ORIGINAL_ROWS_SHA256:
        raise ValueError("original development rows hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("original rows must be a list")
    return payload


def _original_cell(
    rows: Sequence[Mapping[str, object]], *, seed: int, drift: float, budget: float
) -> tuple[float, float, str]:
    matches = [
        row
        for row in rows
        if int(row["seed"]) == seed
        and float(row["drift_scale"]) == drift
        and float(row["budget_rate"]) == budget
    ]
    if {str(row["policy"]) for row in matches} != set(POLICIES):
        raise ValueError("original scenario policy family mismatch")
    by_policy = {str(row["policy"]): row for row in matches}
    online = [
        policy for policy in POLICIES if policy not in {"oracle_h8", "no_refresh"}
    ]
    strong_name = max(
        online,
        key=lambda policy: (float(by_policy[policy]["selected_h_return"]), policy),
    )
    return (
        float(by_policy[strong_name]["selected_h_return"]),
        float(by_policy["no_refresh"]["selected_h_return"]),
        strong_name,
    )


def run_amendment(
    *,
    original_rows_path: Path,
    seeds: Sequence[int],
    drift_scales: Sequence[float] = (0.01, 0.04),
    budget_rates: Sequence[float] = (0.25, 0.5),
    walkers: int = 5,
    events: int = 40,
    horizon: int = 8,
    discount: float = 0.99,
) -> list[ExactOracleRow]:
    original = _load_original_rows(original_rows_path)
    output: list[ExactOracleRow] = []
    for seed, drift in itertools.product(seeds, drift_scales):
        prefix, snapshots, initial = _public_trace(
            seed=seed,
            drift_scale=drift,
            walkers=walkers,
            events=events,
            horizon=horizon,
        )
        options_by_owner = [
            local_owner_options(
                seed=seed,
                drift_scale=drift,
                owner=owner,
                prefix_actions=prefix,
                actor_snapshots=snapshots,
                initial_actors=initial,
                walkers=walkers,
                events=events,
                horizon=horizon,
                discount=discount,
            )
            for owner in range(walkers)
        ]
        for budget in budget_rates:
            exact, spent, success, status, gap, selected = solve_global_prefix_oracle(
                options_by_owner=options_by_owner,
                budget_rate=budget,
                events=events,
                walkers=walkers,
            )
            strong, no_refresh, strong_name = _original_cell(
                original, seed=seed, drift=drift, budget=budget
            )
            zero_option_total = sum(
                next(
                    option.total_h_return
                    for option in options
                    if sum(option.cost_sequence) == 0
                )
                for options in options_by_owner
            )
            gain = exact - no_refresh
            headroom = exact - strong
            maximum_prefix_excess = max(
                sum(
                    cost
                    for option in selected
                    for launch, cost in zip(
                        range(option.owner, events, walkers), option.cost_sequence
                    )
                    if launch <= event
                )
                - math.floor(budget * (event + 1))
                for event in range(events)
            )
            output.append(
                ExactOracleRow(
                    seed=int(seed),
                    drift_scale=float(drift),
                    budget_rate=float(budget),
                    exact_oracle_h_return=float(exact),
                    exact_oracle_spent_edges=int(spent),
                    strong_online_policy=strong_name,
                    strong_online_h_return=float(strong),
                    no_refresh_h_return=float(no_refresh),
                    oracle_gain_over_no_refresh=float(gain),
                    oracle_headroom_over_strong=float(headroom),
                    oracle_recovery_gap=float(headroom / abs(gain)) if abs(gain) > 1e-12 else 0.0,
                    normalized_absolute_headroom=float(
                        headroom / max(abs(no_refresh), 1e-12)
                    ),
                    optimizer_success=bool(success),
                    optimizer_status=int(status),
                    optimizer_mip_gap=float(gap),
                    selected_owner_count=len(selected),
                    maximum_prefix_excess=int(maximum_prefix_excess),
                    no_refresh_replay_error=float(zero_option_total - no_refresh),
                )
            )
    return output


def analyze_amendment(rows: Sequence[ExactOracleRow]) -> dict[str, object]:
    active = [row for row in rows if row.drift_scale == 0.04]
    oracle_gain = sum(row.oracle_gain_over_no_refresh for row in active)
    strong_gain = sum(
        row.strong_online_h_return - row.no_refresh_h_return for row in active
    )
    headroom = oracle_gain - strong_gain
    recovery = headroom / abs(oracle_gain) if abs(oracle_gain) > 1e-12 else 0.0
    direction = float(np.mean([row.oracle_headroom_over_strong > 1e-10 for row in active]))
    normalized = float(np.median([row.normalized_absolute_headroom for row in active]))
    gates = {
        "A1_complete_finite": all(
            math.isfinite(value)
            for row in rows
            for value in (
                row.exact_oracle_h_return,
                row.oracle_headroom_over_strong,
                row.optimizer_mip_gap,
            )
        ),
        "A2_exact_milp_optimum": all(
            row.optimizer_success
            and row.optimizer_status == 0
            and row.optimizer_mip_gap <= 1e-9
            and row.selected_owner_count == 5
            for row in rows
        ),
        "A3_no_refresh_replay": all(
            abs(row.no_refresh_replay_error) <= 1e-10 for row in rows
        ),
        "A4_prefix_budget": all(row.maximum_prefix_excess <= 0 for row in rows),
        "D4_active_oracle_positive": oracle_gain > 0.0,
        "D5_active_recovery_gap": recovery >= 0.10,
        "D6_active_direction": direction >= 0.75,
        "D7_active_absolute_effect": normalized >= 0.001,
    }
    return {
        "status": "development_amendment_exact_budget_oracle",
        "rows": len(rows),
        "active_oracle_gain_over_no_refresh": oracle_gain,
        "active_strong_gain_over_no_refresh": strong_gain,
        "active_oracle_headroom_over_strong": headroom,
        "active_oracle_recovery_gap": recovery,
        "active_direction_rate": direction,
        "active_median_normalized_absolute_headroom": normalized,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "cell_metrics": [asdict(row) for row in rows],
    }


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def merge_exact_rows(paths: Sequence[Path]) -> list[ExactOracleRow]:
    rows: list[ExactOracleRow] = []
    keys: set[tuple[int, float, float]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("each exact-row chunk must be a list")
        for record in payload:
            row = ExactOracleRow(**record)
            key = (row.seed, row.drift_scale, row.budget_rate)
            if key in keys:
                raise ValueError("duplicate amended oracle cell")
            keys.add(key)
            rows.append(row)
    return sorted(rows, key=lambda row: (row.seed, row.drift_scale, row.budget_rate))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seeds", type=int, nargs="+")
    group.add_argument("--merge-inputs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.merge_inputs:
        rows = merge_exact_rows(args.merge_inputs)
    else:
        if args.original_rows is None:
            parser.error("--original-rows is required with --seeds")
        rows = run_amendment(
            original_rows_path=args.original_rows,
            seeds=args.seeds,
        )
    summary = analyze_amendment(rows)
    _write(args.output_dir / "exact_rows.json", [asdict(row) for row in rows])
    _write(args.output_dir / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "cell_metrics"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
