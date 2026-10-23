"""Development-only bridge from exact cache value to composite Lyapunov launch control.

This runner deliberately reuses the independently authenticated Multiwalker
oracle-confirmation population as *design data*.  It does not create
confirmatory evidence.  The frozen public prefix is retained so that the only
new question is whether an exact fixed-universe cache-reset potential can
recover intertemporal headroom left by a privileged H-step greedy scheduler.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .audit_multiwalker_cache_contract import (
    MultiwalkerContractActor,
    _actor_action,
    chain_neighbors,
)
from .multiwalker_oracle_headroom_dev import (
    POLICIES,
    _drift_actor,
    _environment,
    _parameter_gap,
    counterfactual_profile_return,
)
from .paired_freshness_control import communication_queue_update


DESIGN_SEEDS = tuple(range(96100, 96108))
DRIFT_SCALES = (0.01, 0.04)
BUDGET_RATES = (0.25, 0.5)
RESET_BONUS_CAPS = (0.0, 0.025, 0.05, 0.1, 0.2)
QUEUE_STEPS = (0.0025, 0.01)
EVENTS = 40
HORIZON = 8
WALKERS = 5
DISCOUNT = 0.99
MAXIMUM_DRIFT_SCALE = max(DRIFT_SCALES)
CACHE_RADIUS_UPPER = 16.0 * MAXIMUM_DRIFT_SCALE

BASELINE_ROWS_SHA256 = (
    "0F523DC6BA9474BE008533E1FAA90E6A8700CEF246970C6FAD9823080F313ECC"
)
EXACT_ROWS_SHA256 = (
    "4CD2556B605FBD63659100165FD2A95C39FAE87BE551999B697A2A93046F0FF5"
)
STRONG_POLICIES = tuple(policy for policy in POLICIES if policy != "no_refresh")


@dataclass(frozen=True)
class PotentialRow:
    seed: int
    drift_scale: float
    budget_rate: float
    reset_bonus_cap: float
    queue_step: float
    selected_h_return: float
    no_refresh_h_return: float
    spent_edges: int
    launches: int
    nonnull_choices: int
    positive_utility_choices: int
    queue_changed_choices: int
    maximum_queue: float
    maximum_prefix_excess: int
    replay_failures: int
    maximum_reset_benefit: float
    maximum_cache_radius: float


def _canonical_write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _authenticated_records(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if digest != expected_sha256:
        raise ValueError(f"authenticated design artifact hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("authenticated design artifact must contain a list")
    return payload


def cache_radius_upper(*, events: int, walkers: int, maximum_drift: float) -> float:
    """Analytic radius for the registered four-coordinate bounded drift schedule."""

    if events <= 0 or walkers <= 0 or maximum_drift <= 0.0:
        raise ValueError("radius inputs must be positive")
    owner_updates = math.ceil(events / walkers)
    return float(2.0 * owner_updates * maximum_drift)


def reset_benefit(
    *, parameter_gap: float, reset_bonus_cap: float, cache_radius: float
) -> float:
    """Exact quadratic cache reset with outcome-free fixed coefficient beta."""

    if min(parameter_gap, reset_bonus_cap) < 0.0 or cache_radius <= 0.0:
        raise ValueError("invalid reset-benefit inputs")
    beta = 2.0 * reset_bonus_cap / cache_radius**2
    return float(0.5 * beta * parameter_gap**2)


def _select(
    *,
    h_values: Mapping[tuple[int, ...], float],
    reset_by_candidate: Mapping[tuple[int, ...], float],
    capacity: int,
    queue: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    allowed = tuple(candidate for candidate in h_values if len(candidate) <= capacity)
    if not allowed:
        raise ValueError("null action must remain feasible")

    def key(candidate: tuple[int, ...], use_queue: bool) -> tuple[float, int, tuple[int, ...]]:
        score = (
            float(h_values[candidate])
            + float(reset_by_candidate[candidate])
            - (queue * len(candidate) if use_queue else 0.0)
        )
        return (-score, len(candidate), candidate)

    selected = min(allowed, key=lambda candidate: key(candidate, True))
    unpriced = min(allowed, key=lambda candidate: key(candidate, False))
    return selected, unpriced


def run_scenario(
    *,
    seed: int,
    drift_scale: float,
    budget_rate: float,
    reset_bonus_cap: float,
    queue_step: float,
    walkers: int = WALKERS,
    events: int = EVENTS,
    horizon: int = HORIZON,
    discount: float = DISCOUNT,
) -> PotentialRow:
    """Run one composite-potential policy on the frozen all-current public prefix."""

    if (
        seed not in DESIGN_SEEDS
        or drift_scale not in DRIFT_SCALES
        or budget_rate not in BUDGET_RATES
        or reset_bonus_cap not in RESET_BONUS_CAPS
        or queue_step not in QUEUE_STEPS
    ):
        raise ValueError("scenario is outside the frozen development grid")
    radius = cache_radius_upper(
        events=events, walkers=walkers, maximum_drift=MAXIMUM_DRIFT_SCALE
    )
    actors = [MultiwalkerContractActor(seed=95200 + index) for index in range(walkers)]
    cache = [copy.deepcopy(actors) for _ in range(walkers)]
    environment = _environment(walkers=walkers, maximum_cycles=events + horizon + 2)
    observations, _ = environment.reset(seed=int(seed))
    prefix_actions: list[dict[str, np.ndarray]] = []
    selected_h_return = 0.0
    no_refresh_h_return = 0.0
    spent_edges = 0
    queue = 0.0
    maximum_queue = 0.0
    nonnull_choices = 0
    positive_utility_choices = 0
    queue_changed_choices = 0
    maximum_prefix_excess = 0
    replay_failures = 0
    maximum_reset = 0.0
    maximum_radius = 0.0
    launches = 0
    try:
        for event in range(events):
            if not environment.agents:
                break
            owner = event % walkers
            neighbors = chain_neighbors(owner, walkers)
            candidates = ((),) + tuple((donor,) for donor in neighbors)
            h_values: dict[tuple[int, ...], float] = {}
            traces: dict[tuple[int, ...], str] = {}
            reset_by_candidate: dict[tuple[int, ...], float] = {(): 0.0}
            for candidate in candidates:
                value, _, trace = counterfactual_profile_return(
                    seed=seed,
                    prefix_actions=prefix_actions,
                    actors=actors,
                    recipient_cache=cache[owner],
                    owner=owner,
                    refresh_donors=candidate,
                    horizon=horizon,
                    discount=discount,
                    walkers=walkers,
                    maximum_cycles=events + horizon + 2,
                )
                h_values[candidate] = value
                traces[candidate] = trace
                if candidate:
                    gap = _parameter_gap(actors[candidate[0]], cache[owner][candidate[0]])
                    maximum_radius = max(maximum_radius, gap)
                    benefit = reset_benefit(
                        parameter_gap=gap,
                        reset_bonus_cap=reset_bonus_cap,
                        cache_radius=radius,
                    )
                    if benefit > reset_bonus_cap + 1e-12:
                        raise AssertionError("analytic cache radius did not bound reset")
                    reset_by_candidate[candidate] = benefit
                    maximum_reset = max(maximum_reset, benefit)
            repeat = counterfactual_profile_return(
                seed=seed,
                prefix_actions=prefix_actions,
                actors=actors,
                recipient_cache=cache[owner],
                owner=owner,
                refresh_donors=(),
                horizon=horizon,
                discount=discount,
                walkers=walkers,
                maximum_cycles=events + horizon + 2,
            )
            replay_failures += int(repeat[0] != h_values[()] or repeat[2] != traces[()])
            capacity = max(0, math.floor(budget_rate * (event + 1)) - spent_edges)
            selected, unpriced = _select(
                h_values=h_values,
                reset_by_candidate=reset_by_candidate,
                capacity=capacity,
                queue=queue,
            )
            selected_h_return += h_values[selected]
            no_refresh_h_return += h_values[()]
            positive_utility_choices += int(h_values[selected] > h_values[()])
            nonnull_choices += int(bool(selected))
            queue_changed_choices += int(selected != unpriced)
            spent_edges += len(selected)
            maximum_prefix_excess = max(
                maximum_prefix_excess,
                spent_edges - math.floor(budget_rate * (event + 1)),
            )
            for donor in selected:
                cache[owner][donor] = copy.deepcopy(actors[donor])
            queue = communication_queue_update(
                queue=queue,
                queue_step=queue_step,
                realized_cost=float(len(selected)),
                average_budget=float(budget_rate),
            )
            maximum_queue = max(maximum_queue, queue)

            # The prior oracle-confirmation population used this immutable,
            # all-current public prefix.  Cache-dependent behavior is reserved
            # for a fresh-seed gate if this bridge passes.
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
    return PotentialRow(
        seed=int(seed),
        drift_scale=float(drift_scale),
        budget_rate=float(budget_rate),
        reset_bonus_cap=float(reset_bonus_cap),
        queue_step=float(queue_step),
        selected_h_return=float(selected_h_return),
        no_refresh_h_return=float(no_refresh_h_return),
        spent_edges=int(spent_edges),
        launches=int(launches),
        nonnull_choices=int(nonnull_choices),
        positive_utility_choices=int(positive_utility_choices),
        queue_changed_choices=int(queue_changed_choices),
        maximum_queue=float(maximum_queue),
        maximum_prefix_excess=int(maximum_prefix_excess),
        replay_failures=int(replay_failures),
        maximum_reset_benefit=float(maximum_reset),
        maximum_cache_radius=float(maximum_radius),
    )


def crossfit_configuration(
    rows: Sequence[PotentialRow], *, heldout_seed: int
) -> tuple[float, float]:
    """Select one global pair without reading the held-out seed."""

    training = [row for row in rows if row.seed != heldout_seed]
    expected = {
        (bonus, step) for bonus in RESET_BONUS_CAPS for step in QUEUE_STEPS
    }
    observed = {(row.reset_bonus_cap, row.queue_step) for row in training}
    if observed != expected:
        raise ValueError("cross-fit training catalogue is incomplete")
    totals = {
        pair: sum(
            row.selected_h_return
            for row in training
            if (row.reset_bonus_cap, row.queue_step) == pair
        )
        for pair in expected
    }
    return min(expected, key=lambda pair: (-totals[pair], pair[0], pair[1]))


def _baseline_maps(
    records: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[int, float, float], float],
    dict[tuple[int, float, float], float],
]:
    grouped: dict[tuple[int, float, float], dict[str, float]] = {}
    for record in records:
        key = (
            int(record["seed"]),
            float(record["drift_scale"]),
            float(record["budget_rate"]),
        )
        grouped.setdefault(key, {})[str(record["policy"])] = float(
            record["selected_h_return"]
        )
    strong: dict[tuple[int, float, float], float] = {}
    no_refresh: dict[tuple[int, float, float], float] = {}
    for key, by_policy in grouped.items():
        if set(by_policy) != set(POLICIES):
            raise ValueError("authenticated baseline family is incomplete")
        strong[key] = max(by_policy[policy] for policy in STRONG_POLICIES)
        no_refresh[key] = by_policy["no_refresh"]
    return strong, no_refresh


def analyze(
    rows: Sequence[PotentialRow],
    *,
    baseline_records: Sequence[Mapping[str, object]],
    exact_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected_count = (
        len(DESIGN_SEEDS)
        * len(DRIFT_SCALES)
        * len(BUDGET_RATES)
        * len(RESET_BONUS_CAPS)
        * len(QUEUE_STEPS)
    )
    baseline, no_refresh = _baseline_maps(baseline_records)
    exact = {
        (
            int(record["seed"]),
            float(record["drift_scale"]),
            float(record["budget_rate"]),
        ): float(record["exact_oracle_h_return"])
        for record in exact_records
    }
    selected_pairs = {
        seed: crossfit_configuration(rows, heldout_seed=seed) for seed in DESIGN_SEEDS
    }
    row_map = {
        (
            row.seed,
            row.drift_scale,
            row.budget_rate,
            row.reset_bonus_cap,
            row.queue_step,
        ): row
        for row in rows
    }
    cells: list[dict[str, object]] = []
    for seed in DESIGN_SEEDS:
        bonus, step = selected_pairs[seed]
        for drift in DRIFT_SCALES:
            for budget in BUDGET_RATES:
                key = (seed, drift, budget)
                row = row_map[(*key, bonus, step)]
                strong = baseline[key]
                oracle = exact[key]
                cells.append(
                    {
                        "seed": seed,
                        "drift_scale": drift,
                        "budget_rate": budget,
                        "selected_reset_bonus_cap": bonus,
                        "selected_queue_step": step,
                        "candidate_h_return": row.selected_h_return,
                        "strong_h_return": strong,
                        "no_refresh_h_return": no_refresh[key],
                        "exact_oracle_h_return": oracle,
                        "candidate_headroom": row.selected_h_return - strong,
                        "oracle_headroom": oracle - strong,
                        "normalized_candidate_headroom": (
                            (row.selected_h_return - strong)
                            / max(abs(row.no_refresh_h_return), 1e-12)
                        ),
                        "queue_changed_choices": row.queue_changed_choices,
                        "nonnull_choices": row.nonnull_choices,
                    }
                )
    active = [cell for cell in cells if cell["drift_scale"] == 0.04]
    candidate_headroom = sum(float(cell["candidate_headroom"]) for cell in active)
    oracle_headroom = sum(float(cell["oracle_headroom"]) for cell in active)
    normalized = sorted(float(cell["normalized_candidate_headroom"]) for cell in active)
    median_normalized = float(np.median(normalized))
    direction = sum(float(cell["candidate_headroom"]) > 0.0 for cell in active) / len(active)
    recovery = candidate_headroom / oracle_headroom if oracle_headroom > 0.0 else -math.inf
    gates = {
        "P1_complete_grid": len(rows) == expected_count and len(row_map) == expected_count,
        "P2_finite_exact_replay_prefix": all(
            math.isfinite(row.selected_h_return)
            and math.isfinite(row.maximum_queue)
            and row.replay_failures == 0
            and row.maximum_prefix_excess <= 0
            and row.maximum_cache_radius <= CACHE_RADIUS_UPPER + 1e-12
            and row.maximum_reset_benefit <= row.reset_bonus_cap + 1e-12
            for row in rows
        ),
        "P3_crossfit_catalogue_only": all(
            pair[0] in RESET_BONUS_CAPS and pair[1] in QUEUE_STEPS
            for pair in selected_pairs.values()
        ),
        "P4_strengthened_oracle_headroom_positive": oracle_headroom > 0.0,
        "P5_active_recovery_at_least_10pct": recovery >= 0.10,
        "P6_active_direction_at_least_75pct": direction >= 0.75,
        "P7_active_median_normalized_headroom_at_least_0p05pct": (
            median_normalized >= 0.0005
        ),
        "P8_nontrivial_reset_selected": all(pair[0] > 0.0 for pair in selected_pairs.values()),
        "P9_queue_changes_at_least_one_active_choice": sum(
            int(cell["queue_changed_choices"]) for cell in active
        ) > 0,
        "P10_candidate_aggregate_gain_over_no_refresh": sum(
            float(cell["candidate_h_return"] - cell["no_refresh_h_return"])
            for cell in cells
        ) > 0.0,
    }
    return {
        "status": "multiwalker_paired_potential_development",
        "design_seeds": list(DESIGN_SEEDS),
        "rows": len(rows),
        "selected_pair_by_heldout_seed": {
            str(seed): {"reset_bonus_cap": pair[0], "queue_step": pair[1]}
            for seed, pair in selected_pairs.items()
        },
        "active_candidate_headroom_over_strengthened_strong": candidate_headroom,
        "active_exact_oracle_headroom_over_strengthened_strong": oracle_headroom,
        "active_recovery_fraction": recovery,
        "active_direction_rate": direction,
        "active_median_normalized_candidate_headroom": median_normalized,
        "active_queue_changed_choices": sum(
            int(cell["queue_changed_choices"]) for cell in active
        ),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "cells": cells,
    }


def run_chunk(output_dir: Path, seeds: Sequence[int]) -> None:
    normalized = tuple(sorted(set(int(seed) for seed in seeds)))
    if not normalized or len(normalized) != len(seeds):
        raise ValueError("chunk seeds must be nonempty and unique")
    if not set(normalized).issubset(DESIGN_SEEDS):
        raise ValueError("chunk includes a seed outside the frozen registry")
    rows = [
        run_scenario(
            seed=seed,
            drift_scale=drift,
            budget_rate=budget,
            reset_bonus_cap=bonus,
            queue_step=step,
        )
        for seed in normalized
        for drift in DRIFT_SCALES
        for budget in BUDGET_RATES
        for bonus in RESET_BONUS_CAPS
        for step in QUEUE_STEPS
    ]
    rows.sort(
        key=lambda row: (
            row.seed,
            row.drift_scale,
            row.budget_rate,
            row.reset_bonus_cap,
            row.queue_step,
        )
    )
    rows_hash = _canonical_write(output_dir / "rows.json", [asdict(row) for row in rows])
    _canonical_write(
        output_dir / "chunk_manifest.json",
        {"seeds": list(normalized), "rows_sha256": rows_hash, "rows": len(rows)},
    )


def merge_chunks(
    output_dir: Path,
    chunk_dirs: Sequence[Path],
    baseline_rows_path: Path,
    exact_rows_path: Path,
) -> dict[str, object]:
    baseline = _authenticated_records(baseline_rows_path, BASELINE_ROWS_SHA256)
    exact = _authenticated_records(exact_rows_path, EXACT_ROWS_SHA256)
    records: list[dict[str, object]] = []
    observed_seeds: set[int] = set()
    for directory in chunk_dirs:
        manifest = json.loads((directory / "chunk_manifest.json").read_text(encoding="utf-8"))
        seeds = {int(seed) for seed in manifest["seeds"]}
        if observed_seeds & seeds:
            raise ValueError("chunk seed overlap")
        observed_seeds.update(seeds)
        rows_path = directory / "rows.json"
        if hashlib.sha256(rows_path.read_bytes()).hexdigest().upper() != manifest["rows_sha256"]:
            raise ValueError("chunk rows hash mismatch")
        records.extend(json.loads(rows_path.read_text(encoding="utf-8")))
    if tuple(sorted(observed_seeds)) != DESIGN_SEEDS:
        raise ValueError("merged seeds do not match the frozen design population")
    rows = [PotentialRow(**record) for record in records]
    rows.sort(
        key=lambda row: (
            row.seed,
            row.drift_scale,
            row.budget_rate,
            row.reset_bonus_cap,
            row.queue_step,
        )
    )
    summary = analyze(rows, baseline_records=baseline, exact_records=exact)
    _canonical_write(output_dir / "rows.json", [asdict(row) for row in rows])
    _canonical_write(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-rows", type=Path)
    parser.add_argument("--exact-rows", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seeds", type=int, nargs="+")
    group.add_argument("--merge-inputs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.seeds:
        run_chunk(args.output_dir, args.seeds)
        print(json.dumps({"status": "chunk_complete", "seeds": args.seeds}, indent=2))
        return
    if args.baseline_rows is None or args.exact_rows is None:
        raise ValueError("merge requires both authenticated oracle-confirmation tables")
    summary = merge_chunks(
        args.output_dir,
        args.merge_inputs,
        args.baseline_rows,
        args.exact_rows,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "cells"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
