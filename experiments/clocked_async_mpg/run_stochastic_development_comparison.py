"""Development-only comparison of stochastic asynchronous MPG step rules.

This module deliberately uses a named development seed namespace.  Its output
may be used to decide whether a later experiment is worth preregistering, but
it is not confirmatory evidence.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .stochastic_multistate import (
    simulate_stochastic_asynchronous,
    simulate_stochastic_shadow_barrier,
)


POLICIES = (
    "single_flight_local",
    "single_flight_constant",
    "common_global",
    "generic_rate_balanced",
    "fully_utilized_shadow_barrier",
)


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values)/len(values))


def _one(job: tuple[Any, ...]) -> dict[str, Any]:
    (
        coupling,
        service_ratio,
        seed_index,
        namespace,
        maximum_time,
        horizon,
        batch_size,
        step_fraction,
        target_normalized_gap,
        policy,
    ) = job
    parameters = {
        "coupling": coupling,
        "service_ratio": service_ratio,
        "seed_index": seed_index,
        "namespace": namespace,
        "maximum_time": maximum_time,
        "horizon": horizon,
        "batch_size": batch_size,
        "step_fraction": step_fraction,
        "target_normalized_gap": target_normalized_gap,
    }
    if policy == "fully_utilized_shadow_barrier":
        result = simulate_stochastic_shadow_barrier(**parameters)
    else:
        result = simulate_stochastic_asynchronous(
            **parameters, step_rule=policy
        )
    return {
        "coupling": coupling,
        "policy": policy,
        "seed_index": seed_index,
        "service_ratio": service_ratio,
        "step_fraction": step_fraction,
        **result,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    jobs = [
        (
            coupling,
            service_ratio,
            seed_index,
            args.namespace,
            args.maximum_time,
            args.horizon,
            args.batch_size,
            step_fraction,
            args.target_normalized_gap,
            policy,
        )
        for coupling in args.couplings
        for service_ratio in args.service_ratios
        for seed_index in range(args.seeds)
        for step_fraction in args.step_fractions
        for policy in POLICIES
    ]
    if args.workers == 1:
        rows = [_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_one, jobs, chunksize=4))

    tuning: list[dict[str, Any]] = []
    for policy in POLICIES:
        for step_fraction in args.step_fractions:
            selected = [
                row for row in rows
                if row["policy"] == policy
                and row["step_fraction"] == step_fraction
            ]
            reached = [row for row in selected if row["time_to_target"] is not None]
            effective_times = [
                float(row["time_to_target"])
                if row["time_to_target"] is not None
                else args.maximum_time
                for row in selected
            ]
            tuning.append(
                {
                    "coverage": len(reached)/len(selected),
                    "effective_time_geometric_mean": _geometric_mean(effective_times),
                    "final_gap_geometric_mean": _geometric_mean(
                        [max(1e-15, float(row["final_normalized_gap"])) for row in selected]
                    ),
                    "policy": policy,
                    "step_fraction": step_fraction,
                }
            )
    chosen: dict[str, float] = {}
    for policy in POLICIES:
        candidates = [entry for entry in tuning if entry["policy"] == policy]
        best = min(
            candidates,
            key=lambda entry: (
                -float(entry["coverage"]),
                float(entry["effective_time_geometric_mean"]),
                float(entry["final_gap_geometric_mean"]),
                float(entry["step_fraction"]),
            ),
        )
        chosen[policy] = float(best["step_fraction"])

    indexed = {
        (
            float(row["coupling"]),
            float(row["service_ratio"]),
            int(row["seed_index"]),
            str(row["policy"]),
        ): row
        for row in rows
        if float(row["step_fraction"]) == chosen[str(row["policy"])]
    }
    cells: list[dict[str, Any]] = []
    reference = args.reference_policy
    comparators = tuple(policy for policy in POLICIES if policy != reference)
    for coupling in args.couplings:
        for service_ratio in args.service_ratios:
            cell: dict[str, Any] = {
                "coupling": coupling,
                "service_ratio": service_ratio,
            }
            for policy in POLICIES:
                selected = [
                    indexed[(coupling, service_ratio, seed_index, policy)]
                    for seed_index in range(args.seeds)
                ]
                cell[f"{policy}_coverage"] = sum(
                    row["time_to_target"] is not None for row in selected
                )/args.seeds
            for comparator in comparators:
                time_ratios: list[float] = []
                work_ratios: list[float] = []
                gap_ratios: list[float] = []
                for seed_index in range(args.seeds):
                    candidate = indexed[(
                        coupling, service_ratio, seed_index, reference
                    )]
                    baseline = indexed[(
                        coupling, service_ratio, seed_index, comparator
                    )]
                    if (
                        candidate["time_to_target"] is not None
                        and baseline["time_to_target"] is not None
                    ):
                        time_ratios.append(
                            float(candidate["time_to_target"])
                            /float(baseline["time_to_target"])
                        )
                        work_ratios.append(
                            float(candidate["transition_work_at_target"])
                            /float(baseline["transition_work_at_target"])
                        )
                    gap_ratios.append(
                        max(1e-15, float(candidate["final_normalized_gap"]))
                        /max(1e-15, float(baseline["final_normalized_gap"]))
                    )
                cell[f"vs_{comparator}_paired_coverage"] = len(time_ratios)/args.seeds
                cell[f"vs_{comparator}_median_time_ratio"] = (
                    statistics.median(time_ratios) if time_ratios else None
                )
                cell[f"vs_{comparator}_median_work_ratio"] = (
                    statistics.median(work_ratios) if work_ratios else None
                )
                cell[f"vs_{comparator}_final_gap_geometric_ratio"] = (
                    _geometric_mean(gap_ratios)
                )
            cells.append(cell)

    aggregate: dict[str, Any] = {}
    for comparator in comparators:
        time_values = [
            float(cell[f"vs_{comparator}_median_time_ratio"])
            for cell in cells
            if cell[f"vs_{comparator}_median_time_ratio"] is not None
        ]
        work_values = [
            float(cell[f"vs_{comparator}_median_work_ratio"])
            for cell in cells
            if cell[f"vs_{comparator}_median_work_ratio"] is not None
        ]
        gap_values = [
            float(cell[f"vs_{comparator}_final_gap_geometric_ratio"])
            for cell in cells
        ]
        aggregate[comparator] = {
            "cells_with_paired_target": len(time_values),
            "cells_candidate_faster": sum(value < 1.0 for value in time_values),
            "cells_candidate_less_work": sum(value < 1.0 for value in work_values),
            "final_gap_geometric_ratio": _geometric_mean(gap_values),
            "time_geometric_ratio": _geometric_mean(time_values) if time_values else None,
            "work_geometric_ratio": _geometric_mean(work_values) if work_values else None,
        }
    return {
        "aggregate": aggregate,
        "cells": cells,
        "chosen_step_fractions": chosen,
        "development_only": True,
        "job_count": len(jobs),
        "namespace": args.namespace,
        "reference_policy": reference,
        "row_count": len(rows),
        "tuning": tuning,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--namespace", default="stochastic-strong-baseline-development-v1")
    parser.add_argument("--maximum-time", type=float, default=180.0)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-normalized-gap", type=float, default=0.3)
    parser.add_argument(
        "--reference-policy", choices=POLICIES, default="single_flight_constant"
    )
    parser.add_argument("--couplings", type=float, nargs="+", default=[0.0, 0.08, 0.16, 0.24])
    parser.add_argument("--service-ratios", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--step-fractions", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8])
    args = parser.parse_args()
    if args.workers <= 0 or args.seeds <= 0:
        parser.error("workers and seeds must be positive")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "aggregate": payload["aggregate"],
        "chosen_step_fractions": payload["chosen_step_fractions"],
        "job_count": payload["job_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
