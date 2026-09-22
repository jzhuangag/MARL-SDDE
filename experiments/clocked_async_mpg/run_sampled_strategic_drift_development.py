"""Development comparison for the fully charged sample-split controller."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .sampled_strategic_drift import simulate_sample_split_strategic_drift
from .stochastic_multistate import (
    simulate_stochastic_asynchronous,
    simulate_stochastic_shadow_barrier,
)
from .strategic_drift_development import simulate_oracle_strategic_drift


POLICIES = (
    "sample_split_debt",
    "oracle_debt",
    "pathwise_constant",
    "raw_common",
    "shadow_barrier",
)


def _geometric(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return float(math.exp(sum(math.log(value) for value in values)/len(values)))


def _one(job: tuple[Any, ...]) -> dict[str, Any]:
    coupling, ratio, seed, namespace, policy = job
    common = {
        "coupling": coupling,
        "service_ratio": ratio,
        "seed_index": seed,
        "namespace": namespace,
        "maximum_time": 180.0,
        "horizon": 16,
        "batch_size": 16,
        "step_fraction": 1.0,
        "target_normalized_gap": 0.3,
    }
    if policy == "sample_split_debt":
        result = simulate_sample_split_strategic_drift(
            **common,
            risk_budget=0.001,
            tradeoff=10.0,
        )
    elif policy == "oracle_debt":
        result = simulate_oracle_strategic_drift(
            **common,
            risk_budget=0.001,
            tradeoff=10.0,
            hard_no_harm=False,
        )
    elif policy == "pathwise_constant":
        result = simulate_stochastic_asynchronous(
            **common,
            step_rule="single_flight_pathwise_constant",
            history_inflation=2.0,
        )
    elif policy == "raw_common":
        result = simulate_stochastic_asynchronous(
            **common, step_rule="common_global", history_inflation=1.0
        )
    elif policy == "shadow_barrier":
        result = simulate_stochastic_shadow_barrier(**common)
    else:
        raise ValueError("unknown policy")
    return {
        "coupling": coupling,
        "policy": policy,
        "seed": seed,
        "service_ratio": ratio,
        **result,
    }


def summarize(rows: list[dict[str, Any]], seeds: int) -> dict[str, Any]:
    indexed = {
        (row["coupling"], row["service_ratio"], row["seed"], row["policy"]): row
        for row in rows
    }
    couplings = sorted({float(row["coupling"]) for row in rows})
    ratios = sorted({float(row["service_ratio"]) for row in rows})
    cells: list[dict[str, Any]] = []
    for coupling in couplings:
        for ratio in ratios:
            candidate_rows = [
                indexed[(coupling, ratio, seed, "sample_split_debt")]
                for seed in range(seeds)
            ]
            cell: dict[str, Any] = {
                "coupling": coupling,
                "mean_rejection_rate": statistics.mean(
                    float(row["rejected_updates"])/float(row["applied_updates"])
                    for row in candidate_rows
                ),
                "mean_scale": statistics.mean(
                    float(row["mean_scale"]) for row in candidate_rows
                ),
                "mean_terminal_debt": statistics.mean(
                    float(row["debt"]) for row in candidate_rows
                ),
                "service_ratio": ratio,
            }
            for comparator in POLICIES[1:]:
                time_ratios: list[float] = []
                work_ratios: list[float] = []
                gap_ratios: list[float] = []
                for seed in range(seeds):
                    candidate = indexed[
                        (coupling, ratio, seed, "sample_split_debt")
                    ]
                    baseline = indexed[(coupling, ratio, seed, comparator)]
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
                cell[f"vs_{comparator}_coverage"] = len(time_ratios)/seeds
                cell[f"vs_{comparator}_time_ratio"] = statistics.median(time_ratios)
                cell[f"vs_{comparator}_work_ratio"] = statistics.median(work_ratios)
                cell[f"vs_{comparator}_gap_ratio"] = _geometric(gap_ratios)
            cells.append(cell)
    aggregate: dict[str, Any] = {}
    for population, selected in (
        ("all", cells),
        ("heterogeneous", [cell for cell in cells if cell["service_ratio"] > 1.0]),
    ):
        aggregate[population] = {}
        for comparator in POLICIES[1:]:
            times = [float(cell[f"vs_{comparator}_time_ratio"]) for cell in selected]
            works = [float(cell[f"vs_{comparator}_work_ratio"]) for cell in selected]
            gaps = [float(cell[f"vs_{comparator}_gap_ratio"]) for cell in selected]
            aggregate[population][comparator] = {
                "cell_count": len(selected),
                "cells_faster": sum(value < 1.0 for value in times),
                "gap_geometric_ratio": _geometric(gaps),
                "time_geometric_ratio": _geometric(times),
                "work_geometric_ratio": _geometric(works),
            }
    return {"aggregate": aggregate, "cells": cells}


def run(*, seeds: int, workers: int, namespace: str) -> dict[str, Any]:
    jobs = [
        (coupling, ratio, seed, namespace, policy)
        for coupling in (0.0, 0.08, 0.16, 0.24)
        for ratio in (1.0, 2.0, 4.0, 8.0)
        for seed in range(seeds)
        for policy in POLICIES
    ]
    if workers == 1:
        rows = [_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_one, jobs, chunksize=5))
    payload = summarize(rows, seeds)
    return {
        **payload,
        "development_only": True,
        "fixed_controller_parameters": {"risk_budget": 0.001, "tradeoff": 10.0},
        "job_count": len(jobs),
        "namespace": namespace,
        "policies": list(POLICIES),
        "row_count": len(rows),
        "seed_count": seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--namespace", default="sample-split-strategic-drift-development-v1"
    )
    args = parser.parse_args()
    if args.seeds <= 0 or args.workers <= 0:
        parser.error("seeds and workers must be positive")
    payload = run(seeds=args.seeds, workers=args.workers, namespace=args.namespace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
