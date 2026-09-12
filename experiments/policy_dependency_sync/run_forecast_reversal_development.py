"""Development-only phase scan for the Markov forecast-reversal mechanism."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from math import exp, isfinite, log
from pathlib import Path
from typing import Iterable, Sequence

from .certified_factor_async import (
    PolicyName,
    forecast_reversal_config,
    simulate_async_factor_policy,
)


POLICIES: tuple[PolicyName, ...] = (
    "plugin_joint",
    "exact_joint",
    "state_myopic",
    "fixed_edge_0",
    "fixed_edge_1",
    "fixed_edge_2",
    "fixed_initial",
    "round_robin",
    "random_edge",
    "no_refresh",
)
STRONG_BASELINES: tuple[PolicyName, ...] = (
    "state_myopic",
    "fixed_edge_0",
    "fixed_edge_1",
    "fixed_edge_2",
    "fixed_initial",
    "round_robin",
    "random_edge",
    "no_refresh",
)


def geometric_mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items or any(value <= 0.0 or not isfinite(value) for value in items):
        raise ValueError("geometric mean requires finite positive values")
    return float(exp(sum(log(value) for value in items) / len(items)))


def default_grid() -> list[dict[str, float | int]]:
    return [
        {
            "cycle_probability": cycle_probability,
            "maximum_delay": maximum_delay,
            "communication_budget": communication_budget,
            "total_events": 4096,
        }
        for cycle_probability in (0.05, 0.50, 0.80, 0.95)
        for maximum_delay in (1, 4, 8)
        for communication_budget in (0.25, 0.50)
    ]


def _work_item(
    item: tuple[dict[str, float | int], int, tuple[PolicyName, ...]]
) -> list[dict[str, float | int | str]]:
    scenario, seed, policies = item
    config = forecast_reversal_config(
        cycle_probability=float(scenario["cycle_probability"]),
        total_events=int(scenario["total_events"]),
        communication_budget=float(scenario["communication_budget"]),
        maximum_delay=int(scenario["maximum_delay"]),
    )
    rows: list[dict[str, float | int | str]] = []
    for policy in policies:
        result = simulate_async_factor_policy(policy=policy, seed=seed, config=config)
        row: dict[str, float | int | str] = {
            **scenario,
            **asdict(result),
        }
        rows.append(row)
    return rows


def summarize_rows(
    rows: Sequence[dict[str, float | int | str]],
    *,
    policies: Sequence[PolicyName] = POLICIES,
    strong_baselines: Sequence[PolicyName] = STRONG_BASELINES,
) -> dict[str, object]:
    scenario_keys = sorted(
        {
            (
                float(row["cycle_probability"]),
                int(row["maximum_delay"]),
                float(row["communication_budget"]),
                int(row["total_events"]),
            )
            for row in rows
        }
    )
    cells: list[dict[str, object]] = []
    plugin_values: list[float] = []
    strong_values: list[float] = []
    oracle_values: list[float] = []
    favorable_plugin: list[float] = []
    favorable_strong: list[float] = []
    for key in scenario_keys:
        selected = [
            row
            for row in rows
            if (
                float(row["cycle_probability"]),
                int(row["maximum_delay"]),
                float(row["communication_budget"]),
                int(row["total_events"]),
            )
            == key
        ]
        by_policy = {
            policy: [row for row in selected if row["policy"] == policy]
            for policy in policies
        }
        if any(not by_policy[policy] for policy in policies):
            raise ValueError(f"missing policy rows in scenario {key}")
        policy_risk = {
            policy: geometric_mean(
                float(row["average_potential"]) for row in by_policy[policy]
            )
            for policy in policies
        }
        strong_policy = min(strong_baselines, key=lambda name: policy_risk[name])
        plugin = policy_risk["plugin_joint"]
        strong = policy_risk[strong_policy]
        oracle = policy_risk["exact_joint"]
        denominator = strong - oracle
        recovery = (strong - plugin) / denominator if denominator > 0.0 else float("nan")
        plugin_messages = sum(
            int(row["messages"]) for row in by_policy["plugin_joint"]
        ) / len(by_policy["plugin_joint"])
        strong_messages = sum(
            int(row["messages"]) for row in by_policy[strong_policy]
        ) / len(by_policy[strong_policy])
        cell = {
            "cycle_probability": key[0],
            "maximum_delay": key[1],
            "communication_budget": key[2],
            "total_events": key[3],
            "strong_policy": strong_policy,
            "plugin_risk": plugin,
            "strong_risk": strong,
            "oracle_risk": oracle,
            "plugin_over_strong": plugin / strong,
            "oracle_over_strong": oracle / strong,
            "oracle_headroom_recovery": recovery,
            "plugin_messages": plugin_messages,
            "strong_messages": strong_messages,
            "plugin_over_strong_messages": (
                plugin_messages / strong_messages if strong_messages > 0.0 else float("nan")
            ),
        }
        cells.append(cell)
        plugin_values.append(plugin)
        strong_values.append(strong)
        oracle_values.append(oracle)
        if key[0] >= 0.80:
            favorable_plugin.append(plugin)
            favorable_strong.append(strong)

    plugin_aggregate = geometric_mean(plugin_values)
    strong_aggregate = geometric_mean(strong_values)
    oracle_aggregate = geometric_mean(oracle_values)
    return {
        "evidence_class": "development_only_not_preregistered",
        "row_count": len(rows),
        "scenario_count": len(cells),
        "all_finite": all(
            isfinite(float(row[metric]))
            for row in rows
            for metric in ("terminal_potential", "average_potential")
        ),
        "all_environment_counts_equal_within_scenario_seed": all(
            len(
                {
                    int(row["environment_transitions"])
                    for row in rows
                    if float(row["cycle_probability"]) == key[0]
                    and int(row["maximum_delay"]) == key[1]
                    and float(row["communication_budget"]) == key[2]
                    and int(row["total_events"]) == key[3]
                    and int(row["seed"]) == seed
                }
            )
            == 1
            for key in scenario_keys
            for seed in {int(row["seed"]) for row in rows}
        ),
        "aggregate": {
            "plugin_risk": plugin_aggregate,
            "strong_risk": strong_aggregate,
            "oracle_risk": oracle_aggregate,
            "plugin_over_strong": plugin_aggregate / strong_aggregate,
            "oracle_over_strong": oracle_aggregate / strong_aggregate,
            "favorable_plugin_over_strong": geometric_mean(favorable_plugin)
            / geometric_mean(favorable_strong),
            "strictly_improved_cells": sum(
                float(cell["plugin_over_strong"]) < 1.0 for cell in cells
            ),
            "favorable_strictly_improved_cells": sum(
                float(cell["plugin_over_strong"]) < 1.0
                and float(cell["cycle_probability"]) >= 0.80
                for cell in cells
            ),
            "favorable_cell_count": sum(
                float(cell["cycle_probability"]) >= 0.80 for cell in cells
            ),
        },
        "cells": cells,
    }


def run_development(
    *,
    output_dir: Path,
    seeds: Sequence[int],
    workers: int,
    grid: Sequence[dict[str, float | int]] | None = None,
    policies: tuple[PolicyName, ...] = POLICIES,
) -> dict[str, object]:
    scenarios = list(default_grid() if grid is None else grid)
    work = [(scenario, int(seed), policies) for scenario in scenarios for seed in seeds]
    rows: list[dict[str, float | int | str]] = []
    if workers == 1:
        for item in work:
            rows.extend(_work_item(item))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_work_item, work):
                rows.extend(result)
    rows.sort(
        key=lambda row: (
            float(row["cycle_probability"]),
            int(row["maximum_delay"]),
            float(row["communication_budget"]),
            int(row["seed"]),
            str(row["policy"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "endpoints.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows, policies=policies)
    payload = {
        "grid": scenarios,
        "seeds": list(seeds),
        "policies": list(policies),
    }
    summary["configuration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary["endpoints_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=97100)
    parser.add_argument("--seed-count", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0 or args.seed_count <= 0:
        raise ValueError("workers and seed count must be positive")
    summary = run_development(
        output_dir=args.output_dir,
        seeds=range(args.seed_start, args.seed_start + args.seed_count),
        workers=args.workers,
    )
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

