"""Frozen PDSG-FR-001 CPU confirmation runner and gate analyzer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Sequence

from .run_forecast_reversal_development import geometric_mean, run_development


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "docs" / "pdsg_fr001_confirmation_config.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in config["scientific_source_sha256"].items():
        observed = hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"source hash mismatch for {relative}: {observed}")
    return config


def expanded_grid(config: dict[str, object]) -> list[dict[str, float | int]]:
    grid = config["grid"]
    return [
        {
            "cycle_probability": float(cycle_probability),
            "maximum_delay": int(maximum_delay),
            "communication_budget": float(communication_budget),
            "total_events": int(grid["total_events"]),
        }
        for cycle_probability in grid["cycle_probability"]
        for maximum_delay in grid["maximum_delay"]
        for communication_budget in grid["communication_budget"]
    ]


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_confirmation(
    *, output_dir: Path, config: dict[str, object]
) -> dict[str, object]:
    rows = _load_rows(output_dir / "endpoints.csv")
    grid = expanded_grid(config)
    seeds = range(
        int(config["confirmation_seed_start"]),
        int(config["confirmation_seed_start"]) + int(config["confirmation_seed_count"]),
    )
    policies = tuple(config["policies"])
    expected_rows = len(grid) * len(seeds) * len(policies)
    identity = {
        (
            float(row["cycle_probability"]),
            int(row["maximum_delay"]),
            float(row["communication_budget"]),
            int(row["seed"]),
            row["policy"],
        )
        for row in rows
    }
    expected_identity = {
        (
            float(cell["cycle_probability"]),
            int(cell["maximum_delay"]),
            float(cell["communication_budget"]),
            int(seed),
            policy,
        )
        for cell in grid
        for seed in seeds
        for policy in policies
    }
    finite = all(
        math.isfinite(float(row[metric]))
        for row in rows
        for metric in ("terminal_potential", "average_potential", "final_parameter")
    )
    queue_step = float(config["queue_step"])
    budget_violations = []
    for row in rows:
        cap = float(row["communication_budget"]) * int(row["total_events"])
        if row["policy"] in ("plugin_joint", "exact_joint"):
            cap += float(row["final_queue"]) / queue_step
        if int(row["messages"]) > cap + 1e-9:
            budget_violations.append(
                {
                    "identity": [
                        row["cycle_probability"],
                        row["maximum_delay"],
                        row["communication_budget"],
                        row["seed"],
                        row["policy"],
                    ],
                    "messages": row["messages"],
                    "cap": cap,
                }
            )
    environment_equal = all(
        len(
            {
                int(row["environment_transitions"])
                for row in rows
                if float(row["cycle_probability"])
                == float(cell["cycle_probability"])
                and int(row["maximum_delay"]) == int(cell["maximum_delay"])
                and float(row["communication_budget"])
                == float(cell["communication_budget"])
                and int(row["seed"]) == seed
            }
        )
        == 1
        for cell in grid
        for seed in seeds
    )

    cells: list[dict[str, object]] = []
    for cell in grid:
        selected = [
            row
            for row in rows
            if float(row["cycle_probability"]) == float(cell["cycle_probability"])
            and int(row["maximum_delay"]) == int(cell["maximum_delay"])
            and float(row["communication_budget"])
            == float(cell["communication_budget"])
        ]
        by_policy = {
            policy: [row for row in selected if row["policy"] == policy]
            for policy in policies
        }
        risk = {
            policy: geometric_mean(
                float(row["average_potential"]) for row in by_policy[policy]
            )
            for policy in policies
        }
        messages = {
            policy: statistics.mean(int(row["messages"]) for row in by_policy[policy])
            for policy in policies
        }
        strong = risk[str(config["primary_baseline"])]
        plugin = risk[str(config["primary_policy"])]
        oracle = risk[str(config["oracle_diagnostic"])]
        cells.append(
            {
                **cell,
                "plugin_risk": plugin,
                "strong_risk": strong,
                "oracle_risk": oracle,
                "plugin_over_strong": plugin / strong,
                "oracle_over_strong": oracle / strong,
                "oracle_headroom_recovery": (
                    (strong - plugin) / (strong - oracle)
                    if strong > oracle
                    else float("nan")
                ),
                "plugin_messages": messages[str(config["primary_policy"])],
                "strong_messages": messages[str(config["primary_baseline"])],
            }
        )

    favorable = [cell for cell in cells if float(cell["cycle_probability"]) >= 0.8]
    low = [cell for cell in cells if float(cell["cycle_probability"]) == 0.05]
    favorable_ratio = geometric_mean(float(cell["plugin_risk"]) for cell in favorable) / geometric_mean(
        float(cell["strong_risk"]) for cell in favorable
    )
    low_ratio = geometric_mean(float(cell["plugin_risk"]) for cell in low) / geometric_mean(
        float(cell["strong_risk"]) for cell in low
    )
    all_ratio = geometric_mean(float(cell["plugin_risk"]) for cell in cells) / geometric_mean(
        float(cell["strong_risk"]) for cell in cells
    )
    oracle_ratio = geometric_mean(float(cell["oracle_risk"]) for cell in cells) / geometric_mean(
        float(cell["strong_risk"]) for cell in cells
    )
    recoveries = [float(cell["oracle_headroom_recovery"]) for cell in favorable]

    favorable_seed_wins = 0
    favorable_seed_total = 0
    for cell in favorable:
        for seed in seeds:
            lookup = {
                row["policy"]: float(row["average_potential"])
                for row in rows
                if float(row["cycle_probability"])
                == float(cell["cycle_probability"])
                and int(row["maximum_delay"]) == int(cell["maximum_delay"])
                and float(row["communication_budget"])
                == float(cell["communication_budget"])
                and int(row["seed"]) == seed
            }
            favorable_seed_wins += int(
                lookup[str(config["primary_policy"])]
                < lookup[str(config["primary_baseline"])]
            )
            favorable_seed_total += 1

    pareto_points = 0
    for cell in favorable:
        alternatives = [
            other
            for other in favorable
            if float(other["cycle_probability"]) == float(cell["cycle_probability"])
            and int(other["maximum_delay"]) == int(cell["maximum_delay"])
            and float(other["strong_messages"]) >= float(cell["plugin_messages"])
        ]
        pareto_points += int(
            bool(alternatives)
            and min(float(other["strong_risk"]) for other in alternatives)
            > float(cell["plugin_risk"])
        )

    thresholds = config["thresholds"]
    gates = {
        "P1_complete_identity": len(rows) == expected_rows and identity == expected_identity,
        "P2_finite_and_equal_environment": finite and environment_equal,
        "P3_communication_budget": not budget_violations,
        "P4_favorable_risk": favorable_ratio
        <= float(thresholds["maximum_favorable_geometric_risk_ratio"]),
        "P5_favorable_cells": sum(
            float(cell["plugin_over_strong"]) < 1.0 for cell in favorable
        )
        >= int(thresholds["minimum_favorable_strict_cells"]),
        "P6_seedwise_direction": favorable_seed_wins / favorable_seed_total
        >= float(thresholds["minimum_favorable_seedwise_win_fraction"]),
        "P7_oracle_recovery": statistics.median(recoveries)
        >= float(thresholds["minimum_median_oracle_headroom_recovery"]),
        "P8_pareto_points": pareto_points
        >= int(thresholds["minimum_favorable_pareto_dominant_points"]),
        "P9_phase_separation": low_ratio - favorable_ratio
        >= float(thresholds["minimum_low_to_favorable_phase_gap"]),
        "P10_all_cell_risk": all_ratio
        <= float(thresholds["maximum_all_cell_geometric_risk_ratio"]),
        "P11_oracle_headroom": oracle_ratio
        <= float(thresholds["maximum_oracle_geometric_risk_ratio"]),
    }
    validation = {
        "experiment_id": config["experiment_id"],
        "evidence_class": config["evidence_class"],
        "rows": len(rows),
        "expected_rows": expected_rows,
        "endpoint_sha256": hashlib.sha256(
            (output_dir / "endpoints.csv").read_bytes()
        ).hexdigest(),
        "metrics": {
            "favorable_geometric_risk_ratio": favorable_ratio,
            "low_cycle_geometric_risk_ratio": low_ratio,
            "low_to_favorable_phase_gap": low_ratio - favorable_ratio,
            "all_cell_geometric_risk_ratio": all_ratio,
            "oracle_geometric_risk_ratio": oracle_ratio,
            "favorable_strict_cells": sum(
                float(cell["plugin_over_strong"]) < 1.0 for cell in favorable
            ),
            "favorable_cell_count": len(favorable),
            "favorable_seedwise_wins": favorable_seed_wins,
            "favorable_seedwise_total": favorable_seed_total,
            "favorable_seedwise_win_fraction": favorable_seed_wins
            / favorable_seed_total,
            "median_oracle_headroom_recovery": statistics.median(recoveries),
            "favorable_pareto_dominant_points": pareto_points,
            "budget_violation_count": len(budget_violations),
        },
        "gates": gates,
        "scientific_pass": all(gates.values()),
        "reproduction_pass": False,
        "pursuit_design_authorized": False,
        "formal_authorized": False,
        "gpu_authorized": False,
        "cells": cells,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "run":
        run_development(
            output_dir=args.output_dir,
            seeds=range(
                int(config["confirmation_seed_start"]),
                int(config["confirmation_seed_start"])
                + int(config["confirmation_seed_count"]),
            ),
            workers=args.workers,
            grid=expanded_grid(config),
            policies=tuple(config["policies"]),
        )
    validation = evaluate_confirmation(output_dir=args.output_dir, config=config)
    print(json.dumps(validation["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
