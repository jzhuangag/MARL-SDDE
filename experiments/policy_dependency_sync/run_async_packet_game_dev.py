"""Development-only learning scan for the asynchronous packet-debt game."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np

from .async_packet_game import (
    FIXED_PAIR_POLICIES,
    geometric_mean,
    make_budget_cells,
    make_cells,
    simulate,
)


STRONG_BASELINES = (
    "no_refresh_full_h4",
    "packet_debt_full_h4",
    "largest_mismatch_h4",
    "active_donor_h4",
    "largest_coefficient_h4",
    "top2_mismatch_h4",
    "top2_coefficient_h4",
    "round_robin_h4",
    "periodic_full_h4",
    "fixed_offset0_h4",
    "fixed_offset1_h4",
    "fixed_offset2_h4",
    "fixed_offset3_h4",
    "fixed_offset4_h4",
    *FIXED_PAIR_POLICIES,
)

EVALUATED_POLICIES = ("signed_oracle_graph_h4", *STRONG_BASELINES)


def evaluate(seeds: list[int], launches: int, grid: str = "legacy") -> dict:
    experiment_cells = make_cells() if grid == "legacy" else make_budget_cells()
    cells = []
    for cell in experiment_cells:
        rows = {
            policy: [simulate(cell, policy, seed, launches) for seed in seeds]
            for policy in EVALUATED_POLICIES
        }
        aggregates = {
            policy: {
                "risk": geometric_mean([row["risk"] for row in policy_rows]),
                "terminal_risk": geometric_mean(
                    [max(row["terminal_risk"], 1e-15) for row in policy_rows]
                ),
                "messages_per_launch": float(
                    np.mean([row["messages_per_launch"] for row in policy_rows])
                ),
                "environment_per_launch": float(
                    np.mean([row["environment_per_launch"] for row in policy_rows])
                ),
                "accepted_step_fraction": float(
                    np.mean([row["accepted_step_fraction"] for row in policy_rows])
                ),
                "slow_mean_horizon": float(
                    np.mean([row["slow_mean_horizon"] for row in policy_rows])
                ),
                "fast_mean_horizon": float(
                    np.mean([row["fast_mean_horizon"] for row in policy_rows])
                ),
                "distinct_graph_supports": float(
                    np.mean([row["distinct_graph_supports"] for row in policy_rows])
                ),
                "finite": all(row["finite"] for row in policy_rows),
            }
            for policy, policy_rows in rows.items()
        }
        feasible_baselines = [
            policy
            for policy in STRONG_BASELINES
            if aggregates[policy]["messages_per_launch"] <= cell.message_budget + 0.2
            and aggregates[policy]["environment_per_launch"]
            <= cell.environment_budget + 0.2
        ]
        strongest = min(feasible_baselines, key=lambda name: aggregates[name]["risk"])
        strongest_terminal = min(
            feasible_baselines,
            key=lambda name: aggregates[name]["terminal_risk"],
        )
        proposed = aggregates["signed_oracle_graph_h4"]["risk"]
        baseline = aggregates[strongest]["risk"]
        proposed_terminal = aggregates["signed_oracle_graph_h4"]["terminal_risk"]
        baseline_terminal = aggregates[strongest_terminal]["terminal_risk"]
        cells.append(
            {
                "cell": cell.key,
                "coupling": cell.coupling,
                "maximum_extra_delay": cell.maximum_extra_delay,
                "message_budget": cell.message_budget,
                "environment_budget": cell.environment_budget,
                "mode_switch_probability": cell.mode_switch_probability,
                "strongest_baseline": strongest,
                "strongest_terminal_baseline": strongest_terminal,
                "gain": (baseline - proposed) / baseline,
                "terminal_gain": (
                    baseline_terminal - proposed_terminal
                ) / baseline_terminal,
                "aggregates": aggregates,
            }
        )
    gains = np.asarray([row["gain"] for row in cells])
    terminal_gains = np.asarray([row["terminal_gain"] for row in cells])
    active_rows = [row for row in cells if row["coupling"] > 0.0]
    control_rows = [row for row in cells if row["coupling"] == 0.0]
    active_gains = np.asarray([row["gain"] for row in active_rows])
    active_terminal_gains = np.asarray(
        [row["terminal_gain"] for row in active_rows]
    )
    by_delay = {
        str(delay): float(
            np.median(
                [row["gain"] for row in cells if row["maximum_extra_delay"] == delay]
            )
        )
        for delay in (2, 5)
    }
    proposed = [row["aggregates"]["signed_oracle_graph_h4"] for row in cells]
    return {
        "status": "development_only",
        "scientific_outcome_authorized": False,
        "seeds": seeds,
        "launches": launches,
        "grid": grid,
        "cells": cells,
        "metrics": {
            "median_gain": float(np.median(gains)),
            "strict_gain_fraction": float(np.mean(gains > 0.0)),
            "gain_at_least_0_05_fraction": float(np.mean(gains >= 0.05)),
            "median_terminal_gain": float(np.median(terminal_gains)),
            "strict_terminal_gain_fraction": float(np.mean(terminal_gains > 0.0)),
            "terminal_gain_at_least_0_10_fraction": float(
                np.mean(terminal_gains >= 0.10)
            ),
            "minimum_terminal_gain": float(np.min(terminal_gains)),
            "active_cell_count": len(active_rows),
            "control_cell_count": len(control_rows),
            "active_median_gain": float(np.median(active_gains)),
            "active_strict_gain_fraction": float(np.mean(active_gains > 0.0)),
            "active_median_terminal_gain": float(
                np.median(active_terminal_gains)
            ),
            "active_strict_terminal_gain_fraction": float(
                np.mean(active_terminal_gains > 0.0)
            ),
            "maximum_uncoupled_absolute_gain": float(
                max([abs(row["gain"]) for row in control_rows], default=0.0)
            ),
            "maximum_uncoupled_absolute_terminal_gain": float(
                max(
                    [abs(row["terminal_gain"]) for row in control_rows],
                    default=0.0,
                )
            ),
            "strongest_auc_baseline_counts": dict(
                sorted(Counter(row["strongest_baseline"] for row in cells).items())
            ),
            "strongest_terminal_baseline_counts": dict(
                sorted(
                    Counter(
                        row["strongest_terminal_baseline"] for row in cells
                    ).items()
                )
            ),
            "gain_by_maximum_extra_delay": by_delay,
            "mean_slow_minus_fast_horizon": float(
                np.mean(
                    [row["slow_mean_horizon"] - row["fast_mean_horizon"] for row in proposed]
                )
            ),
            "minimum_mean_distinct_graph_supports": float(
                min(row["distinct_graph_supports"] for row in proposed)
            ),
            "maximum_message_excess": float(
                max(
                    row["aggregates"]["signed_oracle_graph_h4"]["messages_per_launch"]
                    - row["message_budget"]
                    for row in cells
                )
            ),
            "maximum_environment_excess": float(
                max(
                    row["aggregates"]["signed_oracle_graph_h4"]["environment_per_launch"]
                    - row["environment_budget"]
                    for row in cells
                )
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launches", type=int, default=240)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--grid", choices=("legacy", "budget"), default="legacy")
    args = parser.parse_args()
    result = evaluate(
        [43001 + index for index in range(args.seeds)],
        args.launches,
        grid=args.grid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
