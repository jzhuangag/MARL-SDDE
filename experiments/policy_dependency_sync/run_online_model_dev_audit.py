"""Development-only comparison of observable and oracle signed graph actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .async_packet_game import make_budget_cells, simulate


def audit(seeds: list[int], launches: int) -> dict:
    rows = []
    for cell in make_budget_cells():
        for seed in seeds:
            online = simulate(cell, "signed_online_model_graph_h4", seed, launches)
            oracle = simulate(cell, "signed_oracle_graph_h4", seed, launches)
            rows.append(
                {
                    "cell": cell.key,
                    "seed": seed,
                    "active": cell.coupling > 0.0,
                    "trace_equal": online["graph_trace_sha256"]
                    == oracle["graph_trace_sha256"],
                    "risk_absolute_difference": abs(
                        float(online["risk"]) - float(oracle["risk"])
                    ),
                    "terminal_absolute_difference": abs(
                        float(online["terminal_risk"])
                        - float(oracle["terminal_risk"])
                    ),
                    "message_absolute_difference": abs(
                        float(online["messages_per_launch"])
                        - float(oracle["messages_per_launch"])
                    ),
                    "target_estimation_error": float(
                        online["target_estimation_error"]
                    ),
                }
            )
    active = [row for row in rows if row["active"]]
    return {
        "status": "development_only",
        "scientific_outcome_authorized": False,
        "seeds": seeds,
        "launches": launches,
        "rows": rows,
        "metrics": {
            "active_comparisons": len(active),
            "active_trace_agreement_fraction": float(
                np.mean([row["trace_equal"] for row in active])
            ),
            "maximum_active_risk_absolute_difference": max(
                row["risk_absolute_difference"] for row in active
            ),
            "maximum_active_terminal_absolute_difference": max(
                row["terminal_absolute_difference"] for row in active
            ),
            "maximum_active_message_absolute_difference": max(
                row["message_absolute_difference"] for row in active
            ),
            "median_active_target_estimation_error": float(
                np.median([row["target_estimation_error"] for row in active])
            ),
            "maximum_active_target_estimation_error": max(
                row["target_estimation_error"] for row in active
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launches", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=43005)
    args = parser.parse_args()
    result = audit(
        [args.seed_start + index for index in range(args.seeds)], args.launches
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
