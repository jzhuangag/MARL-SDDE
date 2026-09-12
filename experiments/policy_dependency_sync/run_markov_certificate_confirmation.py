"""Independent frozen confirmation of the tabular Markov certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .run_markov_certificate_nonvacuity import (
    config_hash,
    run_rows_for_config,
    summarize_for_config,
)


CONFIG: dict[str, Any] = {
    "audit_id": "MCERT-001",
    "seeds": list(range(95000, 95128)),
    "sample_sizes_per_positive_action": [256, 512, 1024, 2048, 4096, 8192],
    "primary_sample_size_per_positive_action": 4096,
    "transition": [[0.9, 0.1], [0.1, 0.9]],
    "score_by_action": [[0.25, -0.25], [-0.25, 0.25]],
    "future_horizon": 4,
    "total_action_count_including_null": 3,
    "event_failure_probability": 0.05,
    "gates": {
        "N1_correct_edge_rate_at_4096_min": 0.99,
        "N2_median_value_recovery_at_4096_min": 0.50,
        "N3_p05_value_recovery_at_4096_min": 0.40,
        "N4_wrong_edge_positive_rate_at_4096_max": 0.0,
        "N5_median_recovery_monotone": True,
        "N6_exact_transition_charging": True,
        "N7_local_robust_dp_scalar_work_max": 128,
        "N8_all_rows_finite": True,
    },
}


def canonical_config_hash() -> str:
    return config_hash(CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "run"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        print(json.dumps({"config_hash": canonical_config_hash()}, sort_keys=True))
        return
    if args.output is None:
        raise SystemExit("--output is required in run mode")
    rows = run_rows_for_config(CONFIG)
    payload = {
        "config": CONFIG,
        "summary": summarize_for_config(rows, CONFIG),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
