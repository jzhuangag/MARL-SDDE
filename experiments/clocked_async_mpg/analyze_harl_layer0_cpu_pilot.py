"""Apply the frozen Layer-0 CPU pilot gates without altering outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(config: dict[str, Any], primary: dict[str, Any], byte_equal: bool) -> dict[str, Any]:
    rows = primary["rows"]
    expected_charge = int(config["expected_actor_transitions_per_run"])
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for value in (
            row["initial_return"],
            row["final_return"],
            row["return_change"],
            row["mean_scale"],
            row["maximum_debt"],
        )
    )
    l1 = (
        len(rows) == config["expected_rows"]
        and finite
        and all(
            row["charged_actor_transitions"] == expected_charge
            and row["completed_actor_transitions"] == expected_charge
            and row["maximum_self_fresh_error"] == 0.0
            for row in rows
        )
    )
    cells = primary["cells"]
    strategic_balanced = cells["balanced"]["modes"]["strategic_split"]
    strategic_heterogeneous = cells["heterogeneous"]["modes"]["strategic_split"]
    l2 = (
        strategic_balanced["mean_return_change"] > 0.0
        and strategic_heterogeneous["mean_return_change"] > 0.0
    )
    heterogeneous_contrast = cells["heterogeneous"]["strategic_contrasts"][
        "raw_full_data"
    ]
    raw_heterogeneous_tail = cells["heterogeneous"]["modes"]["raw_full_data"][
        "lower_quartile_mean_final_return"
    ]
    required_tail_gain = (
        config["gates"]["L3_heterogeneous_lower_tail"][
            "minimum_relative_improvement"
        ]
        * abs(raw_heterogeneous_tail)
    )
    l3 = (
        heterogeneous_contrast["lower_quartile_mean_return_difference"]
        >= required_tail_gain
    )
    l4 = heterogeneous_contrast["relative_mean_shortfall"] <= config["gates"][
        "L4_heterogeneous_mean_safety"
    ]["maximum_relative_shortfall"]
    l5 = heterogeneous_contrast["strategic_strictly_better_fraction"] >= config[
        "gates"
    ]["L5_heterogeneous_directionality"]["minimum_fraction"]
    intermediate_by_profile = {
        profile: float(
            np.mean(
                [
                    row["intermediate_scale_fraction"]
                    for row in rows
                    if row["service_profile"] == profile
                    and row["mode"] == "strategic_split"
                ]
            )
        )
        for profile in config["service_profiles"]
    }
    lower = config["gates"]["L6_nontrivial_debt_control"][
        "minimum_intermediate_scale_fraction"
    ]
    upper = config["gates"]["L6_nontrivial_debt_control"][
        "maximum_intermediate_scale_fraction"
    ]
    l6 = all(lower <= value <= upper for value in intermediate_by_profile.values())
    balanced_contrast = cells["balanced"]["strategic_contrasts"]["raw_full_data"]
    l7 = balanced_contrast["relative_mean_shortfall"] <= config["gates"][
        "L7_balanced_mean_safety"
    ]["maximum_relative_shortfall"]
    split_cost_contrasts = [
        cells[profile]["strategic_contrasts"]["raw_half_data"]
        for profile in config["service_profiles"]
    ]
    l8 = all(
        contrast["mean_paired_final_return_difference"] > 0.0
        and contrast["lower_quartile_mean_return_difference"] > 0.0
        for contrast in split_cost_contrasts
    )
    gate_values = {
        "L1_validity_and_accounting": l1,
        "L2_positive_learning": l2,
        "L3_heterogeneous_lower_tail": l3,
        "L4_heterogeneous_mean_safety": l4,
        "L5_heterogeneous_directionality": l5,
        "L6_nontrivial_debt_control": l6,
        "L7_balanced_mean_safety": l7,
        "L8_split_cost_value": l8,
        "L9_reproducibility": byte_equal,
    }
    return {
        "experiment_id": config["experiment_id"],
        "gate_values": gate_values,
        "all_mandatory_gates_pass": all(gate_values.values()),
        "gpu_pilot_authorized": all(gate_values.values()),
        "diagnostics": {
            "rows": len(rows),
            "finite": finite,
            "required_heterogeneous_lower_tail_gain": required_tail_gain,
            "observed_heterogeneous_lower_tail_gain": heterogeneous_contrast[
                "lower_quartile_mean_return_difference"
            ],
            "heterogeneous_mean_shortfall": heterogeneous_contrast[
                "relative_mean_shortfall"
            ],
            "heterogeneous_directionality": heterogeneous_contrast[
                "strategic_strictly_better_fraction"
            ],
            "intermediate_scale_fraction_by_profile": intermediate_by_profile,
            "balanced_mean_shortfall": balanced_contrast["relative_mean_shortfall"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    byte_equal = _sha256(args.primary) == _sha256(args.reproduction)
    result = analyze(config, primary, byte_equal)
    result["config_sha256"] = _sha256(args.config)
    result["primary_sha256"] = _sha256(args.primary)
    result["reproduction_sha256"] = _sha256(args.reproduction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
