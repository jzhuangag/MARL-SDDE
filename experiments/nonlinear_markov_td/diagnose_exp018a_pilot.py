"""Post-hoc stratified diagnosis of the frozen EXP-018A pilot.

This script never changes or re-evaluates preregistered gates.  It only locates
where their already-recorded failures occur.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp018a_direct_gradient_config import PROJECTION_COUNT, Q_LEVELS, variance_factor


PROJECTIONS = tuple(f"projection_{index:02d}" for index in range(PROJECTION_COUNT))
CELL = ["task", "mixing", "checkpoint", "rho"]


def diagnose(path: Path) -> dict:
    frame = pd.read_csv(path)
    records = []
    adjacent = []
    for cell, group in frame.groupby(CELL, sort=True):
        rho = float(cell[-1])
        values = {
            (q, projection): float(q_group[projection].var(ddof=1))
            for q, q_group in group.groupby("q", sort=True)
            for projection in PROJECTIONS
        }
        for projection in PROJECTIONS:
            baseline = values[(1, projection)]
            path_values = [values[(q, projection)] for q in Q_LEVELS]
            records.append(
                {
                    "rho": rho,
                    "monotone": all(
                        left >= right for left, right in zip(path_values, path_values[1:])
                    ),
                    "errors": [
                        abs((values[(q, projection)] / baseline) / variance_factor(q, rho) - 1.0)
                        for q in Q_LEVELS[1:]
                    ],
                }
            )
            for left_q, right_q in zip(Q_LEVELS, Q_LEVELS[1:]):
                adjacent.append(
                    {
                        "rho": rho,
                        "contrast": f"{left_q}>={right_q}",
                        "passes": values[(left_q, projection)] >= values[(right_q, projection)],
                    }
                )

    by_rho = {}
    for rho in sorted(frame["rho"].unique()):
        selected = [record for record in records if record["rho"] == float(rho)]
        errors = [error for record in selected for error in record["errors"]]
        adjacent_summary = {}
        for contrast in ("1>=4", "4>=16", "16>=32"):
            checks = [
                record["passes"]
                for record in adjacent
                if record["rho"] == float(rho) and record["contrast"] == contrast
            ]
            adjacent_summary[contrast] = float(np.mean(checks))
        by_rho[str(float(rho))] = {
            "monotone_path_fraction": float(np.mean([record["monotone"] for record in selected])),
            "calibration_median_relative_error": float(np.median(errors)),
            "calibration_p90_relative_error": float(np.quantile(errors, 0.9)),
            "adjacent_direction_fractions": adjacent_summary,
            "theoretical_q_path": {str(q): variance_factor(q, float(rho)) for q in Q_LEVELS},
        }
    return {
        "experiment": "EXP-018A",
        "status": "post_hoc_diagnosis_not_a_gate",
        "by_rho": by_rho,
        "interpretation_boundary": "No threshold, seed, task, q, rho, checkpoint, or formal authorization is changed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.input.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
