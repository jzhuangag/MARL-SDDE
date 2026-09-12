"""Analyze the frozen SMACv2 coupling calibration gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(paths: list[Path]) -> dict:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_regime = {
        regime: [row for row in rows if row["coupling"] == regime]
        for regime in ("independent", "shared")
    }
    if any(len(group) != 2 for group in by_regime.values()):
        raise ValueError("expected exactly two independent and two shared runs")
    correlations = {
        regime: [
            row["dependence"]["median_pairwise_correlation"] for row in group
        ]
        for regime, group in by_regime.items()
    }
    selection = {
        "independent_q8_fraction": float(
            np.mean([row["decision"]["selected_q"] == 8 for row in by_regime["independent"]])
        ),
        "shared_q1_fraction": float(
            np.mean([row["decision"]["selected_q"] == 1 for row in by_regime["shared"]])
        ),
    }
    independent_median = float(np.median(correlations["independent"]))
    shared_median = float(np.median(correlations["shared"]))
    gates = {
        "all_finite_and_outcome_free": all(
            row["policy_updates"] == 0
            and not row["return_or_win_rate_observed"]
            and np.isfinite(row["dependence"]["median_pairwise_correlation"])
            for row in rows
        ),
        "independent_median_correlation_at_most_0p25": independent_median <= 0.25,
        "shared_median_correlation_at_least_0p75": shared_median >= 0.75,
        "correlation_separation_at_least_0p50": shared_median - independent_median >= 0.50,
        "independent_selects_q8_both": selection["independent_q8_fraction"] == 1.0,
        "shared_selects_q1_both": selection["shared_q1_fraction"] == 1.0,
    }
    return {
        "experiment_id": "TSP-SMACV2-COUPLING-CAL-001",
        "correlations": correlations,
        "independent_median": independent_median,
        "shared_median": shared_median,
        "separation": shared_median - independent_median,
        "selection": selection,
        "gates": gates,
        "pass": all(gates.values()),
        "authorization_if_pass": "TSP-MARL-SMACV2-DEV-001 development array only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
