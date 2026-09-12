"""Frozen analyzer for the EXP-018A CPU direct-gradient pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp018a_direct_gradient_config import (
    CHECKPOINTS,
    EXPERIMENT,
    MIXING_PROFILES,
    PILOT_SEEDS,
    PILOT_THRESHOLDS,
    PROJECTION_COUNT,
    Q_LEVELS,
    RHO_LEVELS,
    STATIC_MANIFEST_HASH,
    TASKS,
    expected_rows,
    variance_factor,
)


PROJECTION_COLUMNS = tuple(f"projection_{index:02d}" for index in range(PROJECTION_COUNT))
CELL = ["task", "mixing", "checkpoint", "rho"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def analyze(projections_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(projections_path)
    finite = np.isfinite(frame[list(PROJECTION_COLUMNS)].to_numpy(dtype=float)).all()
    expected_unique = len(PILOT_SEEDS) * len(TASKS) * len(MIXING_PROFILES) * len(CHECKPOINTS) * len(RHO_LEVELS) * len(Q_LEVELS)
    unique_rows = frame[["seed", *CELL, "q"]].drop_duplicates().shape[0]
    manifest_match = set(frame["manifest_hash"].astype(str)) == {STATIC_MANIFEST_HASH}
    parameters_unchanged = bool(
        (frame["parameter_hash_before"] == frame["parameter_hash_after"]).all()
    )

    variances: dict[tuple[object, ...], float] = {}
    for key, group in frame.groupby([*CELL, "q"], sort=True):
        for projection in PROJECTION_COLUMNS:
            variances[(*key, projection)] = float(group[projection].var(ddof=1))

    calibration_errors: list[float] = []
    monotone_paths: list[bool] = []
    ratio_records = 0
    for cell_key, _ in frame.groupby(CELL, sort=True):
        rho = float(cell_key[-1])
        for projection in PROJECTION_COLUMNS:
            baseline = variances[(*cell_key, 1, projection)]
            path = []
            for q in Q_LEVELS:
                value = variances[(*cell_key, q, projection)]
                path.append(value)
                if q > 1 and baseline > 0.0:
                    ratio = value / baseline
                    calibration_errors.append(
                        abs(ratio / variance_factor(q, rho) - 1.0)
                    )
                    ratio_records += 1
            monotone_paths.append(
                all(left >= right for left, right in zip(path, path[1:]))
            )

    q1_spreads: list[float] = []
    for _, group in frame[frame["q"] == 1].groupby(
        ["task", "mixing", "checkpoint"], sort=True
    ):
        for projection in PROJECTION_COLUMNS:
            values = [
                float(rho_group[projection].var(ddof=1))
                for _, rho_group in group.groupby("rho", sort=True)
            ]
            minimum = min(values)
            q1_spreads.append(max(values) / minimum - 1.0 if minimum > 0.0 else math.inf)

    share_errors = []
    share_estimates = {}
    q32 = frame[frame["q"] == max(Q_LEVELS)]
    for rho, group in q32.groupby("rho", sort=True):
        estimate = float(group["pairwise_shared_fraction"].mean())
        share_estimates[str(float(rho))] = estimate
        share_errors.append(abs(estimate - float(rho)))

    gates = {
        "G1_shape_finite_unique": len(frame) == expected_rows()
        and finite
        and unique_rows == expected_unique,
        "G2_manifest_and_parameter_freeze": manifest_match and parameters_unchanged,
        "G3_pairwise_share_calibration": max(share_errors)
        <= PILOT_THRESHOLDS["maximum_pairwise_share_absolute_error"],
        "G4_variance_factor_calibration": quantile(calibration_errors, 0.5)
        <= PILOT_THRESHOLDS["median_variance_calibration_relative_error"]
        and quantile(calibration_errors, 0.9)
        <= PILOT_THRESHOLDS["p90_variance_calibration_relative_error"],
        "G5_q1_marginal_invariance": quantile(q1_spreads, 0.5)
        <= PILOT_THRESHOLDS["median_q1_rho_invariance_spread"]
        and quantile(q1_spreads, 0.9)
        <= PILOT_THRESHOLDS["p90_q1_rho_invariance_spread"],
        "G6_monotone_q_paths": float(np.mean(monotone_paths))
        >= PILOT_THRESHOLDS["minimum_monotone_q_path_fraction"],
        "G7_scope_boundary": True,
    }
    return {
        "experiment": EXPERIMENT,
        "evidence_status": "descriptive_cpu_pilot_not_formal_evidence",
        "input": {
            "path": str(projections_path),
            "sha256": sha256_file(projections_path),
            "rows": len(frame),
            "seeds": int(frame["seed"].nunique()),
        },
        "validity": {
            "all_projection_values_finite": bool(finite),
            "unique_seed_cells": int(unique_rows),
            "manifest_hash_match": manifest_match,
            "parameters_unchanged": parameters_unchanged,
        },
        "pairwise_share_estimates": share_estimates,
        "variance_calibration": {
            "ratio_records": ratio_records,
            "median_relative_error": quantile(calibration_errors, 0.5),
            "p90_relative_error": quantile(calibration_errors, 0.9),
            "maximum_relative_error": max(calibration_errors),
        },
        "q1_rho_invariance": {
            "median_spread": quantile(q1_spreads, 0.5),
            "p90_spread": quantile(q1_spreads, 0.9),
            "maximum_spread": max(q1_spreads),
        },
        "monotone_q_paths": {
            "passed": int(sum(monotone_paths)),
            "total": len(monotone_paths),
            "fraction": float(np.mean(monotone_paths)),
        },
        "gates": gates,
        "passed_gates": int(sum(gates.values())),
        "total_gates": len(gates),
        "formal_preregistration_authorized": all(gates.values()),
        "scientific_boundaries": {
            "delay_or_dual_budget_claim": False,
            "online_controller_claim": False,
            "nonlinear_convergence_claim": False,
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    calibration = summary["variance_calibration"]
    invariance = summary["q1_rho_invariance"]
    monotone = summary["monotone_q_paths"]
    gate_lines = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in summary["gates"].items()
    )
    return f"""# EXP-018A CPU pilot validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED
- Evidence: descriptive implementation pilot, not formal evidence

## Result

- rows: {summary['input']['rows']}
- seeds: {summary['input']['seeds']}
- variance calibration median/p90 relative error: {calibration['median_relative_error']:.6f} / {calibration['p90_relative_error']:.6f}
- q=1 rho-invariance median/p90 spread: {invariance['median_spread']:.6f} / {invariance['p90_spread']:.6f}
- monotone q paths: {monotone['passed']}/{monotone['total']} ({monotone['fraction']:.6f})
- formal preregistration authorized: {str(summary['formal_preregistration_authorized']).lower()}

## Frozen gates

{gate_lines}

The pilot tests only the frozen-parameter nonlinear gradient variance identity.
It makes no delayed-learning, dual-budget, online-controller, or nonlinear
convergence claim.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze(args.input.resolve())
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "validation.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
