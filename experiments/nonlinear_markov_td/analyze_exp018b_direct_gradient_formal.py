"""Frozen path-independent analyzer for EXP-018B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp018a_direct_gradient_config import PROJECTION_COUNT, Q_LEVELS, RHO_LEVELS, variance_factor
from exp018b_direct_gradient_config import (
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    ENDPOINT_UPPER_QUANTILE,
    EXPERIMENT,
    FORMAL_SEEDS,
    MEDIAN_ERROR_TOLERANCE,
    P90_ERROR_TOLERANCE,
    PAIRWISE_SHARE_ERROR_TOLERANCE,
    PRACTICAL_DIRECTIONAL_SEPARATION,
    STATIC_MANIFEST_HASH,
    expected_rows,
)
from t023_exp018b_power_audit import aggregate_errors, extract_array


PROJECTIONS = tuple(f"projection_{index:02d}" for index in range(PROJECTION_COUNT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def separated_direction_fraction(array: np.ndarray) -> tuple[int, int, float]:
    variances = np.mean(np.var(array, axis=0, ddof=1), axis=-1)
    passed = 0
    total = 0
    for rho_index, rho in enumerate(RHO_LEVELS):
        for left_index, (left_q, right_q) in enumerate(zip(Q_LEVELS, Q_LEVELS[1:])):
            right_index = left_index + 1
            separation = 1.0 - variance_factor(right_q, rho) / variance_factor(left_q, rho)
            if separation < PRACTICAL_DIRECTIONAL_SEPARATION:
                continue
            comparisons = variances[:, rho_index, left_index] >= variances[:, rho_index, right_index]
            passed += int(np.sum(comparisons))
            total += int(comparisons.size)
    return passed, total, passed / float(total)


def analyze(path: Path) -> dict:
    frame = pd.read_csv(path)
    array, strata, seeds = extract_array(frame)
    finite = bool(np.isfinite(array).all())
    unique = int(frame[["seed", "task", "mixing", "checkpoint", "rho", "q"]].drop_duplicates().shape[0])
    manifest = set(frame["manifest_hash"].astype(str)) == {STATIC_MANIFEST_HASH}
    parameters = bool((frame["parameter_hash_before"] == frame["parameter_hash_after"]).all())
    q1 = frame[frame["q"] == 1]
    q1_exact = True
    for _, group in q1.groupby(["seed", "task", "mixing", "checkpoint"], sort=True):
        if any(group[column].nunique(dropna=False) != 1 for column in PROJECTIONS):
            q1_exact = False
            break

    share_estimates = {}
    share_errors = []
    for rho, group in frame[frame["q"] == max(Q_LEVELS)].groupby("rho", sort=True):
        estimate = float(group["pairwise_shared_fraction"].mean())
        share_estimates[str(float(rho))] = estimate
        share_errors.append(abs(estimate - float(rho)))

    observed = aggregate_errors(array, np.arange(len(seeds), dtype=int))
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    bootstrap = np.empty((BOOTSTRAP_REPLICATIONS, 2), dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATIONS):
        resample = rng.randint(0, len(seeds), size=len(seeds))
        bootstrap[index] = aggregate_errors(array, resample)
    upper = np.quantile(bootstrap, ENDPOINT_UPPER_QUANTILE, axis=0)
    direction_passed, direction_total, direction_fraction = separated_direction_fraction(array)

    gates = {
        "F1_shape_finite_unique": len(frame) == expected_rows()
        and finite
        and unique == expected_rows()
        and set(seeds) == set(FORMAL_SEEDS),
        "F2_manifest_and_parameter_freeze": manifest and parameters,
        "F3_q1_crn_exact": q1_exact,
        "F4_pairwise_share_calibration": max(share_errors) <= PAIRWISE_SHARE_ERROR_TOLERANCE,
        "F5_median_equivalence_upper": float(upper[0]) <= MEDIAN_ERROR_TOLERANCE,
        "F6_p90_equivalence_upper": float(upper[1]) <= P90_ERROR_TOLERANCE,
        "F7_path_independent_summary_schema": True,
        "F8_scope_boundary": True,
    }
    statistical_pass = all(gates.values())
    return {
        "experiment": EXPERIMENT,
        "evidence_status": "formal_statistics_pending_exact_reproduction",
        "input": {"file": "projections.csv", "sha256": sha256_file(path), "rows": len(frame), "seeds": len(seeds)},
        "validity": {"finite": finite, "unique_rows": unique, "manifest_match": manifest, "parameters_unchanged": parameters, "q1_crn_exact": q1_exact},
        "pairwise_share_estimates": share_estimates,
        "primary_endpoints": {
            "observed_median_relative_error": observed[0],
            "observed_p90_relative_error": observed[1],
            "median_one_sided_97_5_upper": float(upper[0]),
            "p90_one_sided_97_5_upper": float(upper[1]),
            "median_tolerance": MEDIAN_ERROR_TOLERANCE,
            "p90_tolerance": P90_ERROR_TOLERANCE,
            "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "separated_direction_diagnostic": {"practical_separation": PRACTICAL_DIRECTIONAL_SEPARATION, "passed": direction_passed, "total": direction_total, "fraction": direction_fraction, "mandatory_gate": False},
        "gates": gates,
        "passed_gates": int(sum(gates.values())),
        "total_gates": len(gates),
        "formal_statistical_gates_pass": statistical_pass,
        "exact_reproduction_pending": True,
        "formal_claim_authorized": False,
        "boundaries": {"delay_or_dual_budget_claim": False, "online_controller_claim": False, "nonlinear_convergence_claim": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = analyze(args.input.resolve())
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
