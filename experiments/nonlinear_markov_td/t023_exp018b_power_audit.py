"""Pilot-informed, outcome-aware power audit for a prospective EXP-018B.

This audit cannot authorize a formal run by itself.  It estimates uncertainty
for frozen aggregate calibration functionals using only EXP-018A design data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp018a_direct_gradient_config import PROJECTION_COUNT, Q_LEVELS, variance_factor


PROJECTIONS = tuple(f"projection_{index:02d}" for index in range(PROJECTION_COUNT))
STRATA = ("task", "mixing", "checkpoint")
RHO_LEVELS = (0.0, 0.5, 0.9)
FORMAL_N = 192
BOOTSTRAP_REPLICATIONS = 2_000
BOOTSTRAP_SEED = 18_230_101
MEDIAN_TOLERANCE = 0.20
P90_TOLERANCE = 0.50
ONE_SIDED_ENDPOINT_QUANTILE = 0.975


def extract_array(frame: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, str, str]], list[int]]:
    seeds = sorted(int(seed) for seed in frame["seed"].unique())
    strata = sorted(
        tuple(str(value) for value in row)
        for row in frame[list(STRATA)].drop_duplicates().itertuples(index=False, name=None)
    )
    array = np.empty(
        (len(seeds), len(strata), len(RHO_LEVELS), len(Q_LEVELS), len(PROJECTIONS)),
        dtype=np.float64,
    )
    seed_index = {value: index for index, value in enumerate(seeds)}
    stratum_index = {value: index for index, value in enumerate(strata)}
    rho_index = {value: index for index, value in enumerate(RHO_LEVELS)}
    q_index = {value: index for index, value in enumerate(Q_LEVELS)}
    for row in frame.itertuples(index=False):
        s = seed_index[int(row.seed)]
        t = stratum_index[(str(row.task), str(row.mixing), str(row.checkpoint))]
        r = rho_index[float(row.rho)]
        q = q_index[int(row.q)]
        array[s, t, r, q] = [float(getattr(row, name)) for name in PROJECTIONS]
    return array, strata, seeds


def aggregate_errors(array: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    selected = array[indices]
    projection_variances = np.var(selected, axis=0, ddof=1)
    cell_variances = np.mean(projection_variances, axis=-1)
    errors = []
    for rho_index, rho in enumerate(RHO_LEVELS):
        baseline = cell_variances[:, rho_index, 0]
        for q_index, q in enumerate(Q_LEVELS[1:], start=1):
            ratio = cell_variances[:, rho_index, q_index] / baseline
            errors.extend(np.abs(ratio / variance_factor(q, rho) - 1.0).tolist())
    return float(np.median(errors)), float(np.quantile(errors, 0.9))


def audit(path: Path) -> dict:
    frame = pd.read_csv(path)
    array, strata, seeds = extract_array(frame)
    observed = aggregate_errors(array, np.arange(len(seeds), dtype=int))
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    boot = np.empty((BOOTSTRAP_REPLICATIONS, 2), dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATIONS):
        resample = rng.randint(0, len(seeds), size=FORMAL_N)
        boot[index] = aggregate_errors(array, resample)
    upper = np.quantile(boot, ONE_SIDED_ENDPOINT_QUANTILE, axis=0)
    pass_projection = bool(upper[0] <= MEDIAN_TOLERANCE and upper[1] <= P90_TOLERANCE)
    return {
        "task": "T-023",
        "status": "pilot_informed_power_audit_not_formal_evidence",
        "source_seed_count": len(seeds),
        "strata_count": len(strata),
        "prospective_formal_seed_count": FORMAL_N,
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "familywise_alpha": 0.05,
        "endpoint_quantile": ONE_SIDED_ENDPOINT_QUANTILE,
        "observed_pilot_aggregate": {
            "median_relative_error": observed[0],
            "p90_relative_error": observed[1],
        },
        "prospective_192_seed_bootstrap": {
            "median_error_upper": float(upper[0]),
            "p90_error_upper": float(upper[1]),
            "median_tolerance": MEDIAN_TOLERANCE,
            "p90_tolerance": P90_TOLERANCE,
            "static_feasibility_pass": pass_projection,
        },
        "formal_run_authorized": False,
        "scientific_outcomes_generated": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
