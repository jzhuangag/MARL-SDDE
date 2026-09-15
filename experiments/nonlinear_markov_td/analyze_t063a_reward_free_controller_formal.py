"""Frozen complete-seed-cluster inference for T-063A formal evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.nonlinear_markov_td.run_t061a_reward_free_controller_pilot import (
    analyze as point_analyze,
    file_sha256,
)
from experiments.nonlinear_markov_td.run_t063a_reward_free_controller_formal import (
    DEFAULT_OUTPUT,
    DEFAULT_SPEC,
    load_config,
)
from experiments.nonlinear_markov_td.t062_t061a_power_audit import matrices


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPRODUCTION = Path(str(DEFAULT_OUTPUT) + "_reproduction")
DEFAULT_TEST_MANIFEST = ROOT / "docs" / "t063a_test_manifest.json"
DEFAULT_VALIDATION = ROOT / "docs" / "validation_t063a_reward_free_controller_formal.json"
ARTIFACTS = ("endpoints.csv", "cells.csv", "summary.json")


def cluster_bootstrap_log_ratio(
    first: np.ndarray,
    second: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 100,
) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired matrices must have matching two-dimensional shapes")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    clusters = first.shape[1]
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, clusters, size=(stop - start, clusters))
        first_mean = first[:, indices].mean(axis=2)
        second_mean = second[:, indices].mean(axis=2)
        values[start:stop] = np.log(first_mean / second_mean).mean(axis=0)
    return values


def ratio_inference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    threshold: float,
    upper_quantile: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    point = float(np.exp(np.log(first.mean(axis=1) / second.mean(axis=1)).mean()))
    draws = cluster_bootstrap_log_ratio(
        first, second, replicates=replicates, seed=seed
    )
    upper = float(np.exp(np.quantile(draws, upper_quantile)))
    return {
        "point_ratio": point,
        "one_sided_upper_quantile": upper_quantile,
        "one_sided_upper_ratio": upper,
        "threshold": threshold,
        "pass": point <= threshold and upper <= threshold,
    }


def breadth_inference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    threshold: float,
    lower_quantile: float,
    replicates: int,
    seed: int,
    batch_size: int = 100,
) -> dict[str, Any]:
    point = float(np.mean(first.mean(axis=1) < second.mean(axis=1)))
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    clusters = first.shape[1]
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, clusters, size=(stop - start, clusters))
        first_mean = first[:, indices].mean(axis=2)
        second_mean = second[:, indices].mean(axis=2)
        values[start:stop] = np.mean(first_mean < second_mean, axis=0)
    lower = float(np.quantile(values, lower_quantile))
    return {
        "point_fraction": point,
        "one_sided_lower_quantile": lower_quantile,
        "one_sided_lower_fraction": lower,
        "threshold": threshold,
        "pass": point >= threshold and lower >= threshold,
    }


def execute(
    *,
    config_path: Path = DEFAULT_SPEC,
    primary: Path = DEFAULT_OUTPUT,
    reproduction: Path = DEFAULT_REPRODUCTION,
    test_manifest_path: Path = DEFAULT_TEST_MANIFEST,
) -> dict[str, Any]:
    config = load_config(config_path)
    spec = config["formal_spec"]
    endpoints = pd.read_csv(primary / "endpoints.csv")
    with (primary / "summary.json").open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    with test_manifest_path.open("r", encoding="utf-8") as handle:
        test_manifest = json.load(handle)
    replay = point_analyze(config, endpoints.to_dict("records"))
    replay.pop("cell_rows")
    exact_replay = replay == stored
    hashes = {
        name: {
            "primary": file_sha256(primary / name),
            "reproduction": file_sha256(reproduction / name),
            "exact": file_sha256(primary / name) == file_sha256(reproduction / name),
        }
        for name in ARTIFACTS
    }
    exact_reproduction = all(item["exact"] for item in hashes.values())
    formal = spec["formal_analysis"]
    replicates = int(formal["bootstrap_replicates"])
    seed = int(formal["bootstrap_seed"])
    aggregate = ratio_inference(
        *matrices(endpoints)[:2],
        threshold=float(formal["aggregate_threshold"]),
        upper_quantile=float(formal["aggregate_upper_quantile"]),
        replicates=replicates,
        seed=seed,
    )
    games = sorted(endpoints["game"].unique())
    tasks = {
        game: ratio_inference(
            *matrices(endpoints, game=game)[:2],
            threshold=float(formal["task_threshold"]),
            upper_quantile=1.0 - 0.05 / len(games),
            replicates=replicates,
            seed=seed + index + 1,
        )
        for index, game in enumerate(games)
    }
    delays_values = sorted(int(value) for value in endpoints["delay"].unique())
    delays = {
        str(delay): ratio_inference(
            *matrices(endpoints, delay=delay)[:2],
            threshold=float(formal["delay_threshold"]),
            upper_quantile=1.0 - 0.05 / len(delays_values),
            replicates=replicates,
            seed=seed + 10 + index,
        )
        for index, delay in enumerate(delays_values)
    }
    breadth = breadth_inference(
        *matrices(endpoints)[:2],
        threshold=float(formal["breadth_threshold"]),
        lower_quantile=float(formal["breadth_lower_quantile"]),
        replicates=replicates,
        seed=seed + 20,
    )
    oracle = ratio_inference(
        *matrices(
            endpoints,
            numerator="controller_risk",
            denominator="true_rho_full_budget_risk",
        )[:2],
        threshold=float(formal["oracle_threshold"]),
        upper_quantile=float(formal["oracle_upper_quantile"]),
        replicates=replicates,
        seed=seed + 21,
    )
    expected_seeds = set(config["pilot_seeds"])
    observed_seeds = set(int(value) for value in endpoints["master_seed"].unique())
    excluded_seeds = {
        seed_value
        for interval in spec["seed_isolation"]["excluded_intervals"]
        for seed_value in range(int(interval[0]), int(interval[1]) + 1)
    }
    expected_endpoints = len(expected_seeds) * int(spec["expected_workload"]["cells"])
    gates = {
        "F1_coverage_and_seed_isolation": len(endpoints) == expected_endpoints
        and observed_seeds == expected_seeds
        and excluded_seeds.isdisjoint(observed_seeds),
        "F2_finite_and_full_cost": bool(stored["gates"]["P2_finite_full_cost"]),
        "F3_aggregate_simultaneous_gain": bool(aggregate["pass"]),
        "F4_taskwise_simultaneous_gain": all(bool(row["pass"]) for row in tasks.values()),
        "F5_delay_simultaneous_gain": all(bool(row["pass"]) for row in delays.values()),
        "F6_strict_cell_breadth": bool(breadth["pass"]),
        "F7_true_rho_oracle_proximity": bool(oracle["pass"]),
        "F8_participation_direction": stored["rho_directional_paths"] == "12/12",
        "F9_fingerprint_calibration": stored["fingerprint_standardized_rmse"]
        <= formal["maximum_fingerprint_rmse"],
        "F10_independent_collision": stored["maximum_seed_level_rho0_match_rate"]
        <= formal["maximum_seed_level_rho0_match_rate"],
        "F11_summary_replay": exact_replay,
        "F12_byte_exact_reproduction": exact_reproduction,
        "F13_full_tests": bool(test_manifest["passed"]),
    }
    return {
        "experiment_id": spec["experiment_id"],
        "classification": "formal evidence under the preregistered fixed-policy delayed nonlinear-feature Markov-TD scope",
        "configuration_sha256": spec["configuration_sha256"],
        "artifact_hashes": hashes,
        "bootstrap": {
            "unit": "complete master-seed column",
            "replicates": replicates,
            "seed": seed,
        },
        "aggregate": aggregate,
        "tasks": tasks,
        "delays": delays,
        "strict_cell_breadth": breadth,
        "oracle_proximity": oracle,
        "fingerprint_standardized_rmse": stored["fingerprint_standardized_rmse"],
        "maximum_seed_level_rho0_match_rate": stored[
            "maximum_seed_level_rho0_match_rate"
        ],
        "gates": gates,
        "all_formal_gates_pass": all(gates.values()),
        "gpu_used": False,
        "hpc4_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--primary", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reproduction", type=Path, default=DEFAULT_REPRODUCTION)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    arguments = parser.parse_args()
    result = execute(
        config_path=arguments.config,
        primary=arguments.primary,
        reproduction=arguments.reproduction,
        test_manifest_path=arguments.test_manifest,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
