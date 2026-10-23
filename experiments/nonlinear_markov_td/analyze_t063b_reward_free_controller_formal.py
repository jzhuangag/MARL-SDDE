"""Frozen inference for T-063B with an aggregate collision certificate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.nonlinear_markov_td.analyze_t063a_reward_free_controller_formal import (
    breadth_inference,
    file_sha256,
    matrices,
    ratio_inference,
)
from experiments.nonlinear_markov_td.run_t063b_reward_free_controller_formal import (
    DEFAULT_OUTPUT,
    DEFAULT_SPEC,
    load_config,
    point_analyze,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPRODUCTION = Path(str(DEFAULT_OUTPUT) + "_reproduction")
DEFAULT_TEST_MANIFEST = ROOT / "docs" / "t063b_test_manifest.json"
ARTIFACTS = ("endpoints.csv", "cells.csv", "summary.json")


def replay_equal(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            replay_equal(left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            replay_equal(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)
    return left == right


def execute(
    *,
    config_path: Path = DEFAULT_SPEC,
    primary: Path = DEFAULT_OUTPUT,
    reproduction: Path = DEFAULT_REPRODUCTION,
    test_manifest_path: Path = DEFAULT_TEST_MANIFEST,
) -> dict[str, Any]:
    config = load_config(config_path)
    spec = config["formal_spec"]["t063b"]
    endpoints = pd.read_csv(primary / "endpoints.csv")
    with (primary / "summary.json").open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    with test_manifest_path.open("r", encoding="utf-8") as handle:
        test_manifest = json.load(handle)
    replay = point_analyze(config, endpoints.to_dict("records"))
    replay.pop("cell_rows")
    replay_gate = spec["replay_gate"]
    exact_replay = replay_equal(
        replay,
        stored,
        atol=float(replay_gate["absolute_tolerance"]),
        rtol=float(replay_gate["relative_tolerance"]),
    )
    hashes = {
        name: {
            "primary": file_sha256(primary / name),
            "reproduction": file_sha256(reproduction / name),
            "exact": file_sha256(primary / name) == file_sha256(reproduction / name),
        }
        for name in ARTIFACTS
    }
    exact_reproduction = all(item["exact"] for item in hashes.values())
    formal = config["formal_analysis"]
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
    registry = spec["formal_seed_registry"]
    expected_seeds = set(range(int(registry["inclusive_start"]), int(registry["inclusive_end"]) + 1))
    observed_seeds = set(int(value) for value in endpoints["master_seed"].unique())
    excluded_seeds = {
        seed_value
        for interval in spec["seed_isolation"]["excluded_intervals"]
        for seed_value in range(int(interval[0]), int(interval[1]) + 1)
    }
    expected_endpoints = len(expected_seeds) * 84
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
        "F9_fingerprint_calibration": stored["fingerprint_standardized_rmse"] <= formal["maximum_fingerprint_rmse"],
        "F10_aggregate_collision": bool(stored["t063b_collision_gate"]["pass"]),
        "F11_summary_replay": exact_replay,
        "F12_byte_exact_reproduction": exact_reproduction,
        "F13_full_tests": bool(test_manifest["passed"]),
    }
    return {
        "experiment_id": spec["experiment_id"],
        "classification": "formal evidence under the prospective fixed-policy delayed nonlinear-feature Markov-TD scope",
        "configuration_sha256": spec["configuration_sha256"],
        "artifact_hashes": hashes,
        "bootstrap": {"unit": "complete master-seed column", "replicates": replicates, "seed": seed},
        "aggregate": aggregate,
        "tasks": tasks,
        "delays": delays,
        "strict_cell_breadth": breadth,
        "oracle_proximity": oracle,
        "collision_gate": stored["t063b_collision_gate"],
        "fingerprint_standardized_rmse": stored["fingerprint_standardized_rmse"],
        "blockwise_maximum_rho0_match_rate": stored["maximum_seed_level_rho0_match_rate"],
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
    parser.add_argument("--output", type=Path, required=True)
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
