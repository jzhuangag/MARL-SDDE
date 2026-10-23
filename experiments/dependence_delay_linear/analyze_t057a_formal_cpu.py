"""Frozen seed-cluster analysis for T-057A formal CPU evidence."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.dependence_delay_linear.run_t055a_confirmation_cpu_pilot import (
    analyze as base_analyze,
)
from experiments.dependence_delay_linear.run_t057a_formal_cpu import load_config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    ROOT
    / "experiments"
    / "dependence_delay_linear"
    / "results"
    / "t057a_formal_cpu"
)
MANIFEST = ROOT / "docs" / "t057a_formal_reproduction_manifest.json"
DEFAULT_OUTPUT = ROOT / "docs" / "validation_t057a_formal_cpu.json"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_seed_matrices(
    endpoints: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cells = np.sort(endpoints["cell_id"].unique())
    if mask is not None:
        if mask.shape != (cells.size,):
            raise ValueError("cell mask has the wrong shape")
        cells = cells[mask]
    seeds = np.sort(endpoints["seed"].unique())
    first = np.empty((cells.size, seeds.size))
    second = np.empty_like(first)
    for index, cell in enumerate(cells):
        rows = endpoints[endpoints["cell_id"] == cell].set_index("seed").loc[seeds]
        first[index] = rows[numerator].to_numpy()
        second[index] = rows[denominator].to_numpy()
    return first, second


def cluster_bootstrap_log_statistic(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 100,
) -> np.ndarray:
    if numerator.shape != denominator.shape or numerator.ndim != 2:
        raise ValueError("paired risk matrices must have the same two dimensions")
    if np.any(numerator <= 0.0) or np.any(denominator <= 0.0):
        raise ValueError("risks must be positive")
    rng = np.random.default_rng(seed)
    clusters = numerator.shape[1]
    statistics = np.empty(replicates)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, clusters, size=(stop - start, clusters))
        first_mean = numerator[:, indices].mean(axis=2)
        second_mean = denominator[:, indices].mean(axis=2)
        statistics[start:stop] = np.log(first_mean / second_mean).mean(axis=0)
    return statistics


def ratio_inference(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    upper_quantile: float,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, float | bool]:
    point = float(
        np.exp(np.log(numerator.mean(axis=1) / denominator.mean(axis=1)).mean())
    )
    boot = cluster_bootstrap_log_statistic(
        numerator, denominator, replicates=replicates, seed=seed
    )
    upper = float(np.exp(np.quantile(boot, upper_quantile)))
    return {
        "point_ratio": point,
        "one_sided_upper_quantile": upper_quantile,
        "one_sided_upper_ratio": upper,
        "threshold": threshold,
        "pass": point <= threshold and upper <= threshold,
    }


def active_fraction_inference(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    lower_quantile: float,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, float | bool]:
    point = float(np.mean(numerator.mean(axis=1) < denominator.mean(axis=1)))
    rng = np.random.default_rng(seed)
    clusters = numerator.shape[1]
    values = np.empty(replicates)
    for start in range(0, replicates, 100):
        stop = min(start + 100, replicates)
        indices = rng.integers(0, clusters, size=(stop - start, clusters))
        first_mean = numerator[:, indices].mean(axis=2)
        second_mean = denominator[:, indices].mean(axis=2)
        values[start:stop] = np.mean(first_mean < second_mean, axis=0)
    lower = float(np.quantile(values, lower_quantile))
    return {
        "point_fraction": point,
        "one_sided_lower_quantile": lower_quantile,
        "one_sided_lower_fraction": lower,
        "threshold": threshold,
        "pass": point >= threshold and lower >= threshold,
    }


def execute(*, results: Path = RESULTS, manifest_path: Path = MANIFEST) -> dict[str, Any]:
    config = load_config()
    endpoints = pd.read_csv(results / "endpoints.csv")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (results / "summary.json").open("r", encoding="utf-8") as handle:
        stored_summary = json.load(handle)
    replay = base_analyze(config, endpoints.to_dict("records"))
    replay.pop("cell_rows")
    bootstrap = config["formal_analysis"]["bootstrap"]
    replicates = int(bootstrap["replicates"])
    seed = int(bootstrap["seed"])

    cells = np.sort(endpoints["cell_id"].unique())
    first_rows = endpoints.drop_duplicates("cell_id").set_index("cell_id").loc[cells]
    active_mask = first_rows["oracle_q"].to_numpy() != first_rows["baseline_q"].to_numpy()
    task_masks = {
        task: first_rows["task"].to_numpy() == task for task in config["tasks"]
    }
    delay_masks = {
        str(delay): first_rows["delay"].to_numpy() == delay
        for delay in config["grid"]["delays"]
    }

    all_strong = cell_seed_matrices(
        endpoints, numerator="controller_risk", denominator="strong_fixed_risk"
    )
    aggregate = ratio_inference(
        *all_strong,
        upper_quantile=0.95,
        threshold=0.95,
        replicates=replicates,
        seed=seed,
    )
    task_results = {
        task: ratio_inference(
            *cell_seed_matrices(
                endpoints,
                numerator="controller_risk",
                denominator="strong_fixed_risk",
                mask=mask,
            ),
            upper_quantile=1.0 - 0.05 / len(task_masks),
            threshold=0.97,
            replicates=replicates,
            seed=seed,
        )
        for task, mask in task_masks.items()
    }
    delay_results = {
        label: ratio_inference(
            *cell_seed_matrices(
                endpoints,
                numerator="controller_risk",
                denominator="strong_fixed_risk",
                mask=mask,
            ),
            upper_quantile=1.0 - 0.05 / len(delay_masks),
            threshold=0.97,
            replicates=replicates,
            seed=seed,
        )
        for label, mask in delay_masks.items()
    }
    active = active_fraction_inference(
        *cell_seed_matrices(
            endpoints,
            numerator="controller_risk",
            denominator="strong_fixed_risk",
            mask=active_mask,
        ),
        lower_quantile=0.05,
        threshold=0.60,
        replicates=replicates,
        seed=seed,
    )
    inactive = ratio_inference(
        *cell_seed_matrices(
            endpoints,
            numerator="controller_risk",
            denominator="strong_fixed_risk",
            mask=~active_mask,
        ),
        upper_quantile=0.95,
        threshold=1.05,
        replicates=replicates,
        seed=seed,
    )
    oracle = ratio_inference(
        *cell_seed_matrices(
            endpoints,
            numerator="controller_risk",
            denominator="true_rho_oracle_risk",
        ),
        upper_quantile=0.95,
        threshold=1.20,
        replicates=replicates,
        seed=seed,
    )

    artifact_hashes = {
        name: sha256_file(results / name)
        for name in ("endpoints.csv", "cells.csv", "summary.json")
    }
    reproduction = (
        manifest["primary_artifacts"] == artifact_hashes
        and manifest["clean_rerun_artifacts"] == artifact_hashes
        and manifest["byte_identical"]
    )
    expected_seeds = set(config["pilot_seeds"])
    observed_seeds = set(int(value) for value in endpoints["seed"].unique())
    seed_registry = observed_seeds == expected_seeds
    gates = {
        "F1_provenance_and_coverage": (
            len(endpoints) == config["expected_workload"]["endpoints"]
            and endpoints["cell_id"].nunique() == config["expected_workload"]["cells"]
            and seed_registry
        ),
        "F2_finite_and_budget_valid": bool(stored_summary["gates"]["P2"]),
        "F3_aggregate_inference": bool(aggregate["pass"]),
        "F4_taskwise_simultaneous_inference": all(
            bool(item["pass"]) for item in task_results.values()
        ),
        "F5_delay_simultaneous_inference": all(
            bool(item["pass"]) for item in delay_results.values()
        ),
        "F6_active_breadth_inference": bool(active["pass"]),
        "F7_inactive_no_harm_inference": bool(inactive["pass"]),
        "F8_oracle_proximity_inference": bool(oracle["pass"]),
        "F9_participation_direction": bool(stored_summary["gates"]["P9"]),
        "F10_fingerprint_calibration": bool(stored_summary["gates"]["P10"]),
        "F11_summary_replay": replay["gates"] == stored_summary["gates"],
        "F12_byte_reproduction": reproduction,
        "F13_seed_isolation": bool(manifest["seed_isolation_verified"]),
        "F14_full_tests": bool(manifest["full_tests_pass"]),
    }
    return {
        "experiment_id": config["experiment_id"],
        "classification": "formal evidence under the preregistered fixed-policy Markov-TD scope",
        "artifact_sha256": artifact_hashes,
        "bootstrap": bootstrap,
        "aggregate": aggregate,
        "tasks": task_results,
        "delays": delay_results,
        "active_breadth": active,
        "inactive_no_harm": inactive,
        "oracle_proximity": oracle,
        "fingerprint_standardized_rmse": stored_summary[
            "fingerprint_standardized_rmse"
        ],
        "gates": gates,
        "all_formal_gates_pass": all(gates.values()),
        "gpu_authorized": False,
        "hpc4_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = execute(results=arguments.results, manifest_path=arguments.manifest)
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    arguments.output.write_text(data, encoding="utf-8")
    print(data, end="")


if __name__ == "__main__":
    main()
