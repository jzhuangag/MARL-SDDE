"""Read-only paired seed-cluster power audit for the failed T-053A pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINTS = (
    ROOT
    / "experiments"
    / "dependence_delay_linear"
    / "results"
    / "t053a_sampled_cpu_pilot"
    / "endpoints.csv"
)
DEFAULT_THEORY = (
    ROOT
    / "experiments"
    / "dependence_delay_linear"
    / "results"
    / "t052a_exact_fingerprint_static"
    / "results.json"
)
DEFAULT_OUTPUT = ROOT / "docs" / "t054_paired_power_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometric_mean(values: np.ndarray) -> float:
    return float(math.exp(float(np.mean(np.log(values)))))


def task_cluster_arrays(
    endpoints: pd.DataFrame, task: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subset = endpoints[endpoints["task"] == task]
    seeds = np.sort(subset["seed"].unique())
    cells = np.sort(subset["cell_id"].unique())
    controller = np.empty((cells.size, seeds.size))
    strong = np.empty_like(controller)
    for index, cell in enumerate(cells):
        rows = subset[subset["cell_id"] == cell].set_index("seed").loc[seeds]
        controller[index] = rows["controller_risk"].to_numpy()
        strong[index] = rows["strong_fixed_risk"].to_numpy()
    return controller, strong, cells, seeds


def seed_cluster_influence(
    controller: np.ndarray, strong: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Delta-method influence for the geometric mean of cell mean ratios."""

    if controller.shape != strong.shape or controller.ndim != 2:
        raise ValueError("controller and strong arrays must be matching matrices")
    if np.any(controller <= 0.0) or np.any(strong <= 0.0):
        raise ValueError("risks must be positive")
    controller_mean = np.mean(controller, axis=1)
    strong_mean = np.mean(strong, axis=1)
    log_ratio = float(np.mean(np.log(controller_mean / strong_mean)))
    influence = np.mean(
        (controller - controller_mean[:, None]) / controller_mean[:, None]
        - (strong - strong_mean[:, None]) / strong_mean[:, None],
        axis=0,
    )
    standard_deviation = float(np.std(influence, ddof=1))
    return {
        "log_ratio": log_ratio,
        "ratio": math.exp(log_ratio),
        "influence": influence,
        "influence_standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(controller.shape[1]),
    }


def vectorized_cluster_bootstrap(
    controller: np.ndarray,
    strong: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 5_000,
) -> np.ndarray:
    """Resample complete master-seed columns and recompute the frozen statistic."""

    if replicates < 1 or batch_size < 1:
        raise ValueError("replicates and batch_size must be positive")
    rng = np.random.default_rng(seed)
    clusters = controller.shape[1]
    statistics = np.empty(replicates)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, clusters, size=(stop - start, clusters))
        controller_mean = controller[:, indices].mean(axis=2)
        strong_mean = strong[:, indices].mean(axis=2)
        statistics[start:stop] = np.log(controller_mean / strong_mean).mean(axis=0)
    return statistics


def assurance_seed_count(
    *,
    theory_ratio: float,
    gate_ratio: float,
    influence_sd_upper: float,
    assurance: float,
) -> int:
    """Normal-design seed count using an upper confidence bound on cluster SD."""

    if not 0.0 < theory_ratio < gate_ratio < 1.0:
        raise ValueError("theory ratio must be strictly below the gate")
    if influence_sd_upper <= 0.0 or not 0.5 < assurance < 1.0:
        raise ValueError("invalid standard deviation or assurance")
    gap = math.log(gate_ratio) - math.log(theory_ratio)
    return int(math.ceil((norm.ppf(assurance) * influence_sd_upper / gap) ** 2))


def execute(
    *,
    endpoints_path: Path = DEFAULT_ENDPOINTS,
    theory_path: Path = DEFAULT_THEORY,
    bootstrap_replicates: int = 50_000,
    bootstrap_seed: int = 54_001,
) -> dict[str, Any]:
    endpoints = pd.read_csv(endpoints_path)
    with theory_path.open("r", encoding="utf-8") as handle:
        theory = json.load(handle)
    tasks = sorted(endpoints["task"].unique())
    task_results: dict[str, Any] = {}
    required = []
    for task in tasks:
        controller, strong, cells, seeds = task_cluster_arrays(endpoints, task)
        influence = seed_cluster_influence(controller, strong)
        boot = vectorized_cluster_bootstrap(
            controller,
            strong,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        theory_ratios = np.asarray(
            [
                row["expected_risk_ratio"]
                for row in theory["rows"]
                if row["task"] == task and row["delay"] in (0, 8)
            ],
            dtype=float,
        )
        theory_ratio = geometric_mean(theory_ratios)
        degrees = seeds.size - 1
        sd_upper = float(
            influence["influence_standard_deviation"]
            * math.sqrt(degrees / chi2.ppf(0.025, degrees))
        )
        seed_count = assurance_seed_count(
            theory_ratio=theory_ratio,
            gate_ratio=0.97,
            influence_sd_upper=sd_upper,
            assurance=0.95,
        )
        required.append(seed_count)
        task_results[task] = {
            "cells": int(cells.size),
            "pilot_seeds": int(seeds.size),
            "observed_ratio": float(influence["ratio"]),
            "influence_standard_deviation": float(
                influence["influence_standard_deviation"]
            ),
            "influence_standard_error": float(influence["standard_error"]),
            "normal_95_ratio_interval": np.exp(
                [
                    influence["log_ratio"] - 1.96 * influence["standard_error"],
                    influence["log_ratio"] + 1.96 * influence["standard_error"],
                ]
            ).tolist(),
            "bootstrap_95_ratio_interval": np.exp(
                np.quantile(boot, [0.025, 0.975])
            ).tolist(),
            "bootstrap_probability_ratio_at_most_0_97": float(
                np.mean(boot <= math.log(0.97))
            ),
            "T052A_theory_ratio": theory_ratio,
            "influence_sd_one_sided_97_5_upper": sd_upper,
            "seeds_for_95_percent_assurance_under_theory_ratio": seed_count,
        }
    recommended = max(64, 2 ** math.ceil(math.log2(max(required))))
    return {
        "audit_id": "T-054-paired-power-audit",
        "classification": "read-only post-pilot design audit; not new evidence",
        "endpoints_path": endpoints_path.as_posix(),
        "endpoints_sha256": sha256_file(endpoints_path),
        "theory_path": theory_path.as_posix(),
        "theory_sha256": sha256_file(theory_path),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "task_results": task_results,
        "maximum_required_seed_count": max(required),
        "recommended_new_confirmation_seeds": int(recommended),
        "reuse_T053A_seeds": False,
        "formal_authorized": False,
        "gpu_authorized": False,
        "hpc4_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = execute()
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    arguments.output.write_text(data, encoding="utf-8")
    print(data, end="")


if __name__ == "__main__":
    main()
