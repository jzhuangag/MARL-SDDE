"""Read-only seed-cluster power audit for a prospective T-061 formal run."""

from __future__ import annotations

import argparse
from hashlib import sha256
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
    / "nonlinear_markov_td"
    / "results"
    / "t061a_reward_free_controller_pilot"
    / "endpoints.csv"
)
DEFAULT_OUTPUT = ROOT / "docs" / "t062_t061a_power_audit.json"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matrices(
    endpoints: pd.DataFrame,
    *,
    numerator: str = "controller_risk",
    denominator: str = "strong_risk",
    game: str | None = None,
    delay: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[Any, ...]], np.ndarray]:
    subset = endpoints
    if game is not None:
        subset = subset[subset["game"] == game]
    if delay is not None:
        subset = subset[subset["delay"] == delay]
    cell_columns = ["game", "rho", "overhead", "delay"]
    cell_frame = subset[cell_columns].drop_duplicates().sort_values(cell_columns)
    cells = [tuple(row) for row in cell_frame.itertuples(index=False, name=None)]
    seeds = np.sort(subset["master_seed"].unique())
    first = np.empty((len(cells), seeds.size), dtype=float)
    second = np.empty_like(first)
    for index, cell in enumerate(cells):
        mask = np.logical_and.reduce(
            [subset[column].to_numpy() == value for column, value in zip(cell_columns, cell)]
        )
        rows = subset.loc[mask].set_index("master_seed").loc[seeds]
        first[index] = rows[numerator].to_numpy(dtype=float)
        second[index] = rows[denominator].to_numpy(dtype=float)
    if np.any(first <= 0.0) or np.any(second <= 0.0):
        raise ValueError("all risks must be positive")
    return first, second, cells, seeds


def cluster_influence(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired matrices must have matching two-dimensional shapes")
    first_mean = first.mean(axis=1)
    second_mean = second.mean(axis=1)
    log_ratio = float(np.log(first_mean / second_mean).mean())
    influence = np.mean(
        (first - first_mean[:, None]) / first_mean[:, None]
        - (second - second_mean[:, None]) / second_mean[:, None],
        axis=0,
    )
    standard_deviation = float(np.std(influence, ddof=1))
    degrees = first.shape[1] - 1
    upper = float(
        standard_deviation * math.sqrt(degrees / chi2.ppf(0.025, degrees))
    )
    return {
        "point_ratio": math.exp(log_ratio),
        "log_ratio": log_ratio,
        "influence_standard_deviation": standard_deviation,
        "influence_sd_one_sided_97_5_upper": upper,
    }


def required_seed_count(
    *,
    observed_ratio: float,
    threshold: float,
    influence_sd_upper: float,
    upper_quantile: float,
) -> dict[str, float | int]:
    """Conservative design after shrinking the observed log effect halfway to zero."""

    planning_ratio = math.sqrt(observed_ratio)
    if not 0.0 < planning_ratio < threshold:
        raise ValueError("shrunk planning ratio must be below the formal threshold")
    gap = math.log(threshold) - math.log(planning_ratio)
    count = int(
        math.ceil((norm.ppf(upper_quantile) * influence_sd_upper / gap) ** 2)
    )
    return {
        "planning_ratio_after_50_percent_log_effect_shrinkage": planning_ratio,
        "formal_threshold": threshold,
        "one_sided_upper_quantile": upper_quantile,
        "required_seeds": max(2, count),
    }


def projected_breadth(
    first: np.ndarray,
    second: np.ndarray,
    *,
    sample_sizes: tuple[int, ...],
    replicates: int,
    seed: int,
    batch_size: int = 250,
) -> dict[str, dict[str, float]]:
    """Empirical-cluster projection for the nonsmooth strict-cell fraction."""

    results: dict[str, dict[str, float]] = {}
    for offset, size in enumerate(sample_sizes):
        rng = np.random.default_rng(seed + offset)
        values = np.empty(replicates, dtype=float)
        for start in range(0, replicates, batch_size):
            stop = min(start + batch_size, replicates)
            indices = rng.integers(0, first.shape[1], size=(stop - start, size))
            first_mean = first[:, indices].mean(axis=2)
            second_mean = second[:, indices].mean(axis=2)
            values[start:stop] = np.mean(first_mean < second_mean, axis=0)
        results[str(size)] = {
            "median": float(np.quantile(values, 0.5)),
            "one_sided_05_lower": float(np.quantile(values, 0.05)),
            "one_sided_025_lower": float(np.quantile(values, 0.025)),
            "probability_point_fraction_at_least_0_60": float(np.mean(values >= 0.60)),
        }
    return results


def execute(
    *,
    endpoints_path: Path = DEFAULT_ENDPOINTS,
    bootstrap_replicates: int = 50_000,
    bootstrap_seed: int = 62_001,
) -> dict[str, Any]:
    endpoints = pd.read_csv(endpoints_path)
    all_first, all_second, cells, seeds = matrices(endpoints)
    specifications: list[tuple[str, str | None, int | None, float, float]] = [
        ("aggregate", None, None, 0.95, 0.95),
    ]
    specifications.extend(
        (f"task:{game}", game, None, 0.98, 1.0 - 0.05 / 3.0)
        for game in sorted(endpoints["game"].unique())
    )
    specifications.extend(
        (f"delay:{delay}", None, int(delay), 0.97, 1.0 - 0.05 / 2.0)
        for delay in sorted(endpoints["delay"].unique())
    )
    ratios: dict[str, Any] = {}
    required = []
    for label, game, delay, threshold, quantile in specifications:
        first, second, scoped_cells, scoped_seeds = matrices(
            endpoints, game=game, delay=delay
        )
        influence = cluster_influence(first, second)
        design = required_seed_count(
            observed_ratio=float(influence["point_ratio"]),
            threshold=threshold,
            influence_sd_upper=float(
                influence["influence_sd_one_sided_97_5_upper"]
            ),
            upper_quantile=quantile,
        )
        required.append(int(design["required_seeds"]))
        ratios[label] = {
            "cells": len(scoped_cells),
            "pilot_seed_clusters": int(scoped_seeds.size),
            **influence,
            **design,
        }
    candidate_sizes = (64, 128, 256, 512, 1024)
    breadth = projected_breadth(
        all_first,
        all_second,
        sample_sizes=candidate_sizes,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    breadth_candidates = [
        size
        for size in candidate_sizes
        if breadth[str(size)]["one_sided_025_lower"] >= 0.60
    ]
    breadth_requirement = min(breadth_candidates) if breadth_candidates else None
    ratio_requirement = max(required)
    raw_requirement = max(ratio_requirement, breadth_requirement or candidate_sizes[-1])
    recommended = next(
        (size for size in candidate_sizes if size >= raw_requirement),
        candidate_sizes[-1],
    )
    return {
        "audit_id": "T-062-T061A-seed-cluster-power-audit",
        "classification": "read-only post-pilot design audit; not formal evidence",
        "endpoints_path": endpoints_path.as_posix(),
        "endpoints_sha256": file_sha256(endpoints_path),
        "pilot_seed_clusters": int(seeds.size),
        "cells": len(cells),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "planning_rule": "shrink every observed log-ratio effect by 50 percent toward the null and use the one-sided 97.5 percent upper confidence bound on cluster influence SD",
        "ratio_designs": ratios,
        "strict_breadth_projection": breadth,
        "maximum_ratio_required_seeds": ratio_requirement,
        "strict_breadth_required_seeds": breadth_requirement,
        "recommended_new_formal_seeds": recommended,
        "reuse_T060A_or_T061A_seeds": False,
        "formal_execution_authorized": False,
        "gpu_authorized": False,
        "hpc4_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = execute(endpoints_path=arguments.endpoints)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
