"""Independent validation and seed-cluster uncertainty audit for T-055A."""

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
    analyze,
    load_config,
)
from experiments.dependence_delay_linear.run_t054_paired_power_audit import (
    vectorized_cluster_bootstrap,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    ROOT
    / "experiments"
    / "dependence_delay_linear"
    / "results"
    / "t055a_confirmation_cpu_pilot"
)
MANIFEST = ROOT / "docs" / "t055a_reproduction_manifest.json"
DEFAULT_OUTPUT = ROOT / "docs" / "validation_t055a_confirmation_cpu_pilot.json"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_matrices(
    endpoints: pd.DataFrame, *, task: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    subset = endpoints if task is None else endpoints[endpoints["task"] == task]
    seeds = np.sort(subset["seed"].unique())
    cells = np.sort(subset["cell_id"].unique())
    controller = np.empty((cells.size, seeds.size))
    strong = np.empty_like(controller)
    for index, cell in enumerate(cells):
        rows = subset[subset["cell_id"] == cell].set_index("seed").loc[seeds]
        controller[index] = rows["controller_risk"].to_numpy()
        strong[index] = rows["strong_fixed_risk"].to_numpy()
    return controller, strong


def observed_ratio(controller: np.ndarray, strong: np.ndarray) -> float:
    return float(
        math.exp(
            float(
                np.mean(
                    np.log(np.mean(controller, axis=1) / np.mean(strong, axis=1))
                )
            )
        )
    )


def cluster_interval(
    controller: np.ndarray,
    strong: np.ndarray,
    *,
    gate: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    boot = vectorized_cluster_bootstrap(
        controller,
        strong,
        replicates=replicates,
        seed=seed,
        batch_size=500,
    )
    return {
        "observed_ratio": observed_ratio(controller, strong),
        "seed_cluster_bootstrap_95_interval": np.exp(
            np.quantile(boot, [0.025, 0.975])
        ).tolist(),
        "bootstrap_probability_ratio_below_1": float(np.mean(boot < 0.0)),
        "bootstrap_probability_ratio_at_most_gate": float(
            np.mean(boot <= math.log(gate))
        ),
        "gate": gate,
    }


def maximum_numeric_difference(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return math.inf
        return max(
            (maximum_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max(
            (maximum_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if (
        isinstance(left, (int, float, np.integer, np.floating))
        and not isinstance(left, (bool, np.bool_))
        and isinstance(right, (int, float, np.integer, np.floating))
        and not isinstance(right, (bool, np.bool_))
    ):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def execute(
    *, bootstrap_replicates: int = 50_000, bootstrap_seed: int = 55_001
) -> dict[str, Any]:
    config = load_config()
    endpoints = pd.read_csv(RESULTS / "endpoints.csv")
    with (RESULTS / "summary.json").open("r", encoding="utf-8") as handle:
        stored_summary = json.load(handle)
    with MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    primary_hashes = {
        name: sha256_file(RESULTS / name)
        for name in ("endpoints.csv", "cells.csv", "summary.json")
    }
    provenance_pass = primary_hashes == manifest["primary_artifacts"]
    reproduction_pass = (
        manifest["byte_identical"]
        and manifest["primary_artifacts"] == manifest["clean_rerun_artifacts"]
        and provenance_pass
    )
    recomputed = analyze(config, endpoints.to_dict("records"))
    recomputed.pop("cell_rows")
    summary_replay_max_abs_error = maximum_numeric_difference(
        recomputed, stored_summary
    )
    summary_replay_pass = summary_replay_max_abs_error <= 1e-12

    tasks = sorted(endpoints["task"].unique())
    uncertainty = {
        "aggregate": cluster_interval(
            *cluster_matrices(endpoints),
            gate=0.95,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "tasks": {
            task: cluster_interval(
                *cluster_matrices(endpoints, task=task),
                gate=0.97,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            for task in tasks
        },
    }
    final_gates = {**stored_summary["gates"], "P12": reproduction_pass}
    return {
        "experiment_id": config["experiment_id"],
        "classification": "independent confirmation pilot; not formal evidence",
        "configuration_sha256": sha256_file(
            ROOT / "docs" / "t055a_confirmation_cpu_pilot_preregistration.json"
        ),
        "artifact_sha256": primary_hashes,
        "endpoint_rows": int(len(endpoints)),
        "unique_cells": int(endpoints["cell_id"].nunique()),
        "unique_seeds": int(endpoints["seed"].nunique()),
        "summary_replay_pass": summary_replay_pass,
        "summary_replay_max_abs_error": summary_replay_max_abs_error,
        "summary_replay_tolerance": 1e-12,
        "provenance_pass": provenance_pass,
        "reproduction_pass": reproduction_pass,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "seed_cluster_uncertainty": uncertainty,
        "frozen_summary": stored_summary,
        "final_gates": final_gates,
        "all_12_gates_pass": all(final_gates.values()),
        "formal_preregistration_authorized": all(final_gates.values()),
        "formal_execution_authorized": False,
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
