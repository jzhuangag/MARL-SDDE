"""Post-result audit of the pre-existing closed-form q rule on T-060A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    ROOT
    / "experiments"
    / "nonlinear_markov_td"
    / "results"
    / "t060a_minatar_fixed_q_pilot"
)
DEFAULT_OUTPUT = ROOT / "docs" / "t060a_theory_rule_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def theory_action(*, overhead: int, rho: float, actions: tuple[int, ...] = (1, 4, 16)) -> int:
    return min(actions, key=lambda q: (overhead + q) * (rho + (1.0 - rho) / q))


def cluster_arrays(
    endpoints: pd.DataFrame,
    *,
    strong: dict[str, int],
    task: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple], np.ndarray]:
    frame = endpoints[endpoints["split"] == "validation"].copy()
    if task is not None:
        frame = frame[frame["game"] == task]
    seeds = np.sort(frame["master_seed"].unique())
    cells = sorted(
        set(zip(frame["game"], frame["overhead"], frame["rho"], frame["delay"])),
        key=str,
    )
    proposed = np.empty((len(cells), seeds.size))
    baseline = np.empty_like(proposed)
    indexed = frame.set_index(
        ["game", "overhead", "rho", "delay", "q", "master_seed"]
    )["prediction_risk"]
    for row, (game, overhead, rho, delay) in enumerate(cells):
        q_theory = theory_action(overhead=int(overhead), rho=float(rho))
        q_strong = int(strong[f"{game}|{int(overhead)}"])
        for column, seed in enumerate(seeds):
            proposed[row, column] = indexed[
                game, overhead, rho, delay, q_theory, seed
            ]
            baseline[row, column] = indexed[
                game, overhead, rho, delay, q_strong, seed
            ]
    return proposed, baseline, cells, seeds


def log_geometric_ratio(proposed: np.ndarray, baseline: np.ndarray) -> float:
    return float(np.mean(np.log(proposed.mean(axis=1) / baseline.mean(axis=1))))


def cluster_bootstrap(
    proposed: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 5000,
) -> np.ndarray:
    random = np.random.default_rng(seed)
    clusters = proposed.shape[1]
    values = np.empty(replicates)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = random.integers(0, clusters, size=(stop - start, clusters))
        proposed_mean = proposed[:, indices].mean(axis=2)
        baseline_mean = baseline[:, indices].mean(axis=2)
        values[start:stop] = np.mean(np.log(proposed_mean / baseline_mean), axis=0)
    return values


def summarize_family(
    proposed: np.ndarray,
    baseline: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    point_log = log_geometric_ratio(proposed, baseline)
    bootstrap = cluster_bootstrap(
        proposed,
        baseline,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    cell_ratios = proposed.mean(axis=1) / baseline.mean(axis=1)
    return {
        "ratio": math.exp(point_log),
        "improvement": 1.0 - math.exp(point_log),
        "bootstrap_95_ratio_interval": np.exp(
            np.quantile(bootstrap, [0.025, 0.975])
        ).tolist(),
        "bootstrap_one_sided_95_upper": float(
            math.exp(float(np.quantile(bootstrap, 0.95)))
        ),
        "bootstrap_probability_ratio_at_most_0_95": float(
            np.mean(bootstrap <= math.log(0.95))
        ),
        "strict_cell_fraction": float(np.mean(cell_ratios < 1.0)),
        "cell_ratio_quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(
                (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
                np.quantile(cell_ratios, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)),
            )
        },
    }


def execute(
    *,
    results: Path = DEFAULT_RESULTS,
    bootstrap_replicates: int = 50_000,
    bootstrap_seed: int = 60_001,
) -> dict[str, Any]:
    endpoints_path = results / "endpoints.csv"
    summary_path = results / "summary.json"
    endpoints = pd.read_csv(endpoints_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        original = json.load(handle)
    strong = {key: int(value) for key, value in original["strong_fixed_q"].items()}
    proposed, baseline, cells, seeds = cluster_arrays(endpoints, strong=strong)
    aggregate = summarize_family(
        proposed,
        baseline,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    task_results = {}
    for offset, task in enumerate(sorted(endpoints["game"].unique())):
        task_proposed, task_baseline, task_cells, _ = cluster_arrays(
            endpoints, strong=strong, task=task
        )
        task_results[task] = {
            "cells": len(task_cells),
            **summarize_family(
                task_proposed,
                task_baseline,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + offset + 1,
            ),
        }
    choices = {
        f"h={overhead}": [
            theory_action(overhead=int(overhead), rho=float(rho))
            for rho in sorted(endpoints["rho"].unique())
        ]
        for overhead in sorted(endpoints["overhead"].unique())
    }
    return {
        "audit_id": "T-060A-post-result-theory-rule-audit",
        "classification": "post-result discovery audit; not preregistered evidence and not a T-060A gate",
        "source_rule": "T-050 stationary coefficient (h+q)[rho+(1-rho)/q]",
        "why_not_the_registered_oracle": "T-060A selected a cellwise empirical action on 16 seeds; this audit applies the outcome-independent closed-form action directly",
        "endpoints_sha256": sha256_file(endpoints_path),
        "summary_sha256": sha256_file(summary_path),
        "validation_seed_clusters": int(seeds.size),
        "cells": len(cells),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "theory_choices_over_increasing_rho": choices,
        "aggregate": aggregate,
        "tasks": task_results,
        "T060A_decision_unchanged": True,
        "new_prospective_confirmation_required": True,
        "controller_authorized": False,
        "gpu_authorized": False,
        "hpc4_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = execute()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    arguments.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
