"""Create publication summaries for TSP-CURVE-001."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(result_dir: Path, output_csv: Path, output_json: Path) -> None:
    metrics_path = result_dir / "metrics.csv"
    summary_path = result_dir / "summary.json"
    metrics = pd.read_csv(metrics_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    group_columns = [
        "persistence",
        "rho",
        "maximum_delay",
        "policy",
        "q",
        "gap",
        "eta",
        "checkpoint",
    ]
    aggregate = (
        metrics.groupby(group_columns, as_index=False, sort=True)
        .agg(
            resource_fraction=("resource_fraction", "mean"),
            parameter_mean=("normalized_parameter_error", "mean"),
            parameter_std=("normalized_parameter_error", "std"),
            return_mean=("normalized_return_error", "mean"),
            return_std=("normalized_return_error", "std"),
            seeds=("seed", "nunique"),
        )
    )
    root_n = np.sqrt(aggregate["seeds"].to_numpy(dtype=float))
    aggregate["parameter_ci95"] = 1.96 * aggregate["parameter_std"] / root_n
    aggregate["return_ci95"] = 1.96 * aggregate["return_std"] / root_n
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_csv, index=False)
    comparisons = summary["paired_comparisons"]
    cell_ratios = np.asarray(
        [row["parameter_auc_ratio"] for row in summary["cell_comparisons"]],
        dtype=float,
    )
    payload = {
        "experiment_id": summary["experiment_id"],
        "confirmation_seeds": summary["num_seeds"],
        "raw_rows": summary["rows"],
        "selected_strong_fixed_q": summary["selected_strong_fixed_q"],
        "parameter_auc_reduction": 1.0 - comparisons["parameter_auc"]["ratio"],
        "parameter_auc_ratio_ci95": [
            comparisons["parameter_auc"]["ci95_low"],
            comparisons["parameter_auc"]["ci95_high"],
        ],
        "return_auc_reduction": 1.0 - comparisons["return_auc"]["ratio"],
        "return_auc_ratio_ci95": [
            comparisons["return_auc"]["ci95_low"],
            comparisons["return_auc"]["ci95_high"],
        ],
        "terminal_parameter_reduction": 1.0 - comparisons["terminal_parameter"]["ratio"],
        "terminal_return_reduction": 1.0 - comparisons["terminal_return"]["ratio"],
        "cells_improved": int(np.sum(cell_ratios < 1.0 - 1e-12)),
        "cells_tied": int(np.sum(np.abs(cell_ratios - 1.0) <= 1e-12)),
        "cells_total": int(len(cell_ratios)),
        "finite": bool(summary["finite"]),
        "within_budget": bool(summary["within_budget"]),
        "metrics_sha256": sha256(metrics_path),
        "source_summary_sha256": sha256(summary_path),
        "aggregate_csv_sha256": sha256(output_csv),
        "scientific_recovery": summary["recovery"],
    }
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analyze(arguments.result_dir, arguments.output_csv, arguments.output_json)
