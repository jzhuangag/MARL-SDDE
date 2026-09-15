"""Recover the TSP-CURVE-001 summary without rerunning trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_convergence_curves import (
    CONFIG_PATH,
    load_config,
    sha256,
    summarize_confirmation,
)


TRAJECTORY_SOURCE_SHA256 = "dc4b74231c6882b0eaffb408c1599ac30150f600f6cd6817a3eea0091e3e9532"


def recover(result_dir: Path, selection_path: Path) -> None:
    metrics_path = result_dir / "metrics.csv"
    actions_path = result_dir / "action_table.csv"
    if not metrics_path.exists() or not actions_path.exists():
        raise FileNotFoundError("complete metrics.csv and action_table.csv are required")
    metrics = pd.read_csv(metrics_path)
    expected_rows = 64 * 3 * 2 * 2 * 5 * 41
    if len(metrics) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, found {len(metrics)}")
    if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("metrics contain a non-finite value")
    if not bool(metrics["within_budget"].all()):
        raise RuntimeError("metrics contain an over-budget row")
    config = load_config()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    summary = summarize_confirmation(
        metrics, int(selection["selected_strong_fixed_q"]), config
    )
    summary.update(
        {
            "experiment_id": config["experiment_id"],
            "num_seeds": int(metrics["seed"].nunique()),
            "seeds": sorted(int(value) for value in metrics["seed"].unique()),
            "rows": len(metrics),
            "config_sha256": sha256(CONFIG_PATH),
            "trajectory_source_sha256": TRAJECTORY_SOURCE_SHA256,
            "analysis_source_sha256": sha256(Path(__file__).with_name("run_convergence_curves.py")),
            "metrics_sha256": sha256(metrics_path),
            "action_table_sha256": sha256(actions_path),
            "recovery": {
                "type": "summary-only",
                "reason": "The complete trajectory metrics were written before a NumPy int64 JSON serialization error.",
                "trajectories_rerun": False
            }
        }
    )
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    recover(arguments.result_dir, arguments.selection)
