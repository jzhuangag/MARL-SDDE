"""Frozen analysis for TSP-V2 online Lyapunov MARL experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = ("controller", "fixed_q1", "fixed_q2", "fixed_q4", "fixed_q8")


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty progress file: {path}")
    fractions = np.asarray([row["budget_fraction"] for row in rows], dtype=float)
    returns = np.asarray(
        [row["evaluation_mean_step_return"] for row in rows], dtype=float
    )
    if not np.isfinite(fractions).all() or not np.isfinite(returns).all():
        raise ValueError(f"non-finite progress: {path}")
    if fractions[0] != 0.0 or np.any(np.diff(fractions) < 0.0):
        raise ValueError(f"invalid budget axis: {path}")
    return rows


def normalized_auc(rows: list[dict], points: int = 101) -> float:
    fractions = np.asarray([row["budget_fraction"] for row in rows], dtype=float)
    values = np.asarray(
        [row["evaluation_mean_step_return"] for row in rows], dtype=float
    )
    keep = np.r_[True, np.diff(fractions) > 0.0]
    fractions = fractions[keep]
    values = values[keep]
    grid = np.linspace(0.0, 1.0, points)
    interpolated = np.interp(grid, fractions, values, left=values[0], right=values[-1])
    return float(np.trapz(interpolated, grid))


def load_runs(root: Path) -> list[dict]:
    runs = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        progress_path = metadata_path.with_name("progress.jsonl")
        rows = _read_jsonl(progress_path)
        accounting = metadata["accounting"]
        if accounting["charged_messages"] > metadata["message_budget"]:
            raise ValueError(f"message budget violation: {metadata_path}")
        if accounting["charged_environment_ticks"] > metadata["environment_budget"]:
            raise ValueError(f"environment budget violation: {metadata_path}")
        if metadata["evaluation_used_by_controller"]:
            raise ValueError(f"evaluation leakage: {metadata_path}")
        run = {
            **metadata,
            "auc": normalized_auc(rows),
            "initial_return": rows[0]["evaluation_mean_step_return"],
            "final_return": rows[-1]["evaluation_mean_step_return"],
            "progress_rows": len(rows),
            "metadata_path": str(metadata_path),
        }
        runs.append(run)
    return runs


def analyze(root: Path, expected_seeds: list[int]) -> tuple[list[dict], dict]:
    runs = load_runs(root)
    keys = [
        (run["environment"], run["task"], run["coupling"], run["method"], run["seed"])
        for run in runs
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate experiment cell")
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["environment"], run["task"], run["coupling"])].append(run)
    expected_methods = set(METHODS)
    summaries = []
    for cell, subset in sorted(grouped.items()):
        observed = {(run["method"], run["seed"]) for run in subset}
        expected = {(method, seed) for method in METHODS for seed in expected_seeds}
        if observed != expected:
            raise ValueError(f"incomplete cell {cell}: expected {expected}, got {observed}")
        fixed_means = {
            method: float(np.mean([r["auc"] for r in subset if r["method"] == method]))
            for method in expected_methods - {"controller"}
        }
        best_fixed = max(fixed_means, key=lambda method: (fixed_means[method], method))
        controller_runs = sorted(
            [run for run in subset if run["method"] == "controller"],
            key=lambda run: run["seed"],
        )
        baseline_runs = sorted(
            [run for run in subset if run["method"] == best_fixed],
            key=lambda run: run["seed"],
        )
        controller_auc = float(np.mean([r["auc"] for r in controller_runs]))
        baseline_auc = float(np.mean([r["auc"] for r in baseline_runs]))
        controller_final = float(np.mean([r["final_return"] for r in controller_runs]))
        baseline_final = float(np.mean([r["final_return"] for r in baseline_runs]))
        auc_improvement = (controller_auc - baseline_auc) / max(abs(baseline_auc), 1e-12)
        final_improvement = (controller_final - baseline_final) / max(
            abs(baseline_final), 1e-12
        )
        q_counts = defaultdict(int)
        for run in controller_runs:
            for q, count in run["accounting"]["selected_counts"].items():
                q_counts[q] += int(count)
        summaries.append(
            {
                "environment": cell[0],
                "task": cell[1],
                "coupling": cell[2],
                "best_fixed_method": best_fixed,
                "controller_auc": controller_auc,
                "best_fixed_auc": baseline_auc,
                "auc_relative_improvement": auc_improvement,
                "controller_final_return": controller_final,
                "best_fixed_final_return": baseline_final,
                "final_relative_improvement": final_improvement,
                "controller_q_counts": dict(sorted(q_counts.items())),
            }
        )
    task_groups = defaultdict(list)
    for row in summaries:
        task_groups[(row["environment"], row["task"])].append(row)
    task_summary = {
        f"{environment}:{task}": {
            "mean_auc_relative_improvement": float(
                np.mean([row["auc_relative_improvement"] for row in subset])
            ),
            "mean_final_relative_improvement": float(
                np.mean([row["final_relative_improvement"] for row in subset])
            ),
        }
        for (environment, task), subset in sorted(task_groups.items())
    }
    gates = {
        "complete_cells": len(summaries) == 4,
        "dual_budget_valid": True,
        "no_evaluation_leakage": True,
        "each_task_auc_gain_at_least_1_percent": all(
            value["mean_auc_relative_improvement"] >= 0.01
            for value in task_summary.values()
        ),
        "each_task_final_noninferior_2_percent": all(
            value["mean_final_relative_improvement"] >= -0.02
            for value in task_summary.values()
        ),
        "nondegenerate_online_selection": all(
            sum(count > 0 for count in row["controller_q_counts"].values()) >= 2
            for row in summaries
        ),
    }
    decision = {
        "runs": len(runs),
        "cells": len(summaries),
        "task_summary": task_summary,
        "gates": gates,
        "development_pass": all(gates.values()),
        "confirmation_authorized": all(gates.values()),
    }
    return summaries, decision


def write_outputs(rows: list[dict], decision: dict, csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "environment",
        "task",
        "coupling",
        "best_fixed_method",
        "controller_auc",
        "best_fixed_auc",
        "auc_relative_improvement",
        "controller_final_return",
        "best_fixed_final_return",
        "final_relative_improvement",
        "controller_q_counts",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "controller_q_counts": json.dumps(row["controller_q_counts"], sort_keys=True)})
    json_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    rows, decision = analyze(args.root, args.seeds)
    write_outputs(rows, decision, args.csv, args.json)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
