"""Frozen analysis for TSP-V2 online Lyapunov MARL experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
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



def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_counts(accounting: dict) -> dict[int, int]:
    return {int(q): int(count) for q, count in accounting["selected_counts"].items()}


def validate_exact_accounting(
    metadata_path: Path, metadata: dict, progress_rows: list[dict]
) -> None:
    """Reconstruct every physical charge independently of controller history."""

    accounting = metadata["accounting"]
    method = metadata["method"]
    message_budget = int(metadata["message_budget"])
    environment_budget = int(metadata["environment_budget"])
    rollout_length = int(metadata["rollout_length"])
    server_overhead = int(metadata["server_overhead"])
    selected = _selected_counts(accounting)

    if method == "controller":
        ledger_name = metadata.get("decision_ledger")
        ledger_sha = metadata.get("decision_ledger_sha256")
        if not ledger_name or not ledger_sha:
            raise ValueError(f"missing controller ledger: {metadata_path}")
        ledger_path = metadata_path.with_name(ledger_name)
        if not ledger_path.is_file() or _sha256(ledger_path) != ledger_sha:
            raise ValueError(f"controller ledger hash mismatch: {metadata_path}")
        ledger = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        decisions = int(metadata["controller_parameters"]["decisions"])
        if len(ledger) != decisions:
            raise ValueError(f"controller ledger length mismatch: {metadata_path}")
        if [int(row["decision"]) for row in ledger] != list(range(1, decisions + 1)):
            raise ValueError(f"controller decision indices invalid: {metadata_path}")

        candidate_q = sorted(int(q) for q in metadata["candidate_q"])
        validation_q = int(metadata["validation_q"])
        validation_horizon = int(metadata["validation_horizon"])
        train_updates = int(metadata["train_updates_per_decision"])
        expected_counts = {q: 0 for q in candidate_q}
        message_remaining = message_budget
        environment_remaining = environment_budget
        for row in ledger:
            q = int(row["q"])
            if q not in expected_counts:
                raise ValueError(f"unregistered q in ledger: {metadata_path}")
            expected_messages = train_updates * (
                server_overhead + q * rollout_length
            ) + 2 * (server_overhead + validation_q * validation_horizon)
            expected_ticks = train_updates * rollout_length + 2 * validation_horizon
            if int(row["messages"]) != expected_messages:
                raise ValueError(f"message charge mismatch: {metadata_path}")
            if int(row["environment_ticks"]) != expected_ticks:
                raise ValueError(f"environment charge mismatch: {metadata_path}")
            message_remaining -= expected_messages
            environment_remaining -= expected_ticks
            if int(row["message_remaining"]) != message_remaining:
                raise ValueError(f"message recurrence mismatch: {metadata_path}")
            if int(row["environment_remaining"]) != environment_remaining:
                raise ValueError(f"environment recurrence mismatch: {metadata_path}")
            if message_remaining < 0 or environment_remaining < 0:
                raise ValueError(f"controller budget violation: {metadata_path}")
            values = [
                row["risk_before"],
                row["risk_after"],
                row["progress"],
                row["message_queue"],
                row["environment_queue"],
            ]
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                raise ValueError(f"non-finite controller ledger: {metadata_path}")
            expected_counts[q] += 1

        if selected != expected_counts:
            raise ValueError(f"selected counts disagree with ledger: {metadata_path}")
        expected_messages = message_budget - message_remaining
        expected_ticks = environment_budget - environment_remaining
    else:
        if metadata.get("decision_ledger") is not None:
            raise ValueError(f"fixed comparator unexpectedly has ledger: {metadata_path}")
        q = int(method[len("fixed_q") :])
        expected_updates = min(
            message_budget // (server_overhead + q * rollout_length),
            environment_budget // rollout_length,
        )
        if selected != {q: expected_updates}:
            raise ValueError(f"fixed selected count mismatch: {metadata_path}")
        expected_messages = expected_updates * (server_overhead + q * rollout_length)
        expected_ticks = expected_updates * rollout_length
        message_remaining = message_budget - expected_messages
        environment_remaining = environment_budget - expected_ticks

    if int(accounting["charged_messages"]) != expected_messages:
        raise ValueError(f"total message charge mismatch: {metadata_path}")
    if int(accounting["charged_environment_ticks"]) != expected_ticks:
        raise ValueError(f"total environment charge mismatch: {metadata_path}")
    if int(accounting["message_remaining"]) != message_remaining:
        raise ValueError(f"final message balance mismatch: {metadata_path}")
    if int(accounting["environment_remaining"]) != environment_remaining:
        raise ValueError(f"final environment balance mismatch: {metadata_path}")
    if int(progress_rows[-1]["cumulative_messages"]) != expected_messages:
        raise ValueError(f"progress/message total mismatch: {metadata_path}")
    if int(progress_rows[-1]["cumulative_environment_ticks"]) != expected_ticks:
        raise ValueError(f"progress/environment total mismatch: {metadata_path}")


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
        validate_exact_accounting(metadata_path, metadata, rows)
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
        "exact_accounting_audited": True,
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


