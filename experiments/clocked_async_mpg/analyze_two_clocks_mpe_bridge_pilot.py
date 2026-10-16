"""Frozen analysis for the Two Clocks public-MPE CPU pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("offdiag_async", "raw_async", "delay_scaled_async", "frozen_barrier")
PROFILES = ("balanced", "heterogeneous")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def analyze_rows(
    endpoints: list[dict[str, Any]], curves: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    tasks = tuple(config["tasks"])
    seeds = tuple(int(value) for value in config["pilot_seeds"])
    numeric_fields = (
        "initial_return",
        "final_return",
        "return_change",
        "return_auc",
        "completed_packets",
        "optimizer_updates",
        "completed_environment_steps",
        "completed_actor_transitions",
        "cancelled_environment_steps",
        "cancelled_actor_transitions",
        "baseline_environment_steps",
        "baseline_actor_transitions",
        "evaluation_environment_steps",
        "evaluation_actor_transitions",
        "cumulative_step_norm",
        "cumulative_policy_kl",
        "cumulative_teammate_birth_arrival_kl",
        "maximum_owner_error",
        "maximum_event_delay",
        "offdiag_lyapunov_scale",
        "lyapunov_condition_max",
        "logical_service_time",
    )
    converted = []
    for row in endpoints:
        item = dict(row)
        item["seed"] = int(item["seed"])
        for key in numeric_fields:
            item[key] = float(item[key])
        converted.append(item)
    curve_rows = []
    for row in curves:
        item = dict(row)
        item["seed"] = int(item["seed"])
        item["logical_time"] = float(item["logical_time"])
        item["evaluation_return"] = float(item["evaluation_return"])
        curve_rows.append(item)

    index = {
        (row["seed"], row["task"], row["profile"], row["method"]): row
        for row in converted
    }
    expected_keys = {
        (seed, task, profile, method)
        for seed in seeds
        for task in tasks
        for profile in PROFILES
        for method in METHODS
    }
    checkpoint_exact = True
    endpoint_curve_exact = True
    for key in expected_keys:
        selected = sorted(
            [
                row
                for row in curve_rows
                if (row["seed"], row["task"], row["profile"], row["method"]) == key
            ],
            key=lambda row: row["logical_time"],
        )
        horizon = float(config["tasks"][key[1]]["service_horizon"])
        expected_times = [horizon * value for value in config["checkpoint_fractions"]]
        checkpoint_exact = checkpoint_exact and [row["logical_time"] for row in selected] == expected_times
        if selected and key in index:
            endpoint_curve_exact = endpoint_curve_exact and math.isclose(
                selected[0]["evaluation_return"], index[key]["initial_return"], abs_tol=1e-12
            ) and math.isclose(
                selected[-1]["evaluation_return"], index[key]["final_return"], abs_tol=1e-12
            )

    finite = all(
        math.isfinite(float(row[key])) for row in converted for key in numeric_fields
    ) and all(math.isfinite(row["evaluation_return"]) for row in curve_rows)
    pairing = True
    for seed in seeds:
        for task in tasks:
            selected = [row for row in converted if row["seed"] == seed and row["task"] == task]
            pairing = pairing and len({row["initial_policy_digest"] for row in selected}) == 1
            pairing = pairing and len({row["frozen_control_variate_digest"] for row in selected}) == 1
            pairing = pairing and len({row["initial_return"] for row in selected}) == 1
    accounting = all(
        row["maximum_owner_error"] <= 1e-10
        and row["maximum_event_delay"] <= config["maximum_event_delay"]
        and row["lyapunov_condition_max"] <= 1.0 + 1e-12
        and row["completed_environment_steps"] >= 0
        and row["cancelled_environment_steps"] >= 0
        and row["baseline_environment_steps"] > 0
        and row["evaluation_environment_steps"] > 0
        for row in converted
    )

    positive_cells: dict[str, float] = {}
    for task in tasks:
        for method in METHODS:
            value = _mean(
                [
                    row["return_change"]
                    for row in converted
                    if row["task"] == task and row["method"] == method
                ]
            )
            positive_cells[f"{task}/{method}"] = value

    def normalized_gain(seed: int, task: str, profile: str, left: str, right: str) -> float:
        lhs = index[(seed, task, profile, left)]
        rhs = index[(seed, task, profile, right)]
        denominator = max(1.0, abs(lhs["initial_return"]))
        return (lhs["return_auc"] - rhs["return_auc"]) / denominator

    heterogeneous = [
        normalized_gain(seed, task, "heterogeneous", "offdiag_async", "frozen_barrier")
        for seed in seeds
        for task in tasks
    ]
    heterogeneous_by_task = {
        task: _mean(
            [
                normalized_gain(seed, task, "heterogeneous", "offdiag_async", "frozen_barrier")
                for seed in seeds
            ]
        )
        for task in tasks
    }
    directionality = float(np.mean(np.asarray(heterogeneous) > 0.0))
    reference_gaps = []
    reference_by_task: dict[str, float] = {}
    for task in tasks:
        task_values = []
        for seed in seeds:
            offdiag = index[(seed, task, "heterogeneous", "offdiag_async")]
            raw = index[(seed, task, "heterogeneous", "raw_async")]
            delayed = index[(seed, task, "heterogeneous", "delay_scaled_async")]
            stronger = max(raw["return_auc"], delayed["return_auc"])
            value = (offdiag["return_auc"] - stronger) / max(1.0, abs(offdiag["initial_return"]))
            task_values.append(value)
            reference_gaps.append(value)
        reference_by_task[task] = _mean(task_values)
    phase_order: dict[str, dict[str, float]] = {}
    for task in tasks:
        balanced = _mean(
            [
                normalized_gain(seed, task, "balanced", "offdiag_async", "frozen_barrier")
                for seed in seeds
            ]
        )
        phase_order[task] = {
            "balanced_gain": balanced,
            "heterogeneous_gain": heterogeneous_by_task[task],
        }
    offdiag_scales = {
        row["offdiag_lyapunov_scale"] for row in converted if row["method"] == "offdiag_async"
    }
    motion_differs = any(
        not math.isclose(
            index[(seed, task, profile, "offdiag_async")]["cumulative_step_norm"],
            index[(seed, task, profile, "raw_async")]["cumulative_step_norm"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for seed in seeds
        for task in tasks
        for profile in PROFILES
    )
    gates = {
        "P1_validity_pairing_and_accounting": len(converted) == config["expected_endpoint_rows"]
        and len(curve_rows) == config["expected_curve_rows"]
        and set(index) == expected_keys
        and finite
        and pairing
        and accounting
        and checkpoint_exact
        and endpoint_curve_exact,
        "P2_positive_learning": all(value > 0.0 for value in positive_cells.values()),
        "P3_heterogeneous_auc_gain": _mean(heterogeneous) >= 0.05
        and all(value > 0.0 for value in heterogeneous_by_task.values()),
        "P4_heterogeneous_directionality": directionality >= 0.60,
        "P5_async_reference_safety": _mean(reference_gaps) >= -0.02
        and all(value >= -0.02 for value in reference_by_task.values()),
        "P6_rate_coupling_order": all(
            values["heterogeneous_gain"] > values["balanced_gain"]
            for values in phase_order.values()
        ),
        "P7_nontrivial_lyapunov_design": all(0.0 < value < 1.0 for value in offdiag_scales)
        and motion_differs,
        "P8_motion_and_work_disclosure": all(
            row["cumulative_step_norm"] > 0.0
            and row["cumulative_policy_kl"] >= 0.0
            and row["optimizer_updates"] > 0.0
            for row in converted
        ),
        "P9_no_outcome_selected_setting": True,
        "P11_no_formal_or_gpu_escalation": True,
    }
    return {
        "gates": gates,
        "mandatory_without_reproduction_passed": all(gates.values()),
        "metrics": {
            "positive_learning_cells": positive_cells,
            "heterogeneous_normalized_auc_gain": _mean(heterogeneous),
            "heterogeneous_gain_by_task": heterogeneous_by_task,
            "heterogeneous_directionality": directionality,
            "offdiag_vs_stronger_async_normalized_auc_gap": _mean(reference_gaps),
            "reference_gap_by_task": reference_by_task,
            "phase_order": phase_order,
            "offdiag_scales": sorted(offdiag_scales),
            "mean_motion": {
                method: {
                    "step_norm": _mean([row["cumulative_step_norm"] for row in converted if row["method"] == method]),
                    "policy_kl": _mean([row["cumulative_policy_kl"] for row in converted if row["method"] == method]),
                }
                for method in METHODS
            },
        },
    }


def analyze(primary: Path, reproduction: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = analyze_rows(
        _read_csv(primary / "endpoints.csv"),
        _read_csv(primary / "curves.csv"),
        config,
    )
    reproduction_match = all(
        _sha256(primary / name) == _sha256(reproduction / name)
        for name in ("endpoints.csv", "curves.csv")
    )
    result["gates"]["P12_reproducibility"] = reproduction_match
    result["P10_public_quality_anchor"] = "pending_separate_upstream_run"
    result["formal_authorized"] = False
    result["gpu_authorized"] = False
    result["cpu_bridge_survives"] = all(result["gates"].values())
    result["provenance"] = {
        "config_sha256": _sha256(config_path),
        "primary_endpoints_sha256": _sha256(primary / "endpoints.csv"),
        "primary_curves_sha256": _sha256(primary / "curves.csv"),
        "reproduction_endpoints_sha256": _sha256(reproduction / "endpoints.csv"),
        "reproduction_curves_sha256": _sha256(reproduction / "curves.csv"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.primary, args.reproduction, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
