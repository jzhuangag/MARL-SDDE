"""Run the theory-selected delay-extremes persistent-certificate calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.run_t072_dual_use_architecture_calibration import (
    geometric_mean, source_rows, write_csv,
)
from experiments.dependence_delay_linear.run_t074_persistent_certificate_architecture_calibration import (
    run_endpoint as run_t074_endpoint,
)
from experiments.dependence_delay_linear.run_t076_parallel_persistent_calibration import (
    ROOT, scientific_config,
)


DEFAULT_CONFIG = ROOT / "docs/t078_delay_extremes_persistent_calibration.json"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t078_delay_extremes_persistent_calibration"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    delays = {int(value) for value in config["source"]["selected_delay_levels"]}
    return [row for row in source_rows() if int(row["delay"]) in delays]


def validate(config: dict[str, Any]) -> dict[str, Any]:
    rows = selected_rows(config)
    workload = config["workload"]
    if config["source"]["selection_uses_outcomes"] is not False:
        raise ValueError("delay selection must be outcome-free")
    if {int(row["delay"]) for row in rows} != {0, 3}:
        raise ValueError("delay-extremes selection mismatch")
    cells = {row["cell_id"] for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    observed = {"cells": len(cells), "old_design_seeds": len(seeds),
                "endpoints": len(rows),
                "nonstationary_cells": len({row["cell_id"] for row in rows
                                             if row["schedule_family"] != "stationary"})}
    for key, value in observed.items():
        if value != workload[key]:
            raise ValueError(f"frozen workload mismatch for {key}")
    scientific_config(config)
    return {"experiment_id": config["experiment_id"], **observed,
            "outcome_free_delay_selection": True}


def _run_payload(payload: tuple[dict[str, Any], dict[str, str]]) -> dict[str, Any]:
    frozen, row = payload
    return run_t074_endpoint(frozen, row)


def analyze(config: dict[str, Any], endpoints: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in endpoints:
        grouped.setdefault(row["cell_id"], []).append(row)
    cells = []
    for cell_id, rows in sorted(grouped.items()):
        first = rows[0]
        persistent = float(np.mean([row["persistent_auc"] for row in rows]))
        static = float(np.mean([row["strong_static_auc"] for row in rows]))
        cells.append({"cell_id": cell_id,
            **{key: first[key] for key in (
                "schedule_family", "target_scale", "initial_common_parameter", "noise_scale",
                "spatial_correlation", "temporal_correlation", "delay")},
            "mean_persistent_auc": persistent, "mean_strong_static_auc": static,
            "persistent_to_static_ratio": persistent / static})
    nonstationary = [row for row in cells if row["schedule_family"] != "stationary"]
    stationary = [row for row in cells if row["schedule_family"] == "stationary"]
    def ratio(rows: list[dict[str, Any]]) -> float:
        return geometric_mean([row["persistent_to_static_ratio"] for row in rows])
    schedule = {name: ratio([row for row in cells if row["schedule_family"] == name])
                for name in ("single_switch", "alternating")}
    delay = {str(value): ratio([row for row in nonstationary if int(row["delay"]) == value])
             for value in (0, 3)}
    low_t = [row for row in endpoints if float(row["temporal_correlation"]) == 0.0]
    high_t = [row for row in endpoints if float(row["temporal_correlation"]) == 0.9]
    primary = ratio(nonstationary)
    metrics = {
        "nonstationary_ratio": primary, "nonstationary_improvement": 1.0 - primary,
        "strict_cell_fraction": float(np.mean([
            row["persistent_to_static_ratio"] < 1.0 for row in nonstationary])),
        "schedule_ratios": schedule, "delay_ratios": delay,
        "stationary_ratio": ratio(stationary),
        "high_temporal_ratio": ratio([row for row in nonstationary
                                      if float(row["temporal_correlation"]) == 0.9]),
        "low_scale_ratio": ratio([row for row in nonstationary
                                  if float(row["target_scale"]) == 0.1]),
        "low_temporal_mean_rho_upper": float(np.mean([row["mean_rho_upper"] for row in low_t])),
        "high_temporal_mean_rho_upper": float(np.mean([row["mean_rho_upper"] for row in high_t])),
        "low_temporal_mean_effective_samples": float(np.mean([row["mean_effective_samples"] for row in low_t])),
        "high_temporal_mean_effective_samples": float(np.mean([row["mean_effective_samples"] for row in high_t])),
        "mean_iterations_per_recipient_decision": float(np.mean([
            row["mean_iterations_per_recipient_decision"] for row in endpoints])),
        "accepted_nonlocal_rate": float(np.mean([row["accepted_nonlocal_rate"] for row in endpoints])),
        "rollback_rate": float(np.mean([row["rollback_rate"] for row in endpoints])),
        "mean_debt": float(np.mean([row["mean_debt"] for row in endpoints])),
        "max_debt": float(np.max([row["max_debt"] for row in endpoints])),
    }
    finite = len(endpoints) == 9_216 and len(cells) == 288 and all(
        math.isfinite(float(row["persistent_auc"])) for row in endpoints)
    budget = all(row["learning_transitions"] == 240 and row["extra_probe_transitions"] == 0
                 and row["message_units"] <= 18 for row in endpoints)
    gates = {
        "R1": finite, "R2": metrics["nonstationary_improvement"] >= 0.07,
        "R3": metrics["strict_cell_fraction"] >= 0.55,
        "R4": all(1.0 - value >= 0.03 for value in schedule.values()),
        "R5": all(1.0 - value >= 0.03 for value in delay.values()),
        "R6": metrics["stationary_ratio"] <= 1.09,
        "R7": metrics["high_temporal_ratio"] <= 1.01,
        "R8": metrics["low_scale_ratio"] <= 1.06, "R9": budget,
        "R10": metrics["high_temporal_mean_rho_upper"] > metrics["low_temporal_mean_rho_upper"]
               and metrics["high_temporal_mean_effective_samples"] < metrics["low_temporal_mean_effective_samples"],
        "R11": metrics["accepted_nonlocal_rate"] > 0.0 and metrics["mean_debt"] >= 0.0
               and metrics["mean_iterations_per_recipient_decision"] <= 35.0,
        "R12": False, "R13": False, "R14": False,
    }
    return ({"experiment_id": config["experiment_id"], "classification": config["classification"],
             "metrics": metrics, "gates": gates,
             "all_pre_runtime_gates_pass": all(gates[f"R{i}"] for i in range(1, 12)),
             "authorization": config["authorization"]}, cells)


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    frozen = scientific_config(config)
    rows = selected_rows(config)
    workload = config["workload"]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workload["workers"]) as pool:
        endpoints = list(pool.map(_run_payload, ((frozen, row) for row in rows),
                                  chunksize=workload["chunksize"]))
    runtime = time.perf_counter() - started
    summary, cells = analyze(config, endpoints)
    summary["gates"]["R12"] = runtime <= 60.0 * workload["hard_timeout_minutes"]
    summary["runtime_seconds"] = runtime
    summary["all_pre_reproduction_gates_pass"] = all(
        summary["gates"][f"R{i}"] for i in range(1, 13))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cells)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    result = validate(config) if args.command == "validate" else run(config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
