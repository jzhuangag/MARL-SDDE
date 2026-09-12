"""Run the explicitly tainted T-073 continuous-QP architecture calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.run_t070a_exact_nonstationary_graph_scan import unit_schedule
from experiments.dependence_delay_linear.run_t071a_sampled_observable_graph_pilot import (
    load_config as load_t071_config,
    load_t070_config,
)
from experiments.dependence_delay_linear.run_t072_dual_use_architecture_calibration import (
    SOURCE_ENDPOINTS,
    geometric_mean,
    sha256,
    source_rows,
    write_csv,
)
from experiments.dependence_delay_linear.t071_sampled_graph_controller import (
    sample_markov_observations,
    stable_seed,
)
from experiments.dependence_delay_linear.t073_continuous_qp_controller import (
    simulate_continuous_qp_controller,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t073_continuous_qp_architecture_calibration.json"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t073_continuous_qp_architecture_calibration"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(config: dict[str, Any]) -> dict[str, Any]:
    rows = source_rows()
    source = config["source"]
    controller = config["controller"]
    if sha256(SOURCE_ENDPOINTS) != source["endpoints_sha256"]:
        raise ValueError("T-071A endpoint hash mismatch")
    if source["reuse_all_source_seeds"] is not True or source["new_scientific_seeds"] is not False:
        raise ValueError("calibration taint declaration is invalid")
    if len(rows) != 13_824 or len({row["cell_id"] for row in rows}) != 432:
        raise ValueError("source coverage mismatch")
    if controller["all_steps_are_learning_updates"] is not True or controller["extra_probe_transitions"] != 0:
        raise ValueError("dual-use accounting mismatch")
    t071 = load_t071_config()
    if controller["drift_weight"] != t071["model"]["agents"]:
        raise ValueError("drift weight is not the agent count")
    if controller["variance_weight"] != 1.0:
        raise ValueError("covariance coefficient is not the frozen unit value")
    if controller["rollback_margin"] != -0.5 * t071["model"]["gain"]:
        raise ValueError("rollback margin is not minus half the gain")
    return {
        "experiment_id": config["experiment_id"], "cells": 432,
        "source_seeds": 32, "endpoints": 13_824,
        "tainted_design_evidence_only": True,
    }


def run_endpoint(config: dict[str, Any], source: dict[str, str]) -> dict[str, Any]:
    t071 = load_t071_config()
    parent = load_t070_config()
    controller = config["controller"]
    targets = float(source["target_scale"]) * unit_schedule(parent, source["schedule_family"])
    observations, _ = sample_markov_observations(
        targets=targets, steps_per_block=t071["model"]["learning_steps_per_block"],
        noise_scale=float(source["noise_scale"]),
        spatial_correlation=float(source["spatial_correlation"]),
        temporal_correlation=float(source["temporal_correlation"]),
        seed=stable_seed(int(source["seed"]), source["cell_id"], "observations"),
    )
    result = simulate_continuous_qp_controller(
        observations=observations, targets=targets,
        initial_parameter=float(source["initial_common_parameter"]),
        gain=t071["model"]["gain"], delay=int(source["delay"]),
        decision_blocks=t071["model"]["decision_blocks"],
        pre_steps=controller["pre_steps"], drift_weight=controller["drift_weight"],
        variance_weight=controller["variance_weight"], safety_slack=controller["safety_slack"],
        rollback_margin=controller["rollback_margin"],
        fingerprint_message_units=controller["fingerprint_message_units_per_decision"],
        mixing_message_units=controller["mixing_message_units_per_decision"],
    )
    identity = np.eye(t071["model"]["agents"])
    accepted_nonlocal = np.max(np.abs(result.accepted_weights - identity), axis=2) > 1e-8
    return {
        **{key: source[key] for key in (
            "cell_id", "schedule_family", "target_scale", "initial_common_parameter",
            "noise_scale", "spatial_correlation", "temporal_correlation", "delay", "seed",
        )},
        "strong_static_auc": float(source["strong_static_auc"]),
        "continuous_auc": result.auc_risk,
        "continuous_terminal": result.terminal_risk,
        "learning_transitions": result.learning_transitions,
        "extra_probe_transitions": 0,
        "fingerprint_reused_transitions": result.fingerprint_transitions,
        "message_units": result.message_units,
        "qp_iterations": result.qp_iterations,
        "mean_iterations_per_recipient_decision": result.qp_iterations / 24.0,
        "mean_debt": float(np.mean(result.debt_path)),
        "max_debt": float(np.max(result.debt_path)),
        "rollback_rate": float(np.mean(result.rollback_states)),
        "accepted_nonlocal_rate": float(np.mean(accepted_nonlocal)),
    }


def analyze(config: dict[str, Any], endpoints: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in endpoints:
        grouped.setdefault(row["cell_id"], []).append(row)
    cells = []
    for cell_id, rows in sorted(grouped.items()):
        first = rows[0]
        continuous = float(np.mean([row["continuous_auc"] for row in rows]))
        static = float(np.mean([row["strong_static_auc"] for row in rows]))
        cells.append({
            "cell_id": cell_id,
            **{key: first[key] for key in (
                "schedule_family", "target_scale", "initial_common_parameter", "noise_scale",
                "spatial_correlation", "temporal_correlation", "delay",
            )},
            "mean_continuous_auc": continuous,
            "mean_strong_static_auc": static,
            "continuous_to_static_ratio": continuous / static,
        })
    nonstationary = [row for row in cells if row["schedule_family"] != "stationary"]
    stationary = [row for row in cells if row["schedule_family"] == "stationary"]
    primary = geometric_mean([row["continuous_to_static_ratio"] for row in nonstationary])
    schedule = {}
    for family in ("single_switch", "alternating"):
        ratio = geometric_mean([
            row["continuous_to_static_ratio"] for row in cells if row["schedule_family"] == family
        ])
        schedule[family] = {"ratio": ratio, "improvement": 1.0 - ratio}
    delay = {}
    for value in (0, 1, 3):
        ratio = geometric_mean([
            row["continuous_to_static_ratio"] for row in nonstationary if int(row["delay"]) == value
        ])
        delay[str(value)] = {"ratio": ratio, "improvement": 1.0 - ratio}
    metrics = {
        "nonstationary_ratio": primary,
        "nonstationary_improvement": 1.0 - primary,
        "nonstationary_strict_cell_fraction": float(np.mean([
            row["continuous_to_static_ratio"] < 1.0 for row in nonstationary
        ])),
        "stationary_ratio": geometric_mean([
            row["continuous_to_static_ratio"] for row in stationary
        ]),
        "schedule": schedule, "delay": delay,
        "mean_iterations_per_recipient_decision": float(np.mean([
            row["mean_iterations_per_recipient_decision"] for row in endpoints
        ])),
        "maximum_iterations_per_recipient_decision": 50,
        "mean_debt": float(np.mean([row["mean_debt"] for row in endpoints])),
        "max_debt": float(np.max([row["max_debt"] for row in endpoints])),
        "rollback_rate": float(np.mean([row["rollback_rate"] for row in endpoints])),
        "accepted_nonlocal_rate": float(np.mean([
            row["accepted_nonlocal_rate"] for row in endpoints
        ])),
    }
    finite_complete = len(endpoints) == 13_824 and len(cells) == 432 and all(
        math.isfinite(float(row["continuous_auc"])) for row in endpoints
    )
    budget = all(
        row["learning_transitions"] == 240 and row["extra_probe_transitions"] == 0
        and row["message_units"] <= 18 for row in endpoints
    )
    criteria = {
        "Q1": finite_complete,
        "Q2": metrics["nonstationary_improvement"] >= 0.05,
        "Q3": metrics["nonstationary_strict_cell_fraction"] >= 0.55,
        "Q4": all(value["improvement"] >= 0.02 for value in schedule.values()),
        "Q5": all(value["improvement"] >= 0.0 for value in delay.values()),
        "Q6": metrics["stationary_ratio"] <= 1.10,
        "Q7": budget,
        "Q8": metrics["accepted_nonlocal_rate"] > 0.0 and metrics["mean_debt"] >= 0.0,
        "Q9": metrics["mean_iterations_per_recipient_decision"] <= 35.0,
    }
    return ({
        "experiment_id": config["experiment_id"], "classification": config["classification"],
        "metrics": metrics, "descriptive_criteria": criteria,
        "all_descriptive_criteria_pass": all(criteria.values()),
        "authorization": config["authorization"],
    }, cells)


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    started = time.perf_counter()
    endpoints = [run_endpoint(config, row) for row in source_rows()]
    summary, cells = analyze(config, endpoints)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cells)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**summary, "runtime_seconds": time.perf_counter() - started}


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
