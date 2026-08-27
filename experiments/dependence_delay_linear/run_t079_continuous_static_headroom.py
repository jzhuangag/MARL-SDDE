"""Run the preregistered T-079 continuous-static headroom audit."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.run_t070a_exact_nonstationary_graph_scan import (
    scenarios,
    unit_schedule,
)
from experiments.dependence_delay_linear.run_t072_dual_use_architecture_calibration import (
    geometric_mean,
    write_csv,
)
from experiments.dependence_delay_linear.t079_continuous_static_graph import (
    catalogue_graph_to_weights,
    optimize_static_graph,
    simulate_continuous_graph,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t079_continuous_static_headroom_preregistration.json"
PARENT_CONFIG = ROOT / "docs/t070a_exact_nonstationary_graph_preregistration.json"
SOURCE_CELLS = ROOT / "experiments/dependence_delay_linear/results/t070a_exact_nonstationary_graph_scan/cells.csv"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t079_continuous_static_headroom"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_parent() -> dict[str, Any]:
    return json.loads(PARENT_CONFIG.read_text(encoding="utf-8"))


def source_cells() -> dict[str, dict[str, str]]:
    with SOURCE_CELLS.open(newline="", encoding="utf-8") as handle:
        return {row["cell_id"]: row for row in csv.DictReader(handle)}


def validate(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    if sha256(PARENT_CONFIG) != source["parent_config_sha256"]:
        raise ValueError("parent configuration hash mismatch")
    if sha256(SOURCE_CELLS) != source["discrete_cells_sha256"]:
        raise ValueError("discrete static source hash mismatch")
    if source["source_role"] != "initialization_and_dominance_check_only":
        raise ValueError("source role mismatch")
    parent = load_parent()
    if config["model"] != parent["model"]:
        raise ValueError("T-079 model must equal the frozen parent model")
    if config["grid"] != parent["grid"]:
        raise ValueError("T-079 grid must equal the frozen parent grid")
    if config["static_optimizer"]["uses_sampled_outcomes"] is not False:
        raise ValueError("sampled outcomes are forbidden")
    if config["dynamic_oracle"]["extra_probe_transitions"] != 0:
        raise ValueError("dual-use dynamic oracle cannot consume extra probes")
    rows = scenarios(config)
    sources = source_cells()
    if set(sources) != {row["cell_id"] for row in rows}:
        raise ValueError("source cell identity mismatch")
    observed = {
        "cells": len(rows),
        "nonstationary_cells": sum(row["schedule_family"] != "stationary" for row in rows),
        "optimizer_starts": config["static_optimizer"]["deterministic_starts"],
    }
    for key, value in observed.items():
        if value != config["expected_workload"][key]:
            raise ValueError(f"workload mismatch for {key}")
    return {"experiment_id": config["experiment_id"], **observed,
            "scientific_outcome_created": False}


def _simulate(
    config: dict[str, Any], scenario: dict[str, Any], weights: np.ndarray | None,
    *, dynamic: bool,
):
    model = config["model"]
    schedule = scenario["target_scale"] * unit_schedule(config, scenario["schedule_family"])
    return simulate_continuous_graph(
        target_schedule=schedule,
        initial_parameter=scenario["initial_common_parameter"],
        delay=scenario["delay"],
        decision_blocks=model["decision_blocks"],
        gain=model["gain"],
        curvature=model["curvature"],
        local_steps=model["learning_steps_per_block"],
        noise_scale=scenario["noise_scale"],
        spatial_correlation=scenario["spatial_correlation"],
        temporal_correlation=scenario["temporal_correlation"],
        fixed_weights=weights,
        safe_dynamic_oracle=dynamic,
        fingerprint_message_units=config["budget"]["fingerprint_message_units_per_decision"],
        mixing_message_units=config["budget"]["mixing_message_units_per_decision"],
    )


def run_cell(payload: tuple[dict[str, Any], dict[str, Any], dict[str, str]]) -> dict[str, Any]:
    config, scenario, source = payload
    model = config["model"]
    graph = np.asarray(json.loads(source["best_static_graph"]), dtype=int)
    discrete_weights = catalogue_graph_to_weights(
        graph, agents=model["agents"], alpha_grid=config["actions"]["alpha"]
    )
    discrete = _simulate(config, scenario, discrete_weights, dynamic=False)
    optimized = optimize_static_graph(
        lambda matrix: _simulate(config, scenario, matrix, dynamic=False),
        agents=model["agents"],
        discrete_start=discrete_weights,
        maximum_iterations=config["static_optimizer"]["maximum_iterations_per_start"],
    )
    dynamic = _simulate(config, scenario, None, dynamic=True)
    changes = int(np.sum([
        np.max(np.abs(dynamic.weights_path[index] - dynamic.weights_path[index - 1]))
        > config["analysis"]["graph_change_tolerance"]
        for index in range(1, len(dynamic.weights_path))
    ]))
    return {
        **scenario,
        "source_discrete_static_auc": float(source["best_static_auc"]),
        "replayed_discrete_static_auc": discrete.auc_risk,
        "continuous_static_auc": optimized.auc_risk,
        "continuous_static_terminal": optimized.terminal_risk,
        "continuous_static_weights": json.dumps(optimized.weights.tolist(), separators=(",", ":")),
        "continuous_to_discrete_static_ratio": optimized.auc_risk / discrete.auc_risk,
        "dynamic_continuous_auc": dynamic.auc_risk,
        "dynamic_continuous_terminal": dynamic.terminal_risk,
        "dynamic_to_continuous_static_ratio": dynamic.auc_risk / optimized.auc_risk,
        "dynamic_graph_change_count": changes,
        "dynamic_shadow_rate": float(np.mean(dynamic.used_shadow)),
        "dynamic_maximum_row_kkt_residual": dynamic.maximum_row_kkt_residual,
        "learning_transitions": dynamic.learning_transitions,
        "extra_probe_transitions": dynamic.extra_probe_transitions,
        "dynamic_message_units": dynamic.message_units,
        "static_successful_starts": optimized.successful_starts,
        "static_total_starts": optimized.total_starts,
        "static_best_start_index": optimized.best_start_index,
        "static_row_sum_residual": optimized.row_sum_residual,
        "static_minimum_weight": optimized.minimum_weight,
    }


def analyze(config: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    nonstationary = [row for row in cells if row["schedule_family"] != "stationary"]
    stationary = [row for row in cells if row["schedule_family"] == "stationary"]

    def ratio(rows: list[dict[str, Any]], key: str = "dynamic_to_continuous_static_ratio") -> float:
        return geometric_mean([float(row[key]) for row in rows])

    primary = ratio(nonstationary)
    family = {
        name: ratio([row for row in nonstationary if row["schedule_family"] == name])
        for name in ("single_switch", "alternating")
    }
    delay = {
        str(value): ratio([row for row in nonstationary if int(row["delay"]) == value])
        for value in config["grid"]["delay"]
    }
    source_replay_relative = max(
        abs(float(row["replayed_discrete_static_auc"]) - float(row["source_discrete_static_auc"]))
        / float(row["source_discrete_static_auc"])
        for row in cells
    )
    metrics = {
        "nonstationary_dynamic_to_continuous_static_ratio": primary,
        "nonstationary_dynamic_improvement": 1.0 - primary,
        "nonstationary_strict_cell_fraction": float(np.mean([
            float(row["dynamic_to_continuous_static_ratio"]) < 1.0 for row in nonstationary
        ])),
        "stationary_dynamic_to_continuous_static_ratio": ratio(stationary),
        "schedule_ratios": family,
        "delay_ratios": delay,
        "continuous_to_discrete_static_ratio": ratio(cells, "continuous_to_discrete_static_ratio"),
        "continuous_static_strictly_strengthened_fraction": float(np.mean([
            float(row["continuous_to_discrete_static_ratio"]) < 1.0 - 1e-10 for row in cells
        ])),
        "dynamic_graph_change_fraction": float(np.mean([
            int(row["dynamic_graph_change_count"]) > 0 for row in nonstationary
        ])),
        "mean_dynamic_shadow_rate": float(np.mean([
            float(row["dynamic_shadow_rate"]) for row in cells
        ])),
        "maximum_dynamic_row_kkt_residual": max(
            float(row["dynamic_maximum_row_kkt_residual"]) for row in cells
        ),
        "maximum_source_replay_relative_error": source_replay_relative,
        "minimum_successful_static_starts": min(int(row["static_successful_starts"]) for row in cells),
        "maximum_static_row_sum_residual": max(float(row["static_row_sum_residual"]) for row in cells),
        "minimum_static_weight": min(float(row["static_minimum_weight"]) for row in cells),
    }
    complete = len(cells) == config["expected_workload"]["cells"] and all(
        math.isfinite(float(row[key])) and float(row[key]) > 0.0
        for row in cells
        for key in ("continuous_static_auc", "dynamic_continuous_auc")
    )
    budget = all(
        int(row["learning_transitions"]) == config["budget"]["environment_transitions"]
        and int(row["extra_probe_transitions"]) == 0
        and int(row["dynamic_message_units"]) <= config["budget"]["message_units"]
        for row in cells
    )
    gates = {
        "H1": complete,
        "H2": metrics["maximum_source_replay_relative_error"] <= 1e-10,
        "H3": all(float(row["continuous_to_discrete_static_ratio"]) <= 1.0 + 1e-10 for row in cells),
        "H4": metrics["minimum_successful_static_starts"] >= 1
              and metrics["maximum_static_row_sum_residual"] <= 1e-9
              and metrics["minimum_static_weight"] >= -1e-12,
        "H5": budget,
        "H6": metrics["maximum_dynamic_row_kkt_residual"] <= 1e-7,
        "H7": metrics["nonstationary_dynamic_improvement"] >= 0.05,
        "H8": metrics["nonstationary_strict_cell_fraction"] >= 0.60,
        "H9": all(1.0 - value >= 0.03 for value in family.values()),
        "H10": all(1.0 - value >= 0.0 for value in delay.values()),
        "H11": 1.0 - metrics["stationary_dynamic_to_continuous_static_ratio"] <= 0.05,
        "H12": metrics["dynamic_graph_change_fraction"] >= 0.60,
        "H13": False,
        "H14": False,
    }
    return {
        "experiment_id": config["experiment_id"],
        "classification": config["classification"],
        "metrics": metrics,
        "gates": gates,
        "all_pre_reproduction_gates_pass": all(gates[f"H{i}"] for i in range(1, 13)),
        "authorization": config["authorization"],
    }


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    sources = source_cells()
    payloads = [(config, row, sources[row["cell_id"]]) for row in scenarios(config)]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=config["expected_workload"]["workers"]) as pool:
        cells = list(pool.map(run_cell, payloads, chunksize=1))
    summary = analyze(config, cells)
    output.mkdir(parents=True, exist_ok=False)
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
