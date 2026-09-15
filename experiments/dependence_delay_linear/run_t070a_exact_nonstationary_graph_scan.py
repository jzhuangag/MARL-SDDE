"""Run the preregistered T-070A exact nonstationary graph feasibility scan."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.t068_safe_personalized_mixing import initial_moment_state
from experiments.dependence_delay_linear.t070_nonstationary_graph import (
    propagate_graph_block,
    recipient_actions,
    registered_static_graphs,
    retarget_state,
    static_graph_components,
    static_graph_risks,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t070a_exact_nonstationary_graph_preregistration.json"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t070a_exact_nonstationary_graph_scan"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unit_schedule(config: dict[str, Any], family: str) -> np.ndarray:
    model = config["model"]
    blocks = model["blocks"]
    patterns = {key: np.asarray(value, dtype=float) for key, value in model["patterns"].items()}
    if family == "stationary":
        keys = ["A"] * blocks
    elif family == "single_switch":
        keys = ["A"] * 12 + ["B"] * 12
    elif family == "alternating":
        keys = ["A"] * 8 + ["B"] * 8 + ["A"] * 8
    else:
        raise ValueError(f"unknown schedule family: {family}")
    return np.stack([patterns[key] for key in keys])


def scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    grid = config["grid"]
    for family in grid["schedule_family"]:
        for target_scale in grid["target_scale"]:
            for initial in grid["initial_common_parameter"]:
                for noise in grid["noise_scale"]:
                    for spatial in grid["spatial_correlation"]:
                        for temporal in grid["temporal_correlation"]:
                            for delay in grid["delay"]:
                                cell_id = (
                                    f"F{family}-H{target_scale:g}-I{initial:g}-N{noise:g}-"
                                    f"S{spatial:g}-T{temporal:g}-D{delay}"
                                )
                                rows.append({
                                    "cell_id": cell_id,
                                    "schedule_family": family,
                                    "target_scale": float(target_scale),
                                    "initial_common_parameter": float(initial),
                                    "noise_scale": float(noise),
                                    "spatial_correlation": float(spatial),
                                    "temporal_correlation": float(temporal),
                                    "delay": int(delay),
                                })
    return rows


def validate(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    budget = config["budget"]
    action_count = len(recipient_actions(model["agents"], 0, config["actions"]["alpha"]))
    graphs = action_count ** model["agents"]
    cells = scenarios(config)
    if config["analysis"]["uses_sampled_outcome"] is not False:
        raise ValueError("sampled outcomes are forbidden")
    if model["probe_samples_are_learning_updates"] is not False:
        raise ValueError("probe transitions cannot be learning updates")
    if model["blocks"] * model["learning_steps_per_block"] != budget["environment_transitions"]:
        raise ValueError("environment budget mismatch")
    if len(model["decision_blocks"]) * (
        budget["probe_message_units_per_decision"] + budget["mixing_message_units_per_decision"]
    ) > budget["message_units"]:
        raise ValueError("dynamic policy exceeds message budget")
    expected = config["expected_workload"]
    observed = {
        "cells": len(cells),
        "nonstationary_cells": sum(row["schedule_family"] != "stationary" for row in cells),
        "actions_per_recipient": action_count,
        "static_graphs_per_cell": graphs,
        "evaluated_static_graph_risks": len(cells) * graphs,
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise ValueError(f"frozen workload mismatch for {key}")
    return {"experiment_id": config["experiment_id"], **observed}


def execute_dynamic(config: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    schedule = scenario["target_scale"] * unit_schedule(config, scenario["schedule_family"])
    initial = np.repeat(scenario["initial_common_parameter"], model["agents"])
    state = initial_moment_state(schedule[0], initial, scenario["delay"])
    previous_target = schedule[0]
    decisions = set(model["decision_blocks"])
    risk_path = []
    shadow_path = []
    selected = []
    checkpoint_safe = True
    learning_transitions = 0
    probe_transitions = 0
    messages = 0
    for block in range(model["blocks"]):
        target = schedule[block]
        if block > 0 and not np.array_equal(target, previous_target):
            state = retarget_state(state, previous_target, target)
        if block in decisions:
            local_steps = model["learning_steps_per_block"] - model["probe_steps_per_decision"]
            result = propagate_graph_block(
                state,
                targets=target,
                gain=model["gain"],
                curvature=model["curvature"],
                local_steps=local_steps,
                noise_scale=scenario["noise_scale"],
                spatial_correlation=scenario["spatial_correlation"],
                temporal_correlation=scenario["temporal_correlation"],
                alpha_grid=config["actions"]["alpha"],
                safe_oracle=True,
            )
            probe_transitions += model["probe_steps_per_decision"]
            messages += config["budget"]["probe_message_units_per_decision"]
            if np.any(result.action_indices > 0):
                messages += config["budget"]["mixing_message_units_per_decision"]
            selected.append(result.action_indices.tolist())
            checkpoint_safe = checkpoint_safe and bool(
                np.all(result.personalized_risk <= result.shadow_risk + 1e-12)
            )
        else:
            local_steps = model["learning_steps_per_block"]
            result = propagate_graph_block(
                state,
                targets=target,
                gain=model["gain"],
                curvature=model["curvature"],
                local_steps=local_steps,
                noise_scale=scenario["noise_scale"],
                spatial_correlation=scenario["spatial_correlation"],
                temporal_correlation=scenario["temporal_correlation"],
                alpha_grid=config["actions"]["alpha"],
                fixed_action_indices=np.zeros(model["agents"], dtype=int),
            )
        state = result.state
        learning_transitions += local_steps
        risk_path.append(float(np.mean(result.personalized_risk)))
        shadow_path.append(float(np.mean(result.shadow_risk)))
        previous_target = target
    return {
        "auc_risk": float(np.mean(risk_path)),
        "terminal_risk": risk_path[-1],
        "charged_shadow_auc": float(np.mean(shadow_path)),
        "checkpoint_safe": checkpoint_safe,
        "selected_actions": selected,
        "learning_transitions": learning_transitions,
        "probe_transitions": probe_transitions,
        "environment_used": learning_transitions + probe_transitions,
        "message_used": messages,
    }


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def execute(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    model = config["model"]
    action_count = len(recipient_actions(model["agents"], 0, config["actions"]["alpha"]))
    graphs = registered_static_graphs(action_count, model["agents"])
    components = {}
    for family in config["grid"]["schedule_family"]:
        for delay in config["grid"]["delay"]:
            components[(family, delay)] = static_graph_components(
                graphs=graphs,
                agents=model["agents"],
                delay=delay,
                blocks=model["blocks"],
                decision_blocks=model["decision_blocks"],
                gain=model["gain"],
                curvature=model["curvature"],
                local_steps=model["learning_steps_per_block"],
                alpha_grid=config["actions"]["alpha"],
                unit_target_schedule=unit_schedule(config, family),
            )
    cells = []
    selected_graphs = set()
    for scenario in scenarios(config):
        static_auc, static_terminal = static_graph_risks(
            components[(scenario["schedule_family"], scenario["delay"])],
            initial_parameter=scenario["initial_common_parameter"],
            target_scale=scenario["target_scale"],
            gain=model["gain"],
            curvature=model["curvature"],
            local_steps=model["learning_steps_per_block"],
            noise_scale=scenario["noise_scale"],
            spatial_correlation=scenario["spatial_correlation"],
            temporal_correlation=scenario["temporal_correlation"],
        )
        best_index = int(np.argmin(static_auc))
        graph = graphs[best_index]
        selected_graphs.add(tuple(int(value) for value in graph))
        dynamic = execute_dynamic(config, scenario)
        decisions = np.asarray(dynamic["selected_actions"], dtype=int)
        action_changes = int(np.sum(np.any(decisions[1:] != decisions[:-1], axis=1)))
        cells.append({
            **scenario,
            "best_static_graph": json.dumps(graph.tolist(), separators=(",", ":")),
            "best_static_auc": float(static_auc[best_index]),
            "best_static_terminal": float(static_terminal[best_index]),
            "dynamic_auc": dynamic["auc_risk"],
            "dynamic_terminal": dynamic["terminal_risk"],
            "dynamic_to_static_auc_ratio": dynamic["auc_risk"] / float(static_auc[best_index]),
            "dynamic_to_static_terminal_ratio": dynamic["terminal_risk"] / float(static_terminal[best_index]),
            "charged_shadow_auc": dynamic["charged_shadow_auc"],
            "checkpoint_safe": dynamic["checkpoint_safe"],
            "selected_actions": json.dumps(dynamic["selected_actions"], separators=(",", ":")),
            "action_change_count": action_changes,
            "environment_used": dynamic["environment_used"],
            "message_used": dynamic["message_used"],
        })
    nonstationary = [row for row in cells if row["schedule_family"] != "stationary"]
    stationary = [row for row in cells if row["schedule_family"] == "stationary"]
    primary_ratio = geometric_mean([row["dynamic_to_static_auc_ratio"] for row in nonstationary])
    stationary_ratio = geometric_mean([row["dynamic_to_static_auc_ratio"] for row in stationary])
    family_metrics = {}
    for family in config["grid"]["schedule_family"]:
        subset = [row for row in cells if row["schedule_family"] == family]
        ratio = geometric_mean([row["dynamic_to_static_auc_ratio"] for row in subset])
        family_metrics[family] = {"ratio": ratio, "improvement": 1.0 - ratio}
    delay_metrics = {}
    for delay in config["grid"]["delay"]:
        subset = [row for row in nonstationary if row["delay"] == delay]
        ratio = geometric_mean([row["dynamic_to_static_auc_ratio"] for row in subset])
        delay_metrics[str(delay)] = {"ratio": ratio, "improvement": 1.0 - ratio}
    metrics = {
        "nonstationary_dynamic_to_static_auc_geometric_ratio": primary_ratio,
        "nonstationary_dynamic_improvement": 1.0 - primary_ratio,
        "nonstationary_strict_cell_fraction": float(np.mean([
            row["dynamic_auc"] < row["best_static_auc"] - 1e-15 for row in nonstationary
        ])),
        "stationary_dynamic_to_static_auc_geometric_ratio": stationary_ratio,
        "stationary_dynamic_improvement": 1.0 - stationary_ratio,
        "nonstationary_action_change_fraction": float(np.mean([
            row["action_change_count"] > 0 for row in nonstationary
        ])),
        "checkpoint_safe_fraction": float(np.mean([row["checkpoint_safe"] for row in cells])),
        "selected_static_graph_count": len(selected_graphs),
        "schedule_family": family_metrics,
        "delay": delay_metrics,
    }
    budget = config["budget"]
    finite = all(
        math.isfinite(row["dynamic_auc"])
        and row["dynamic_auc"] > 0.0
        and math.isfinite(row["best_static_auc"])
        and row["best_static_auc"] > 0.0
        and row["environment_used"] <= budget["environment_transitions"]
        and row["message_used"] <= budget["message_units"]
        for row in cells
    )
    gates = {
        "R1": finite and len(cells) == config["expected_workload"]["cells"],
        "R2": metrics["checkpoint_safe_fraction"] == 1.0,
        "R3": all(row["environment_used"] == budget["environment_transitions"] for row in cells),
        "R4": all(row["message_used"] <= budget["message_units"] for row in cells),
        "R5": metrics["nonstationary_dynamic_improvement"] >= 0.05,
        "R6": metrics["nonstationary_strict_cell_fraction"] >= 0.60,
        "R7": all(family_metrics[name]["improvement"] >= 0.03 for name in ("single_switch", "alternating")),
        "R8": all(value["improvement"] >= 0.0 for value in delay_metrics.values()),
        "R9": metrics["nonstationary_action_change_fraction"] >= 0.60,
        "R10": metrics["stationary_dynamic_improvement"] <= 0.05,
        "R11": len(selected_graphs) >= 8,
        "R12": False,
    }
    summary = {"experiment_id": config["experiment_id"], "metrics": metrics, "gates": gates}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    result = validate(config) if args.command in {"validate", "estimate"} else execute(config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
