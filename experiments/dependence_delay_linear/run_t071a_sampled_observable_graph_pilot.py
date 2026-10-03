"""Run the preregistered T-071A sampled observable graph CPU pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.run_t070a_exact_nonstationary_graph_scan import (
    unit_schedule,
)
from experiments.dependence_delay_linear.t071_sampled_graph_controller import (
    action_change_statistics,
    sample_markov_observations,
    simulate_policy,
    stable_seed,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t071a_sampled_observable_graph_preregistration.json"
T070_CONFIG = ROOT / "docs/t070a_exact_nonstationary_graph_preregistration.json"
T070_CELLS = ROOT / "experiments/dependence_delay_linear/results/t070a_exact_nonstationary_graph_scan/cells.csv"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t071a_sampled_observable_graph_pilot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_t070_config() -> dict[str, Any]:
    return json.loads(T070_CONFIG.read_text(encoding="utf-8"))


def load_static_graphs() -> dict[str, np.ndarray]:
    with T070_CELLS.open(newline="", encoding="utf-8") as handle:
        return {
            row["cell_id"]: np.asarray(json.loads(row["best_static_graph"]), dtype=int)
            for row in csv.DictReader(handle)
        }


def scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["grid"]
    rows = []
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


def shift_decision_indices(family: str) -> list[int]:
    if family == "stationary":
        return []
    if family == "single_switch":
        return [3]
    if family == "alternating":
        return [2, 4]
    raise ValueError(f"unknown schedule family: {family}")


def validate(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source_experiment"]
    if sha256(T070_CELLS) != source["cells_sha256"]:
        raise ValueError("T-070A cells hash mismatch")
    parent = load_t070_config()
    structural_keys = (
        "agents",
        "patterns",
        "curvature",
        "gain",
        "blocks",
        "learning_steps_per_block",
        "decision_blocks",
    )
    if any(config["model"][key] != parent["model"][key] for key in structural_keys):
        raise ValueError("T-071A structural model must match T-070A")
    if config["model"]["probe_steps_per_decision"] != config["probe"]["steps_per_decision"]:
        raise ValueError("model and split-probe charges disagree")
    if config["grid"] != parent["grid"]:
        raise ValueError("T-071A grid must match T-070A exactly")
    if config["actions"] != parent["actions"]:
        raise ValueError("T-071A action catalogue must match T-070A exactly")
    if len(config["pilot_seeds"]) != len(set(config["pilot_seeds"])):
        raise ValueError("pilot seeds must be unique")
    if config["analysis"]["uses_t070_outcomes_for_controller"] is not False:
        raise ValueError("T-070 outcomes cannot enter the controller")
    if config["probe"]["samples_are_learning_updates"] is not False:
        raise ValueError("probe samples cannot be learning updates")
    rows = scenarios(config)
    static = load_static_graphs()
    if set(static) != {row["cell_id"] for row in rows}:
        raise ValueError("static comparator cell identity mismatch")
    expected = config["expected_workload"]
    observed = {
        "cells": len(rows),
        "nonstationary_cells": sum(row["schedule_family"] != "stationary" for row in rows),
        "pilot_seeds": len(config["pilot_seeds"]),
        "endpoints": len(rows) * len(config["pilot_seeds"]),
        "policy_trajectories": len(rows) * len(config["pilot_seeds"]) * len(config["comparators"]),
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise ValueError(f"frozen workload mismatch for {key}")
    return {"experiment_id": config["experiment_id"], **observed}


def run_endpoint(
    config: dict[str, Any],
    parent: dict[str, Any],
    scenario: dict[str, Any],
    static_graph: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    model = config["model"]
    targets = scenario["target_scale"] * unit_schedule(parent, scenario["schedule_family"])
    observations, noise = sample_markov_observations(
        targets=targets,
        steps_per_block=model["learning_steps_per_block"],
        noise_scale=scenario["noise_scale"],
        spatial_correlation=scenario["spatial_correlation"],
        temporal_correlation=scenario["temporal_correlation"],
        seed=stable_seed(seed, scenario["cell_id"], "observations"),
    )
    common = {
        "observations": observations,
        "targets": targets,
        "initial_parameter": scenario["initial_common_parameter"],
        "gain": model["gain"],
        "delay": scenario["delay"],
        "decision_blocks": model["decision_blocks"],
        "probe_steps": config["probe"]["steps_per_decision"],
        "selection_steps": config["probe"]["selection_steps"],
        "alpha_grid": config["actions"]["alpha"],
        "probe_message_units": config["budget"]["probe_message_units_per_decision"],
        "mixing_message_units": config["budget"]["mixing_message_units_per_decision"],
    }
    local = simulate_policy(**common, policy="local_no_probe")
    shadow = simulate_policy(**common, policy="charged_shadow")
    static = simulate_policy(**common, policy="static_graph", static_graph=static_graph)
    observable = simulate_policy(**common, policy="observable")
    clairvoyant = simulate_policy(**common, policy="clairvoyant")
    full = simulate_policy(**common, policy="full_sharing")
    change = action_change_statistics(
        observable.selected_actions,
        shift_decision_indices(scenario["schedule_family"]),
    )
    return {
        **scenario,
        "seed": int(seed),
        "noise_mean": float(np.mean(noise)),
        "noise_second_moment": float(np.mean(np.square(noise))),
        "local_auc": local.auc_risk,
        "charged_shadow_auc": shadow.auc_risk,
        "strong_static_auc": static.auc_risk,
        "observable_auc": observable.auc_risk,
        "clairvoyant_auc": clairvoyant.auc_risk,
        "full_sharing_auc": full.auc_risk,
        "local_terminal": local.terminal_risk,
        "strong_static_terminal": static.terminal_risk,
        "observable_terminal": observable.terminal_risk,
        "observable_actions": json.dumps(observable.selected_actions.tolist(), separators=(",", ":")),
        "observable_shadow_states": json.dumps(observable.used_shadow.tolist(), separators=(",", ":")),
        "observable_checkpoint_violations": observable.checkpoint_violations,
        "observable_learning_transitions": observable.learning_transitions,
        "observable_probe_transitions": observable.probe_transitions,
        "observable_environment_used": observable.learning_transitions + observable.probe_transitions,
        "observable_message_units": observable.message_units,
        "observable_candidate_scores": observable.candidate_scores,
        **change,
    }


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def cvar90(values: list[float]) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(math.ceil(0.1 * array.size)))
    return float(np.mean(array[-count:]))


def analyze(config: dict[str, Any], endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    expected = config["expected_workload"]
    if len(endpoints) != expected["endpoints"]:
        raise ValueError("endpoint count mismatch")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in endpoints:
        grouped.setdefault(row["cell_id"], []).append(row)
    if len(grouped) != expected["cells"] or any(
        len(rows) != expected["pilot_seeds"] for rows in grouped.values()
    ):
        raise ValueError("cell or seed coverage mismatch")
    cells = []
    for cell_id, rows in sorted(grouped.items()):
        first = rows[0]
        mean = {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "local_auc",
                "charged_shadow_auc",
                "strong_static_auc",
                "observable_auc",
                "clairvoyant_auc",
                "full_sharing_auc",
            )
        }
        cells.append({
            "cell_id": cell_id,
            **{key: first[key] for key in (
                "schedule_family",
                "target_scale",
                "initial_common_parameter",
                "noise_scale",
                "spatial_correlation",
                "temporal_correlation",
                "delay",
            )},
            **{f"mean_{key}": value for key, value in mean.items()},
            "observable_to_static_ratio": mean["observable_auc"] / mean["strong_static_auc"],
            "observable_to_shadow_ratio": mean["observable_auc"] / mean["charged_shadow_auc"],
            "clairvoyant_to_static_ratio": mean["clairvoyant_auc"] / mean["strong_static_auc"],
            "observable_cvar90": cvar90([row["observable_auc"] for row in rows]),
            "static_cvar90": cvar90([row["strong_static_auc"] for row in rows]),
            "shadow_cvar90": cvar90([row["charged_shadow_auc"] for row in rows]),
            "checkpoint_violation_rate": float(np.mean([
                row["observable_checkpoint_violations"] / (6 * 4) for row in rows
            ])),
            "shift_change_fraction": (
                sum(row["shift_changes"] for row in rows) / sum(row["shift_total"] for row in rows)
                if sum(row["shift_total"] for row in rows) else 0.0
            ),
        })
    nonstationary = [row for row in cells if row["schedule_family"] != "stationary"]
    stationary = [row for row in cells if row["schedule_family"] == "stationary"]
    primary_ratio = geometric_mean([row["observable_to_static_ratio"] for row in nonstationary])
    shadow_ratio = geometric_mean([row["observable_to_shadow_ratio"] for row in nonstationary])
    clairvoyant_ratio = geometric_mean([row["clairvoyant_to_static_ratio"] for row in nonstationary])
    benefit_capture = (
        (1.0 - primary_ratio) / (1.0 - clairvoyant_ratio)
        if clairvoyant_ratio < 1.0 else float("-inf")
    )
    schedule = {}
    for family in config["grid"]["schedule_family"]:
        subset = [row for row in cells if row["schedule_family"] == family]
        ratio = geometric_mean([row["observable_to_static_ratio"] for row in subset])
        schedule[family] = {"ratio": ratio, "improvement": 1.0 - ratio}
    delay = {}
    for value in config["grid"]["delay"]:
        subset = [row for row in nonstationary if row["delay"] == value]
        ratio = geometric_mean([row["observable_to_static_ratio"] for row in subset])
        delay[str(value)] = {"ratio": ratio, "improvement": 1.0 - ratio}
    cvar_ratio = geometric_mean([
        row["observable_cvar90"] / row["static_cvar90"] for row in nonstationary
    ])
    finite_budget = all(
        all(math.isfinite(row[key]) and row[key] >= 0.0 for key in (
            "local_auc",
            "charged_shadow_auc",
            "strong_static_auc",
            "observable_auc",
            "clairvoyant_auc",
            "full_sharing_auc",
        ))
        and row["observable_environment_used"] == config["budget"]["environment_transitions"]
        and row["observable_message_units"] <= config["budget"]["message_units"]
        and row["observable_candidate_scores"] == 6 * 4 * 7
        for row in endpoints
    )
    metrics = {
        "nonstationary_observable_to_static_geometric_ratio": primary_ratio,
        "nonstationary_observable_improvement": 1.0 - primary_ratio,
        "nonstationary_strict_cell_fraction": float(np.mean([
            row["mean_observable_auc"] < row["mean_strong_static_auc"] for row in nonstationary
        ])),
        "nonstationary_observable_to_shadow_geometric_ratio": shadow_ratio,
        "nonstationary_shadow_ratio_at_most_1_05_fraction": float(np.mean([
            row["observable_to_shadow_ratio"] <= 1.05 for row in nonstationary
        ])),
        "nonstationary_cvar90_to_static_geometric_ratio": cvar_ratio,
        "clairvoyant_to_static_geometric_ratio": clairvoyant_ratio,
        "clairvoyant_benefit_capture": benefit_capture,
        "nonstationary_shift_change_fraction": float(np.mean([
            row["shift_change_fraction"] for row in nonstationary
        ])),
        "checkpoint_violation_rate": float(np.mean([
            row["checkpoint_violation_rate"] for row in nonstationary
        ])),
        "stationary_observable_to_static_geometric_ratio": geometric_mean([
            row["observable_to_static_ratio"] for row in stationary
        ]),
        "schedule": schedule,
        "delay": delay,
    }
    gates = {
        "S1": finite_budget,
        "S2": len(cells) == 432 and len(endpoints) == 13_824,
        "S3": metrics["nonstationary_observable_improvement"] >= 0.05,
        "S4": metrics["nonstationary_strict_cell_fraction"] >= 0.55,
        "S5": all(schedule[name]["improvement"] >= 0.02 for name in ("single_switch", "alternating")),
        "S6": all(value["improvement"] >= 0.0 for value in delay.values()),
        "S7": shadow_ratio <= 1.0 and metrics["nonstationary_shadow_ratio_at_most_1_05_fraction"] >= 0.90,
        "S8": cvar_ratio <= 0.97,
        "S9": benefit_capture >= 0.35,
        "S10": metrics["nonstationary_shift_change_fraction"] >= 0.60,
        "S11": metrics["stationary_observable_to_static_geometric_ratio"] <= 1.10,
        "S12": False,
    }
    return {"experiment_id": config["experiment_id"], "metrics": metrics, "gates": gates, "cells": cells}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    parent = load_t070_config()
    graphs = load_static_graphs()
    started = time.perf_counter()
    endpoints = []
    for scenario in scenarios(config):
        for seed in config["pilot_seeds"]:
            endpoints.append(run_endpoint(
                config, parent, scenario, graphs[scenario["cell_id"]], int(seed)
            ))
    summary = analyze(config, endpoints)
    cells = summary.pop("cells")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cells)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**summary, "runtime_seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    result = validate(config) if args.command in {"validate", "estimate"} else run(config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
