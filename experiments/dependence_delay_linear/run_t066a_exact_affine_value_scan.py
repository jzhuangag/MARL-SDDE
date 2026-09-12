"""Execute the preregistered T-066A exact delayed-affine value scan."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.t065_discrete_joint_certificate import (
    audit_certificate,
    common_quadratic_certificate,
)
from experiments.dependence_delay_linear.t066_exact_delayed_affine_risk import (
    ScheduledAction,
    exact_learning_horizon,
    propagate_schedule,
    terminal_mean_square_risk,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t066a_exact_affine_value_scan_preregistration.json"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t066a_exact_affine_value_scan"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sensor_costs(config: dict[str, Any], overhead: int) -> tuple[int, int]:
    sensor = config["sensor"]
    residual_environment = (
        2 * sensor["paired_replicates"] * sensor["residual_block_length"] * sensor["q_residual"]
    )
    fingerprint_environment = (
        sensor["fingerprint_blocks"] * sensor["fingerprint_length"] * sensor["q_fingerprint"]
    )
    residual_messages = 2 * sensor["paired_replicates"] * (overhead + sensor["q_residual"])
    fingerprint_messages = sensor["fingerprint_blocks"] * (overhead + sensor["q_fingerprint"])
    return int(residual_messages + fingerprint_messages), int(
        residual_environment + fingerprint_environment
    )


def scenario_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in config["tasks"]:
        for budget in config["budget_regimes"]:
            for delay in config["grid"]["delays"]:
                for rho in config["grid"]["rho"]:
                    for noise in config["grid"]["noise_scale"]:
                        for energy in config["grid"]["initial_error_energy"]:
                            rows.append(
                                {
                                    "cell_id": f"{task}-{budget}-D{delay}-R{rho:g}-N{noise:g}-E{energy:g}",
                                    "task": task,
                                    "budget_regime": budget,
                                    "delay": int(delay),
                                    "rho": float(rho),
                                    "noise_scale": float(noise),
                                    "initial_error_energy": float(energy),
                                }
                            )
    return rows


def certificates(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    eta_min = min(config["actions"]["gain"])
    eta_max = max(config["actions"]["gain"])
    output = {}
    for task, specification in config["tasks"].items():
        drift = np.asarray(specification["drift"], dtype=float)
        for delay in config["grid"]["delays"]:
            certificate = common_quadratic_certificate(
                drift, eta_min=eta_min, eta_max=eta_max, delay=int(delay)
            )
            audit = audit_certificate(drift, certificate, grid_size=101)
            output[f"{task}-D{delay}"] = {
                "margin": certificate.margin,
                **audit,
            }
    return output


def validate(config: dict[str, Any]) -> dict[str, Any]:
    scenarios = scenario_rows(config)
    expected = config["expected_workload"]
    actions = len(config["actions"]["participation"]) * len(config["actions"]["gain"])
    if config["analysis"]["uses_sampled_outcome"] is not False:
        raise ValueError("sampled outcome is forbidden in the analytic scan")
    if config["sensor"]["probe_samples_are_learning_updates"] is not False:
        raise ValueError("probe samples cannot be credited as learning updates")
    if len(scenarios) != expected["cells"] or actions != expected["actions_per_cell"]:
        raise ValueError("frozen workload mismatch")
    if len(scenarios) * actions != expected["action_rows"]:
        raise ValueError("action-row count mismatch")
    certificate_rows = certificates(config)
    if any(row["margin"] <= 0.0 or row["worst_drift_eigenvalue"] >= 0.0 for row in certificate_rows.values()):
        raise ValueError("common Lyapunov certificate gate failed")
    return {
        "experiment_id": config["experiment_id"],
        "cells": len(scenarios),
        "actions_per_cell": actions,
        "action_rows": len(scenarios) * actions,
        "certificates": certificate_rows,
    }


def evaluate_action(
    config: dict[str, Any], scenario: dict[str, Any], participation: int, gain: float
) -> dict[str, Any]:
    task = config["tasks"][scenario["task"]]
    budget = config["budget_regimes"][scenario["budget_regime"]]
    sensor_messages, sensor_environment = sensor_costs(config, budget["message_overhead"])
    updates = exact_learning_horizon(
        participation=participation,
        message_overhead=budget["message_overhead"],
        message_budget=budget["message_budget"],
        environment_budget=budget["environment_budget"],
        sensor_message_cost=sensor_messages,
        sensor_environment_cost=sensor_environment,
        reserved_delay_updates=scenario["delay"],
    )
    direction = np.asarray(task["initial_direction"], dtype=float)
    direction /= np.linalg.norm(direction)
    initial_error = math.sqrt(scenario["initial_error_energy"]) * direction
    state = propagate_schedule(
        drift=np.asarray(task["drift"], dtype=float),
        base_noise_covariance=scenario["noise_scale"] * np.eye(direction.size),
        initial_error=initial_error,
        delay=scenario["delay"],
        rho=scenario["rho"],
        schedule=[ScheduledAction(participation, gain, updates)],
    )
    message_used = sensor_messages + updates * (budget["message_overhead"] + participation)
    environment_used = sensor_environment + updates * participation
    return {
        **scenario,
        "participation": participation,
        "gain": gain,
        "updates": updates,
        "risk": terminal_mean_square_risk(state, direction.size),
        "sensor_message_cost": sensor_messages,
        "sensor_environment_cost": sensor_environment,
        "message_used": message_used,
        "message_budget": budget["message_budget"],
        "environment_used": environment_used,
        "environment_budget": budget["environment_budget"],
    }


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def analyze(config: dict[str, Any], action_rows: list[dict[str, Any]], certificate_rows: dict[str, Any]) -> dict[str, Any]:
    expected = config["expected_workload"]
    if len(action_rows) != expected["action_rows"]:
        raise ValueError("action-row count mismatch")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in action_rows:
        grouped.setdefault(row["cell_id"], []).append(row)
    if len(grouped) != expected["cells"]:
        raise ValueError("cell count mismatch")
    fixed_pairs = {}
    for task in config["tasks"]:
        for budget in config["budget_regimes"]:
            relevant = [rows for rows in grouped.values() if rows[0]["task"] == task and rows[0]["budget_regime"] == budget]
            candidates = []
            for q in config["actions"]["participation"]:
                for eta in config["actions"]["gain"]:
                    risks = [next(row["risk"] for row in rows if row["participation"] == q and row["gain"] == eta) for rows in relevant]
                    candidates.append((geometric_mean(risks), q, eta))
            fixed_pairs[f"{task}-{budget}"] = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    cells = []
    for cell_id, rows in sorted(grouped.items()):
        first = rows[0]
        fixed_risk, fixed_q, fixed_eta = fixed_pairs[f"{first['task']}-{first['budget_regime']}"]
        del fixed_risk
        fixed = next(row for row in rows if row["participation"] == fixed_q and row["gain"] == fixed_eta)
        oracle = min(rows, key=lambda row: (row["risk"], row["participation"], row["gain"]))
        q_only = min(
            (row for row in rows if row["gain"] == fixed_eta),
            key=lambda row: (row["risk"], row["participation"]),
        )
        eta_only = min(
            (row for row in rows if row["participation"] == fixed_q),
            key=lambda row: (row["risk"], row["gain"]),
        )
        cells.append(
            {
                "cell_id": cell_id,
                **{key: first[key] for key in ("task", "budget_regime", "delay", "rho", "noise_scale", "initial_error_energy")},
                "fixed_q": fixed_q,
                "fixed_eta": fixed_eta,
                "fixed_risk": fixed["risk"],
                "oracle_q": oracle["participation"],
                "oracle_eta": oracle["gain"],
                "oracle_updates": oracle["updates"],
                "oracle_risk": oracle["risk"],
                "oracle_to_fixed_ratio": oracle["risk"] / fixed["risk"],
                "q_only_risk": q_only["risk"],
                "eta_only_risk": eta_only["risk"],
            }
        )
    ratio = geometric_mean([row["oracle_to_fixed_ratio"] for row in cells])
    strict_fraction = float(np.mean([row["oracle_risk"] < row["fixed_risk"] - 1e-15 for row in cells]))
    joint_fraction = float(np.mean([
        row["oracle_risk"] < min(row["q_only_risk"], row["eta_only_risk"]) - 1e-15 for row in cells
    ]))
    diversity = {}
    for budget in config["budget_regimes"]:
        subset = [row for row in cells if row["budget_regime"] == budget]
        diversity[budget] = {
            "distinct_oracle_q": sorted({row["oracle_q"] for row in subset}),
            "distinct_oracle_eta": sorted({row["oracle_eta"] for row in subset}),
        }
    finite_budget = all(
        math.isfinite(row["risk"])
        and row["risk"] > 0.0
        and row["message_used"] <= row["message_budget"]
        and row["environment_used"] <= row["environment_budget"]
        for row in action_rows
    )
    metrics = {
        "oracle_to_strong_fixed_geometric_ratio": ratio,
        "oracle_aggregate_improvement": 1.0 - ratio,
        "strictly_improved_cell_fraction": strict_fraction,
        "joint_beats_both_one_dimensional_fraction": joint_fraction,
        "action_and_budget_valid": finite_budget,
        "oracle_action_diversity": diversity,
    }
    gates = {
        "S1": all(row["margin"] > 0.0 for row in certificate_rows.values()),
        "S2": finite_budget and len(cells) == 648 and len(action_rows) == 27216,
        "S3": metrics["oracle_aggregate_improvement"] >= 0.10,
        "S4": strict_fraction >= 0.60,
        "S5": joint_fraction >= 0.30,
        "S6": all(len(value["distinct_oracle_q"]) >= 2 and len(value["distinct_oracle_eta"]) >= 2 for value in diversity.values()),
        "S7": True,
        "S8": False,
    }
    return {
        "experiment_id": config["experiment_id"],
        "metrics": metrics,
        "gates": gates,
        "fixed_pairs": {
            key: {"geometric_risk": value[0], "q": value[1], "eta": value[2]}
            for key, value in fixed_pairs.items()
        },
        "certificates": certificate_rows,
        "cells": cells,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validated = validate(config)
    action_rows = [
        evaluate_action(config, scenario, q, eta)
        for scenario in scenario_rows(config)
        for q in config["actions"]["participation"]
        for eta in config["actions"]["gain"]
    ]
    summary = analyze(config, action_rows, validated["certificates"])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "actions.csv", action_rows)
    write_csv(output / "cells.csv", summary["cells"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command in {"validate", "estimate"}:
        result = validate(config)
        if args.command == "estimate":
            result["maximum_scalar_recursion_updates"] = 2 * result["action_rows"] * 1200
    else:
        result = execute(config, args.output)
    compact = {key: value for key, value in result.items() if key != "cells"}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
