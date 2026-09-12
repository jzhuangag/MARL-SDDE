"""Execute the preregistered T-068A exact personalized-mixing phase scan."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.t068_safe_personalized_mixing import (
    initial_moment_state,
    propagate_personalized_block,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t068a_exact_safe_mixing_preregistration.json"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t068a_exact_safe_mixing_scan"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    grid = config["grid"]
    for heterogeneity in grid["target_heterogeneity"]:
        for initial in grid["initial_common_parameter"]:
            for noise in grid["noise_scale"]:
                for spatial in grid["spatial_correlation"]:
                    for temporal in grid["temporal_correlation"]:
                        for delay in grid["delay"]:
                            cell_id = (
                                f"H{heterogeneity:g}-I{initial:g}-N{noise:g}-"
                                f"S{spatial:g}-T{temporal:g}-D{delay}"
                            )
                            rows.append(
                                {
                                    "cell_id": cell_id,
                                    "target_heterogeneity": float(heterogeneity),
                                    "initial_common_parameter": float(initial),
                                    "noise_scale": float(noise),
                                    "spatial_correlation": float(spatial),
                                    "temporal_correlation": float(temporal),
                                    "delay": int(delay),
                                }
                            )
    return rows


def validate(config: dict[str, Any]) -> dict[str, Any]:
    expected = config["expected_workload"]
    cells = scenario_rows(config)
    alpha = config["actions"]["alpha"]
    policies = len(alpha) + len(alpha) ** 2 + 1
    model = config["model"]
    budget = config["budget"]
    if config["analysis"]["uses_sampled_outcome"] is not False:
        raise ValueError("sampled outcomes are forbidden")
    if model["probe_samples_are_learning_updates"] is not False:
        raise ValueError("probe transitions cannot be learning updates")
    if model["learning_steps_per_block"] * model["blocks"] != budget["environment_transitions"]:
        raise ValueError("environment budget does not match frozen block design")
    if len(cells) != expected["cells"] or policies != 43:
        raise ValueError("frozen workload mismatch")
    if len(cells) * policies != expected["policy_rows"]:
        raise ValueError("policy-row count mismatch")
    if sorted(model["decision_blocks"]) != model["decision_blocks"]:
        raise ValueError("decision blocks must be sorted")
    if any(block < 0 or block >= model["blocks"] for block in model["decision_blocks"]):
        raise ValueError("decision block lies outside the horizon")
    safe_messages = len(model["decision_blocks"]) * (
        budget["safe_probe_message_units_per_decision"]
        + budget["safe_mixing_message_units_per_decision"]
    )
    if safe_messages > budget["message_units"]:
        raise ValueError("safe policy exceeds the message budget")
    return {
        "experiment_id": config["experiment_id"],
        "cells": len(cells),
        "policies_per_cell": policies,
        "policy_rows": len(cells) * policies,
        "safe_message_units": safe_messages,
        "maximum_scalar_blocks": len(cells) * policies * model["blocks"],
    }


def _targets(config: dict[str, Any], scenario: dict[str, Any]) -> np.ndarray:
    return scenario["target_heterogeneity"] * np.asarray(
        config["model"]["target_pattern"], dtype=float
    )


def execute_policy(
    config: dict[str, Any],
    scenario: dict[str, Any],
    *,
    policy: str,
    early_alpha: float = 0.0,
    late_alpha: float = 0.0,
) -> dict[str, Any]:
    model = config["model"]
    budget = config["budget"]
    targets = _targets(config, scenario)
    initial = np.repeat(scenario["initial_common_parameter"], model["agents"])
    state = initial_moment_state(targets, initial, scenario["delay"])
    decisions = {block: index for index, block in enumerate(model["decision_blocks"])}
    risk_path = []
    shadow_path = []
    selected = []
    fallbacks = []
    learning_transitions = 0
    probe_transitions = 0
    message_units = 0
    checkpoint_safe = True
    for block in range(model["blocks"]):
        decision_index = decisions.get(block)
        if policy == "safe_oracle" and decision_index is not None:
            learning_steps = model["learning_steps_per_block"] - model["probe_steps_per_decision"]
            result = propagate_personalized_block(
                state,
                targets=targets,
                gain=model["gain"],
                curvature=model["curvature"],
                local_steps=learning_steps,
                noise_scale=scenario["noise_scale"],
                spatial_correlation=scenario["spatial_correlation"],
                temporal_correlation=scenario["temporal_correlation"],
                safe_alpha_grid=config["actions"]["alpha"],
            )
            probe_transitions += model["probe_steps_per_decision"]
            message_units += budget["safe_probe_message_units_per_decision"]
            if np.any(result.selected_alpha > 0.0):
                message_units += budget["safe_mixing_message_units_per_decision"]
            checkpoint_safe = checkpoint_safe and bool(
                np.all(result.personalized_risk <= result.shadow_risk + 1e-12)
            )
        else:
            learning_steps = model["learning_steps_per_block"]
            value = 0.0
            if decision_index is not None:
                if policy == "fixed":
                    value = early_alpha
                elif policy == "two_phase":
                    value = (
                        early_alpha
                        if decision_index < config["actions"]["two_phase_switch_decision"]
                        else late_alpha
                    )
                else:
                    raise ValueError(f"unknown policy: {policy}")
                if value > 0.0:
                    message_units += budget["fixed_mixing_message_units_per_decision"]
            result = propagate_personalized_block(
                state,
                targets=targets,
                gain=model["gain"],
                curvature=model["curvature"],
                local_steps=learning_steps,
                noise_scale=scenario["noise_scale"],
                spatial_correlation=scenario["spatial_correlation"],
                temporal_correlation=scenario["temporal_correlation"],
                alpha=value,
            )
        state = result.state
        learning_transitions += learning_steps
        risk_path.append(result.personalized_risk)
        shadow_path.append(result.shadow_risk)
        if decision_index is not None:
            selected.append(result.selected_alpha)
            fallbacks.append(result.used_shadow)
    environment_used = learning_transitions + probe_transitions
    risk_array = np.asarray(risk_path)
    shadow_array = np.asarray(shadow_path)
    initial_risk = max(float(np.mean(np.square(initial - targets))), 1e-12)
    return {
        **scenario,
        "policy": policy,
        "early_alpha": early_alpha,
        "late_alpha": late_alpha,
        "terminal_risk": float(np.mean(risk_array[-1])),
        "worst_agent_terminal_risk": float(np.max(risk_array[-1])),
        "terminal_charged_shadow_risk": float(np.mean(shadow_array[-1])),
        "normalized_auc": float(np.mean(risk_array) / initial_risk),
        "learning_transitions": learning_transitions,
        "probe_transitions": probe_transitions,
        "environment_used": environment_used,
        "environment_budget": budget["environment_transitions"],
        "message_used": message_units,
        "message_budget": budget["message_units"],
        "checkpoint_safe": checkpoint_safe,
        "selected_alpha": np.asarray(selected).tolist(),
        "used_shadow": np.asarray(fallbacks).tolist(),
    }


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def analyze(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = config["expected_workload"]
    if len(rows) != expected["policy_rows"]:
        raise ValueError("policy-row count mismatch")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["cell_id"], []).append(row)
    cells = []
    all_alpha = set()
    all_fallback = set()
    for cell_id, policies in sorted(grouped.items()):
        first = policies[0]
        fixed = [row for row in policies if row["policy"] == "fixed"]
        two_phase = [row for row in policies if row["policy"] == "two_phase"]
        safe = next(row for row in policies if row["policy"] == "safe_oracle")
        local = next(row for row in fixed if row["early_alpha"] == 0.0)
        strong_fixed = min(
            fixed, key=lambda row: (row["terminal_risk"], row["early_alpha"])
        )
        phase = min(
            two_phase,
            key=lambda row: (row["terminal_risk"], row["early_alpha"], row["late_alpha"]),
        )
        selected = np.asarray(safe["selected_alpha"], dtype=float)
        fallback = np.asarray(safe["used_shadow"], dtype=bool)
        all_alpha.update(selected.ravel().tolist())
        all_fallback.update(fallback.ravel().tolist())
        split = config["actions"]["two_phase_switch_decision"]
        early_mean = float(np.mean(selected[:split]))
        late_mean = float(np.mean(selected[split:]))
        cells.append(
            {
                "cell_id": cell_id,
                **{
                    key: first[key]
                    for key in (
                        "target_heterogeneity",
                        "initial_common_parameter",
                        "noise_scale",
                        "spatial_correlation",
                        "temporal_correlation",
                        "delay",
                    )
                },
                "local_risk": local["terminal_risk"],
                "strong_fixed_alpha": strong_fixed["early_alpha"],
                "strong_fixed_risk": strong_fixed["terminal_risk"],
                "phase_early_alpha": phase["early_alpha"],
                "phase_late_alpha": phase["late_alpha"],
                "phase_risk": phase["terminal_risk"],
                "safe_risk": safe["terminal_risk"],
                "safe_charged_shadow_risk": safe["terminal_charged_shadow_risk"],
                "safe_to_local_ratio": safe["terminal_risk"] / local["terminal_risk"],
                "safe_to_fixed_ratio": safe["terminal_risk"] / strong_fixed["terminal_risk"],
                "phase_to_fixed_ratio": phase["terminal_risk"] / strong_fixed["terminal_risk"],
                "safe_checkpoint_valid": safe["checkpoint_safe"],
                "safe_early_alpha_mean": early_mean,
                "safe_late_alpha_mean": late_mean,
                "safe_uses_nonzero": bool(np.any(selected > 0.0)),
                "safe_uses_shadow": bool(np.any(fallback)),
                "safe_environment_used": safe["environment_used"],
                "safe_message_used": safe["message_used"],
            }
        )
    phase_ratio = geometric_mean([row["phase_to_fixed_ratio"] for row in cells])
    safe_fixed_ratio = geometric_mean([row["safe_to_fixed_ratio"] for row in cells])
    safe_local_ratio = geometric_mean([row["safe_to_local_ratio"] for row in cells])
    active = [row for row in cells if row["safe_uses_nonzero"]]
    metrics = {
        "phase_to_fixed_geometric_ratio": phase_ratio,
        "phase_aggregate_improvement": 1.0 - phase_ratio,
        "phase_strict_cell_fraction": float(
            np.mean([row["phase_risk"] < row["strong_fixed_risk"] - 1e-15 for row in cells])
        ),
        "safe_to_fixed_geometric_ratio": safe_fixed_ratio,
        "safe_aggregate_improvement_over_fixed": 1.0 - safe_fixed_ratio,
        "safe_strict_fixed_cell_fraction": float(
            np.mean([row["safe_risk"] < row["strong_fixed_risk"] - 1e-15 for row in cells])
        ),
        "safe_to_local_geometric_ratio": safe_local_ratio,
        "safe_local_ratio_at_most_1_05_fraction": float(
            np.mean([row["safe_to_local_ratio"] <= 1.05 for row in cells])
        ),
        "safe_checkpoint_fraction": float(np.mean([row["safe_checkpoint_valid"] for row in cells])),
        "safe_nonzero_cell_fraction": float(np.mean([row["safe_uses_nonzero"] for row in cells])),
        "collaborate_then_personalize_fraction_among_active": float(
            np.mean(
                [row["safe_early_alpha_mean"] > row["safe_late_alpha_mean"] + 1e-12 for row in active]
            )
        ) if active else 0.0,
        "selected_alpha_values": sorted(all_alpha),
        "selected_shadow_states": sorted(all_fallback),
    }
    finite_valid = all(
        math.isfinite(row["terminal_risk"])
        and row["terminal_risk"] > 0.0
        and row["environment_used"] <= row["environment_budget"]
        and row["message_used"] <= row["message_budget"]
        for row in rows
    )
    gates = {
        "P1": finite_valid and len(cells) == 648 and len(rows) == 27864,
        "P2": all(
            abs(row["terminal_risk"] - row["terminal_charged_shadow_risk"]) <= 1e-12
            for row in rows
            if row["policy"] == "fixed" and row["early_alpha"] == 0.0
        ),
        "P3": metrics["safe_checkpoint_fraction"] == 1.0,
        "P4": finite_valid,
        "P5": metrics["phase_aggregate_improvement"] >= 0.05,
        "P6": metrics["phase_strict_cell_fraction"] >= 0.50,
        "P7": metrics["safe_aggregate_improvement_over_fixed"] >= 0.03,
        "P8": metrics["safe_strict_fixed_cell_fraction"] >= 0.40,
        "P9": safe_local_ratio <= 1.0 and metrics["safe_local_ratio_at_most_1_05_fraction"] >= 0.95,
        "P10": metrics["collaborate_then_personalize_fraction_among_active"] >= 0.40,
        "P11": len(all_alpha) >= 4 and all_fallback == {False, True},
        "P12": False,
    }
    return {
        "experiment_id": config["experiment_id"],
        "metrics": metrics,
        "gates": gates,
        "cells": cells,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in rows:
        serializable.append(
            {
                key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)


def execute(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    rows = []
    alpha = config["actions"]["alpha"]
    for scenario in scenario_rows(config):
        rows.extend(
            execute_policy(config, scenario, policy="fixed", early_alpha=value)
            for value in alpha
        )
        rows.extend(
            execute_policy(
                config,
                scenario,
                policy="two_phase",
                early_alpha=early,
                late_alpha=late,
            )
            for early in alpha
            for late in alpha
        )
        rows.append(execute_policy(config, scenario, policy="safe_oracle"))
    summary = analyze(config, rows)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "policies.csv", rows)
    write_csv(output / "cells.csv", summary["cells"])
    compact = {key: value for key, value in summary.items() if key != "cells"}
    (output / "summary.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
    else:
        result = execute(config, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "cells"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
