"""T-020 read-only nonlinear adaptation-value feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ENDPOINT_SHA256 = "bc241c772d20b76c5f42f72bd8a5523bda2ba225e113811e695dd840007191f0"
SOURCE_ARTIFACT = "/scratch/jzhuangag/exp017a-pilot-17a4c32/endpoints.csv"
SCENARIO = ["task", "mixing", "rho", "delay_trace", "budget"]
FIXED_Q = {
    "single_agent": 1,
    "fixed_q4": 4,
    "fixed_q16": 16,
    "fixed_q32": 32,
}
EXPECTED_FALLBACK = {
    ("acrobot", "environment_binding"): 16,
    ("acrobot", "message_binding"): 1,
    ("cartpole", "environment_binding"): 32,
    ("cartpole", "message_binding"): 1,
}
PARAMETERS = {"cartpole": 4545, "acrobot": 4673}
SERVER_OVERHEAD_BYTES = 65_536
FLOAT_BYTES = 4
PROBE_Q = 4
PROBE_B = 1
BLOCK_SERVER_TICKS = 16
INITIAL_PROBE_BLOCKS = 8
PERIODIC_PROBE_INTERVAL = 32
ORACLE_AGGREGATE_GATE = 0.05
ORACLE_DIRECTIONAL_GATE = 0.60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0 or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError("positive finite values required")
    return float(np.exp(np.mean(np.log(array))))


def cvar90(values: Iterable[float]) -> float:
    array = np.sort(np.asarray(list(values), dtype=float))
    count = max(1, int(math.ceil(0.10 * len(array))))
    return float(np.mean(array[-count:]))


def probe_due(block: int) -> bool:
    return block < INITIAL_PROBE_BLOCKS or block % PERIODIC_PROBE_INTERVAL == 0


def probe_count(learning_updates: int) -> int:
    if learning_updates <= 0:
        return 0
    blocks = int(math.ceil(learning_updates / BLOCK_SERVER_TICKS))
    return sum(probe_due(block) for block in range(blocks))


def action_message_cost(q: int, parameters: int) -> int:
    return SERVER_OVERHEAD_BYTES + q * parameters * FLOAT_BYTES


def usable_learning_updates(row: pd.Series) -> dict[str, int | float]:
    """Deduct the complete T-019 probe schedule from both frozen budgets."""

    parameters = PARAMETERS[str(row["task"])]
    learning_cost = action_message_cost(int(row["q"]), parameters)
    probe_cost = action_message_cost(PROBE_Q, parameters)
    message_budget = int(row["message_budget"])
    environment_budget = int(row["environment_budget"])
    no_probe_updates = int(row["server_ticks"])
    usable = 0
    for candidate in range(no_probe_updates + 1):
        probes = probe_count(candidate)
        if (
            candidate * learning_cost + probes * probe_cost <= message_budget
            and candidate + probes * PROBE_B <= environment_budget
        ):
            usable = candidate
    probes = probe_count(usable)
    return {
        "usable_learning_updates": usable,
        "no_probe_learning_updates": no_probe_updates,
        "probe_count": probes,
        "probe_message_bytes": probes * probe_cost,
        "probe_environment_steps": probes * PROBE_B,
        "total_message_bytes": usable * learning_cost + probes * probe_cost,
        "total_environment_steps": usable + probes * PROBE_B,
    }


def aggregate_outcome_metrics(rows: pd.DataFrame) -> dict[str, float | int]:
    cellwise_cvar = rows.groupby(SCENARIO, sort=True)["terminal_prediction_mse"].apply(
        cvar90
    )
    return {
        "seed_cell_rows": int(len(rows)),
        "geometric_terminal_prediction_mse": geometric_mean(
            rows["terminal_prediction_mse"]
        ),
        "pooled_seed_cell_cvar90_terminal_prediction_mse": cvar90(
            rows["terminal_prediction_mse"]
        ),
        "mean_cellwise_two_seed_cvar90_terminal_prediction_mse": float(
            cellwise_cvar.mean()
        ),
        "geometric_cellwise_two_seed_cvar90_terminal_prediction_mse": geometric_mean(
            cellwise_cvar
        ),
        "mean_normalized_prediction_auc": float(
            rows["normalized_prediction_auc"].mean()
        ),
        "mean_no_probe_learning_updates": float(rows["server_ticks"].mean()),
    }


def append_probe_accounting(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    accounting = rows.apply(usable_learning_updates, axis=1, result_type="expand")
    enriched = pd.concat([rows.reset_index(drop=True), accounting], axis=1)
    return enriched, {
        "mean_usable_learning_updates": float(
            enriched["usable_learning_updates"].mean()
        ),
        "usable_over_no_probe_update_fraction": float(
            enriched["usable_learning_updates"].sum()
            / enriched["no_probe_learning_updates"].sum()
        ),
        "mean_probe_count": float(enriched["probe_count"].mean()),
        "mean_probe_message_bytes": float(enriched["probe_message_bytes"].mean()),
        "mean_probe_environment_steps": float(
            enriched["probe_environment_steps"].mean()
        ),
        "mean_total_message_bytes": float(enriched["total_message_bytes"].mean()),
        "mean_total_environment_steps": float(
            enriched["total_environment_steps"].mean()
        ),
    }


def analyze(endpoints_path: Path) -> dict[str, Any]:
    observed_hash = sha256_file(endpoints_path)
    if observed_hash != ENDPOINT_SHA256:
        raise ValueError(f"unexpected endpoint hash: {observed_hash}")
    endpoints = pd.read_csv(endpoints_path)
    fixed = endpoints[endpoints["policy"].isin(FIXED_Q)].copy()
    fixed["q"] = fixed["policy"].map(FIXED_Q).astype(int)

    task_budget = (
        fixed.groupby(["task", "budget", "policy", "q"], as_index=False)
        .agg(
            geometric_terminal_prediction_mse=(
                "terminal_prediction_mse",
                geometric_mean,
            )
        )
        .sort_values(
            ["task", "budget", "geometric_terminal_prediction_mse", "q"],
            kind="mergesort",
        )
    )
    fallback_rows = task_budget.groupby(["task", "budget"], as_index=False).first()
    observed_fallback = {
        (str(row.task), str(row.budget)): int(row.q)
        for row in fallback_rows.itertuples()
    }
    if observed_fallback != EXPECTED_FALLBACK:
        raise ValueError(f"unexpected task-budget fallback: {observed_fallback}")

    cell_arm = (
        fixed.groupby(SCENARIO + ["policy", "q"], as_index=False)
        .agg(
            geometric_terminal_prediction_mse=(
                "terminal_prediction_mse",
                geometric_mean,
            )
        )
        .sort_values(
            SCENARIO + ["geometric_terminal_prediction_mse", "q"],
            kind="mergesort",
        )
    )
    oracle_choice = cell_arm.groupby(SCENARIO, as_index=False).first()[
        SCENARIO + ["policy", "q", "geometric_terminal_prediction_mse"]
    ].rename(
        columns={
            "policy": "oracle_policy",
            "q": "oracle_q",
            "geometric_terminal_prediction_mse": "oracle_cell_geometric_error",
        }
    )
    fallback_cell = cell_arm[
        [
            int(q) == EXPECTED_FALLBACK[(str(task), str(budget))]
            for task, budget, q in zip(
                cell_arm["task"], cell_arm["budget"], cell_arm["q"]
            )
        ]
    ][SCENARIO + ["policy", "q", "geometric_terminal_prediction_mse"]].rename(
        columns={
            "policy": "fallback_policy",
            "q": "fallback_q",
            "geometric_terminal_prediction_mse": "fallback_cell_geometric_error",
        }
    )
    cells = fallback_cell.merge(oracle_choice, on=SCENARIO, validate="one_to_one")
    cells["oracle_over_fallback_ratio"] = (
        cells["oracle_cell_geometric_error"]
        / cells["fallback_cell_geometric_error"]
    )
    cells["relative_improvement"] = 1.0 - cells["oracle_over_fallback_ratio"]

    selected = fixed.merge(
        oracle_choice[SCENARIO + ["oracle_policy", "oracle_q"]],
        on=SCENARIO,
        validate="many_to_one",
    )
    oracle_seed_rows = selected[selected["policy"] == selected["oracle_policy"]].copy()
    fallback_seed_rows = fixed[
        [
            int(q) == EXPECTED_FALLBACK[(str(task), str(budget))]
            for task, budget, q in zip(fixed["task"], fixed["budget"], fixed["q"])
        ]
    ].copy()

    fallback_outcomes = aggregate_outcome_metrics(fallback_seed_rows)
    oracle_outcomes = aggregate_outcome_metrics(oracle_seed_rows)
    _, fallback_probe = append_probe_accounting(fallback_seed_rows)
    _, oracle_probe = append_probe_accounting(oracle_seed_rows)
    aggregate_ratio = geometric_mean(cells["oracle_over_fallback_ratio"])
    strict_cells = int((cells["relative_improvement"] > 0.0).sum())
    two_percent_cells = int((cells["relative_improvement"] >= 0.02).sum())
    five_percent_cells = int((cells["relative_improvement"] >= 0.05).sum())
    directional_fraction = strict_cells / len(cells)
    aggregate_improvement = 1.0 - aggregate_ratio
    current_gate = bool(
        aggregate_improvement >= ORACLE_AGGREGATE_GATE
        and directional_fraction >= ORACLE_DIRECTIONAL_GATE
    )

    return {
        "experiment": "T-020",
        "evidence_status": "descriptive_two_seed_design_audit_no_inference",
        "input": {
            "source_artifact": SOURCE_ARTIFACT,
            "sha256": observed_hash,
            "endpoint_rows": int(len(endpoints)),
            "fixed_q_rows_used": int(len(fixed)),
        },
        "strong_task_budget_fallback": [
            {
                "task": str(row.task),
                "budget": str(row.budget),
                "q": int(row.q),
                "policy": str(row.policy),
                "geometric_terminal_prediction_mse": float(
                    row.geometric_terminal_prediction_mse
                ),
            }
            for row in fallback_rows.itertuples()
        ],
        "cellwise_fixed_q_oracle": {
            "registered_cells": int(len(cells)),
            "oracle_over_fallback_geometric_ratio": aggregate_ratio,
            "relative_geometric_improvement": aggregate_improvement,
            "strictly_improved_cells": strict_cells,
            "strictly_improved_fraction": directional_fraction,
            "cells_at_least_2_percent": two_percent_cells,
            "cells_at_least_5_percent": five_percent_cells,
            "maximum_cell_improvement": float(cells["relative_improvement"].max()),
        },
        "full_probe_cost_optimistic_ceiling": {
            "interpretation": "outcomes retain the no-probe endpoint error/AUC despite fewer usable updates; actual post-probe outcomes are not identifiable from endpoints alone",
            "fallback": {**fallback_outcomes, **fallback_probe},
            "oracle": {**oracle_outcomes, **oracle_probe},
            "oracle_over_fallback_geometric_error_ratio": float(
                oracle_outcomes["geometric_terminal_prediction_mse"]
                / fallback_outcomes["geometric_terminal_prediction_mse"]
            ),
            "oracle_over_fallback_pooled_cvar90_ratio": float(
                oracle_outcomes["pooled_seed_cell_cvar90_terminal_prediction_mse"]
                / fallback_outcomes["pooled_seed_cell_cvar90_terminal_prediction_mse"]
            ),
            "oracle_over_fallback_mean_cellwise_cvar90_ratio": float(
                oracle_outcomes[
                    "mean_cellwise_two_seed_cvar90_terminal_prediction_mse"
                ]
                / fallback_outcomes[
                    "mean_cellwise_two_seed_cvar90_terminal_prediction_mse"
                ]
            ),
            "oracle_over_fallback_mean_auc_ratio": float(
                oracle_outcomes["mean_normalized_prediction_auc"]
                / fallback_outcomes["mean_normalized_prediction_auc"]
            ),
        },
        "gate_audit": {
            "old_no_harm_ratio_ceiling": 1.05,
            "always_fallback_ratio": 1.0,
            "always_fallback_trivially_passes_old_no_harm": True,
            "nontriviality_gate_required": True,
            "new_static_oracle_aggregate_improvement_minimum": ORACLE_AGGREGATE_GATE,
            "new_static_oracle_directional_fraction_minimum": ORACLE_DIRECTIONAL_GATE,
            "aggregate_gate_pass": aggregate_improvement >= ORACLE_AGGREGATE_GATE,
            "directional_gate_pass": directional_fraction >= ORACLE_DIRECTIONAL_GATE,
            "all_static_oracle_gates_pass": current_gate,
            "exp017b_permanently_stopped": not current_gate,
        },
        "pilot_seed_count": 2,
        "significance_inference_performed": False,
        "academic_research_suite_available": False,
        "surrogate_audit": {
            "formula_specified": True,
            "controller_inputs_observable_or_public": True,
            "candidate_complexity": "O(|Q||B|)=O(12) scalar operations; O(1) online state",
            "hessian_or_covariance_inverse": False,
            "cpu_tests_passed": 10,
            "conditional_safety_shield_specified": True,
            "relu_smoothness_and_confidence_certificate_complete": False,
            "full_surrogate_gate_pass": False,
        },
        "outcome_free_redesign_scan": {
            "candidates": {
                "FrozenLake-v1-8x8-slippery": {
                    "intrinsic_transition_stochasticity": True,
                    "public_transition_probabilities": "1/3 intended and 1/3 for each perpendicular direction",
                    "low_rho_variance_benefit": True,
                    "high_rho_saturation": True,
                    "message_internal_q_condition": True,
                    "delay_changes_usable_horizon": True,
                    "task_budget_oracle_value_at_least_5_percent_certified": False,
                    "all_static_feasibility_conditions_pass": False,
                },
                "MinAtar-Asterix-v1-sticky-0.1": {
                    "intrinsic_transition_and_reward_stochasticity": True,
                    "sticky_action_probability": 0.1,
                    "low_rho_variance_benefit": True,
                    "high_rho_saturation": True,
                    "message_internal_q_condition_with_small_public_model": True,
                    "delay_changes_usable_horizon": True,
                    "mixing_and_task_budget_oracle_value_at_least_5_percent_certified": False,
                    "all_static_feasibility_conditions_pass": False,
                },
            },
            "scientific_trajectories_generated": 0,
            "any_candidate_passes_all_static_conditions": False,
        },
        "scientific_trajectories_generated": 0,
        "gpu_or_slurm_jobs_submitted": 0,
        "exp017b_preregistration_authorized": False,
        "new_number_gpu_pilot_preregistration_authorized": False,
    }


def write_json(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoints", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.endpoints)
    write_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
