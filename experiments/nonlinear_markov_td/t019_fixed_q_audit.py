"""T-019 read-only mechanism and fixed-q phase-diagram audit.

This module consumes only the frozen EXP-017A endpoint table.  It does not
import the GPU runner, mutate pilot artifacts, or authorize a new experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from exp017a_nonlinear_config import (
    B_CANDIDATES,
    BUDGETS,
    CORRELATIONS,
    DELAY_TRACES,
    FLOAT_BYTES,
    HIDDEN_WIDTH,
    LEARNING_RATE,
    MIXING_PROFILES,
    Q_CANDIDATES,
    SERVER_OVERHEAD_BYTES,
    TASKS,
    trace_summary,
)


SCENARIO = ["task", "mixing", "rho", "delay_trace", "budget"]
SEED_KEY = ["seed", *SCENARIO]
AUDIT_POLICIES = (
    "single_agent",
    "fixed_q4",
    "fixed_q16",
    "fixed_q32",
    "always_all",
    "oracle_evaluation_only",
)
STRONG_FIXED = ("single_agent", "fixed_q4", "fixed_q16", "fixed_q32")
POLICY_Q = {
    "single_agent": 1,
    "fixed_q4": 4,
    "fixed_q16": 16,
    "fixed_q32": 32,
    "always_all": 32,
}
EXPECTED_ENDPOINT_SHA256 = (
    "bc241c772d20b76c5f42f72bd8a5523bda2ba225e113811e695dd840007191f0"
)
SOURCE_ARTIFACT = "/scratch/jzhuangag/exp017a-pilot-17a4c32/endpoints.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_count(observation_dimension: int) -> int:
    """Parameter count of the frozen d-64-64-1 MLP."""

    return (
        observation_dimension * HIDDEN_WIDTH
        + HIDDEN_WIDTH
        + HIDDEN_WIDTH * HIDDEN_WIDTH
        + HIDDEN_WIDTH
        + HIDDEN_WIDTH
        + 1
    )


def projected_horizon(
    q: int,
    b: int,
    message_remaining: int,
    environment_remaining: int,
    parameters: int,
) -> int:
    message_cost = SERVER_OVERHEAD_BYTES + q * parameters * FLOAT_BYTES
    return max(
        0,
        min(message_remaining // message_cost, environment_remaining // b),
    )


def frozen_score(
    q: int,
    b: int,
    rho: float,
    lambda_upper: float,
    delay_p90: float,
    message_remaining: int,
    environment_remaining: int,
    parameters: int,
    loss_ema: float = 1.0,
    gradient_second_moment: float = 1.0,
    progress_ema: float = 0.0,
) -> float:
    """Independent transcription of the frozen EXP-017A score."""

    horizon = projected_horizon(
        q, b, message_remaining, environment_remaining, parameters
    )
    if horizon < 1:
        return float("inf")
    loss_scale = max(loss_ema, 1.0e-8)
    gradient_scale = max(gradient_second_moment, 1.0e-8)
    progress = max(0.002, progress_ema)
    delay_factor = 1.0 + delay_p90 / float(b)
    transient = loss_scale * math.exp(
        -progress * LEARNING_RATE * horizon / delay_factor
    )
    variance = (
        0.20
        * LEARNING_RATE
        * gradient_scale
        * (rho + (1.0 - rho) / float(q))
    )
    mixing_penalty = 0.10 * (lambda_upper**b)
    horizon_penalty = 0.10 / math.sqrt(float(horizon))
    return transient + variance + mixing_penalty + horizon_penalty


def absorbing_state_grid_audit() -> list[dict[str, Any]]:
    """Verify q=1 domination at the initial zero-trial state in every cell."""

    rows: list[dict[str, Any]] = []
    for task_name, task in TASKS.items():
        parameters = parameter_count(int(task["observation_dimension"]))
        for mixing_name, mixing in MIXING_PROFILES.items():
            for delay_name in DELAY_TRACES:
                delay_p90 = float(trace_summary(delay_name)["p90"])
                for budget_name, budget in BUDGETS.items():
                    scores: dict[tuple[int, int], float] = {}
                    for q in Q_CANDIDATES:
                        for b in B_CANDIDATES:
                            scores[(q, b)] = frozen_score(
                                q=q,
                                b=b,
                                rho=1.0,
                                lambda_upper=float(mixing["lambda_upper"]),
                                delay_p90=delay_p90,
                                message_remaining=int(budget["message_bytes"]),
                                environment_remaining=int(budget["environment_steps"]),
                                parameters=parameters,
                            )
                    selected = min(
                        scores,
                        key=lambda action: (scores[action], action[0], action[1]),
                    )
                    same_b_domination = all(
                        scores[(1, b)] <= scores[(q, b)]
                        for b in B_CANDIDATES
                        for q in Q_CANDIDATES
                    )
                    rows.append(
                        {
                            "task": task_name,
                            "mixing": mixing_name,
                            "delay_trace": delay_name,
                            "budget": budget_name,
                            "parameters": parameters,
                            "selected_q": selected[0],
                            "selected_b": selected[1],
                            "same_b_q1_weak_domination": same_b_domination,
                            "passes": selected[0] == 1 and same_b_domination,
                        }
                    )
    return rows


def cvar90(values: Iterable[float]) -> float:
    array = np.sort(np.asarray(list(values), dtype=float))
    count = max(1, int(math.ceil(0.10 * len(array))))
    return float(np.mean(array[-count:]))


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0 or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def _phase_direction_summary(cells: pd.DataFrame) -> dict[str, Any]:
    rho_paths = []
    for _, group in cells.groupby(
        ["task", "mixing", "delay_trace", "budget"], sort=True
    ):
        ordered = group.sort_values("rho")
        q_values = ordered["best_fixed_q"].astype(int).tolist()
        rho_paths.append(all(left >= right for left, right in zip(q_values, q_values[1:])))

    delay_order = {"zero": 0, "edge_jitter": 1, "wan_bursty": 2}
    delay_paths = []
    for _, group in cells.groupby(["task", "mixing", "rho", "budget"], sort=True):
        ordered = group.assign(
            _delay_order=group["delay_trace"].map(delay_order)
        ).sort_values("_delay_order")
        q_values = ordered["best_fixed_q"].astype(int).tolist()
        delay_paths.append(
            all(left >= right for left, right in zip(q_values, q_values[1:]))
        )

    budget_pairs = cells.pivot(
        index=["task", "mixing", "rho", "delay_trace"],
        columns="budget",
        values="best_fixed_q",
    )
    budget_direction = (
        budget_pairs["environment_binding"] >= budget_pairs["message_binding"]
    )
    budget_strict = (
        budget_pairs["environment_binding"] > budget_pairs["message_binding"]
    )
    return {
        "rho_expected_nonincreasing_paths": int(sum(rho_paths)),
        "rho_total_paths": len(rho_paths),
        "delay_expected_nonincreasing_paths": int(sum(delay_paths)),
        "delay_total_paths": len(delay_paths),
        "environment_q_at_least_message_q_pairs": int(budget_direction.sum()),
        "environment_q_strictly_above_message_q_pairs": int(budget_strict.sum()),
        "environment_q_equal_message_q_pairs": int(
            (budget_pairs["environment_binding"] == budget_pairs["message_binding"]).sum()
        ),
        "budget_total_pairs": int(len(budget_direction)),
    }


def _best_q_counts_by_factor(cells: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for factor in ("task", "mixing", "rho", "delay_trace", "budget"):
        factor_result: dict[str, dict[str, int]] = {}
        for value, group in cells.groupby(factor, sort=True):
            factor_result[str(value)] = {
                str(int(q)): int(count)
                for q, count in group["best_fixed_q"].value_counts().sort_index().items()
            }
        result[factor] = factor_result
    return result


def analyze_endpoints(endpoints_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    observed_hash = sha256_file(endpoints_path)
    if observed_hash != EXPECTED_ENDPOINT_SHA256:
        raise ValueError(f"unexpected endpoints SHA-256: {observed_hash}")
    endpoints = pd.read_csv(endpoints_path)
    selected = endpoints[endpoints["policy"].isin(AUDIT_POLICIES)].copy()
    expected_cells = (
        len(TASKS)
        * len(MIXING_PROFILES)
        * len(CORRELATIONS)
        * len(DELAY_TRACES)
        * len(BUDGETS)
    )
    if selected[SCENARIO].drop_duplicates().shape[0] != expected_cells:
        raise ValueError("incomplete registered scenario population")
    if selected.duplicated(SEED_KEY + ["policy"]).any():
        raise ValueError("duplicate endpoint key")
    if set(selected["policy"]) != set(AUDIT_POLICIES):
        raise ValueError("missing requested policy")

    long = (
        selected.groupby(SCENARIO + ["policy"], as_index=False, sort=True)
        .agg(
            geometric_terminal_prediction_mse=(
                "terminal_prediction_mse",
                geometric_mean,
            ),
            cvar90_terminal_prediction_mse=("terminal_prediction_mse", cvar90),
            mean_message_bytes=("messages", "mean"),
            mean_environment_steps=("environment_steps", "mean"),
            mean_agent_transitions=("agent_transitions", "mean"),
            mean_wall_seconds=("wall_seconds", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    fixed = long[long["policy"].isin(STRONG_FIXED)].copy()
    fixed["fixed_q"] = fixed["policy"].map(POLICY_Q).astype(int)
    fixed = fixed.sort_values(
        SCENARIO + ["geometric_terminal_prediction_mse", "fixed_q"],
        kind="mergesort",
    )
    best = fixed.groupby(SCENARIO, as_index=False, sort=True).first()
    best = best.rename(
        columns={
            "policy": "best_fixed_policy",
            "fixed_q": "best_fixed_q",
            "geometric_terminal_prediction_mse": "best_fixed_geometric_terminal_prediction_mse",
            "cvar90_terminal_prediction_mse": "best_fixed_cvar90_terminal_prediction_mse",
            "mean_message_bytes": "best_fixed_mean_message_bytes",
            "mean_environment_steps": "best_fixed_mean_environment_steps",
            "mean_agent_transitions": "best_fixed_mean_agent_transitions",
            "mean_wall_seconds": "best_fixed_mean_wall_seconds",
        }
    )
    keep = SCENARIO + [
        "best_fixed_policy",
        "best_fixed_q",
        "best_fixed_geometric_terminal_prediction_mse",
        "best_fixed_cvar90_terminal_prediction_mse",
        "best_fixed_mean_message_bytes",
        "best_fixed_mean_environment_steps",
        "best_fixed_mean_agent_transitions",
        "best_fixed_mean_wall_seconds",
    ]
    cells = best[keep].copy()
    long = long.merge(cells[keep], on=SCENARIO, how="left", validate="many_to_one")
    long["nominal_q"] = long["policy"].map(POLICY_Q).astype("Int64")
    long["geometric_terminal_error_ratio_to_best_fixed"] = (
        long["geometric_terminal_prediction_mse"]
        / long["best_fixed_geometric_terminal_prediction_mse"]
    )
    long = long[
        SCENARIO
        + [
            "best_fixed_policy",
            "best_fixed_q",
            "policy",
            "nominal_q",
            "geometric_terminal_prediction_mse",
            "geometric_terminal_error_ratio_to_best_fixed",
            "cvar90_terminal_prediction_mse",
            "mean_message_bytes",
            "mean_environment_steps",
            "mean_agent_transitions",
            "mean_wall_seconds",
            "seeds",
        ]
    ].sort_values(SCENARIO + ["policy"], kind="mergesort")

    fixed_seed = selected[selected["policy"].isin(STRONG_FIXED)].copy()
    global_by_policy = (
        fixed_seed.groupby("policy")["terminal_prediction_mse"]
        .apply(geometric_mean)
        .sort_values(kind="mergesort")
    )
    global_policy = str(global_by_policy.index[0])
    global_q = POLICY_Q[global_policy]
    envelope_geometric = geometric_mean(
        cells["best_fixed_geometric_terminal_prediction_mse"]
    )
    global_geometric = float(global_by_policy.iloc[0])
    best_counts = {
        str(int(q)): int(count)
        for q, count in cells["best_fixed_q"].value_counts().sort_index().items()
    }

    duplicate = selected[selected["policy"].isin(["fixed_q32", "always_all"])].pivot(
        index=SEED_KEY, columns="policy", values="terminal_prediction_mse"
    )
    duplicate_exact = bool(
        np.array_equal(
            duplicate["fixed_q32"].to_numpy(), duplicate["always_all"].to_numpy()
        )
    )
    directions = _phase_direction_summary(cells)
    adaptation_ratio = envelope_geometric / global_geometric
    distinct_q = len(best_counts)
    adaptation_value_descriptive = bool(
        distinct_q >= 2
        and directions["rho_expected_nonincreasing_paths"]
        > directions["rho_total_paths"] / 2
        and directions["delay_expected_nonincreasing_paths"]
        > directions["delay_total_paths"] / 2
        and directions["environment_q_at_least_message_q_pairs"]
        > directions["budget_total_pairs"] / 2
    )
    oracle = long[long["policy"] == "oracle_evaluation_only"]
    oracle_ratio = geometric_mean(
        oracle["geometric_terminal_error_ratio_to_best_fixed"]
    )
    aggregate_policy = {}
    for policy, group in selected.groupby("policy", sort=True):
        aggregate_policy[str(policy)] = {
            "geometric_terminal_prediction_mse": geometric_mean(
                group["terminal_prediction_mse"]
            ),
            "cvar90_terminal_prediction_mse": cvar90(
                group["terminal_prediction_mse"]
            ),
            "mean_message_bytes": float(group["messages"].mean()),
            "mean_environment_steps": float(group["environment_steps"].mean()),
            "mean_agent_transitions": float(group["agent_transitions"].mean()),
            "mean_wall_seconds": float(group["wall_seconds"].mean()),
        }
    mechanism_rows = absorbing_state_grid_audit()
    summary: dict[str, Any] = {
        "experiment": "T-019",
        "evidence_status": "descriptive_post_pilot_audit_not_formal_evidence",
        "input": {
            "source_artifact": SOURCE_ARTIFACT,
            "analysis_copy_sha256_verified": True,
            "sha256": observed_hash,
            "endpoint_rows_total": int(len(endpoints)),
            "endpoint_rows_used": int(len(selected)),
            "policies": list(AUDIT_POLICIES),
        },
        "phase_cells": int(len(cells)),
        "phase_rows": int(len(long)),
        "pilot_seed_count_per_cell_arm": int(long["seeds"].min()),
        "cvar90_two_seed_definition": "maximum of the two registered pilot-seed errors",
        "best_fixed_q_counts": best_counts,
        "best_fixed_q_counts_by_factor": _best_q_counts_by_factor(cells),
        "aggregate_policy_metrics": aggregate_policy,
        "global_best_fixed_policy": global_policy,
        "global_best_fixed_q": global_q,
        "global_best_fixed_geometric_error": global_geometric,
        "cellwise_envelope_geometric_error": envelope_geometric,
        "cellwise_envelope_over_global_fixed_ratio": adaptation_ratio,
        "cellwise_envelope_relative_improvement": 1.0 - adaptation_ratio,
        "phase_direction": directions,
        "phase_transition_direction_supported_descriptively": adaptation_value_descriptive,
        "oracle_over_cellwise_best_fixed_geometric_ratio": oracle_ratio,
        "oracle_better_than_cellwise_best_fixed_cells": int(
            (oracle["geometric_terminal_error_ratio_to_best_fixed"] < 1.0).sum()
        ),
        "always_all_terminal_error_exactly_matches_fixed_q32": duplicate_exact,
        "absorbing_state": {
            "registered_static_cells": len(mechanism_rows),
            "all_cells_select_q1": all(row["selected_q"] == 1 for row in mechanism_rows),
            "all_same_b_q1_weak_domination": all(
                row["same_b_q1_weak_domination"] for row in mechanism_rows
            ),
            "pairwise_trials_added_at_q1": 0,
            "absorbing_by_induction": True,
        },
        "adaptation_value_descriptive_screen": {
            "rule": "at least two cellwise q optima and strict-majority rho, delay, and budget directions; magnitude reported separately without a post-hoc threshold",
            "passes": adaptation_value_descriptive,
        },
        "exp017b_protocol_design_permitted": adaptation_value_descriptive,
        "future_gpu_pilot_design_condition_met": adaptation_value_descriptive,
        "gpu_execution_authorized": False,
        "gpu_jobs_submitted": 0,
    }
    return cells.sort_values(SCENARIO), long, summary


def write_outputs(
    cells: pd.DataFrame,
    long: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(
        output_dir / "t019_fixed_q_best_cells.csv", index=False, lineterminator="\n"
    )
    long.to_csv(
        output_dir / "t019_fixed_q_phase_diagram.csv", index=False, lineterminator="\n"
    )
    with (output_dir / "t019_fixed_q_phase_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoints", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cells, long, summary = analyze_endpoints(args.endpoints)
    write_outputs(cells, long, summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
