"""Frozen descriptive progression-gate analyzer for EXP-017A pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from exp017a_nonlinear_config import (
    BUDGETS,
    CORRELATIONS,
    DELAY_TRACES,
    EXPERIMENT,
    FIXED_Q_POLICIES,
    MIXING_PROFILES,
    PILOT_NONINFERIORITY_RATIO,
    PILOT_SEEDS,
    PILOT_TAIL_RATIO,
    POLICIES,
    PRIMARY_EFFECT_THRESHOLD,
    TASKS,
    expected_runs,
)


KEY = ["seed", "task", "mixing", "rho", "delay_trace", "budget"]
SCENARIO_KEY = ["task", "mixing", "rho", "delay_trace", "budget"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cvar90(values: Iterable[float]) -> float:
    array = np.sort(np.asarray(list(values), dtype=float))
    count = max(1, int(np.ceil(0.10 * len(array))))
    return float(np.mean(array[-count:]))


def load_pilot(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    endpoints = []
    trajectories = []
    metadata = []
    for seed in PILOT_SEEDS:
        directory = root / "seeds" / str(seed)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        endpoints.append(pd.read_csv(directory / "endpoints.csv"))
        trajectories.append(pd.read_csv(directory / "trajectories.csv"))
        metadata.append(json.loads((directory / "metadata.json").read_text(encoding="utf-8")))
    return (
        pd.concat(endpoints, ignore_index=True),
        pd.concat(trajectories, ignore_index=True),
        metadata,
    )


def freeze_best_fixed_q(endpoints: pd.DataFrame) -> pd.DataFrame:
    fixed = endpoints[endpoints["policy"].isin(FIXED_Q_POLICIES)].copy()
    grouped = (
        fixed.groupby(SCENARIO_KEY + ["policy"], as_index=False)
        .agg(mean_log_error=("terminal_prediction_mse", lambda x: float(np.mean(np.log(x)))))
        .sort_values(SCENARIO_KEY + ["mean_log_error", "policy"], kind="mergesort")
    )
    return grouped.groupby(SCENARIO_KEY, as_index=False).first()[
        SCENARIO_KEY + ["policy", "mean_log_error"]
    ].rename(columns={"policy": "pilot_selected_best_fixed_q"})


def paired_ratios(
    endpoints: pd.DataFrame,
    numerator: str,
    denominator: str,
    subset: pd.Series | None = None,
) -> np.ndarray:
    selected = endpoints if subset is None else endpoints[subset]
    pivot = selected.pivot(index=KEY, columns="policy", values="terminal_prediction_mse")
    return pivot[numerator].to_numpy(dtype=float) / pivot[denominator].to_numpy(dtype=float)


def analyze(root: Path) -> dict[str, object]:
    endpoints, trajectories, metadata = load_pilot(root)
    if len(endpoints) != expected_runs():
        raise ValueError(f"expected {expected_runs()} endpoints, found {len(endpoints)}")
    if endpoints.duplicated(KEY + ["policy"]).any():
        raise ValueError("duplicate endpoint keys")
    expected_policies = set(POLICIES)
    population_complete = True
    for _, group in endpoints.groupby(KEY):
        population_complete &= set(group["policy"]) == expected_policies
    best_fixed = freeze_best_fixed_q(endpoints)
    enriched = endpoints.merge(best_fixed, on=SCENARIO_KEY, how="left", validate="many_to_one")
    fixed_lookup = enriched[enriched["policy"] == enriched["pilot_selected_best_fixed_q"]][
        KEY + ["terminal_prediction_mse"]
    ].rename(columns={"terminal_prediction_mse": "best_fixed_prediction_mse"})
    enriched = enriched.merge(fixed_lookup, on=KEY, how="left", validate="many_to_one")
    primary_mask = (endpoints["rho"] == 0.9) & (endpoints["delay_trace"] != "zero")
    primary_ratios = paired_ratios(endpoints, "learning_aware", "information_only", primary_mask)
    task_ratios = {}
    for task in TASKS:
        mask = primary_mask & (endpoints["task"] == task)
        values = paired_ratios(endpoints, "learning_aware", "information_only", mask)
        task_ratios[task] = float(np.exp(np.mean(np.log(values))))
    learning = enriched[enriched["policy"] == "learning_aware"].copy()
    fixed_ratio = (
        learning["terminal_prediction_mse"].to_numpy(dtype=float)
        / learning["best_fixed_prediction_mse"].to_numpy(dtype=float)
    )
    q_summary = (
        trajectories[trajectories["policy"] == "learning_aware"]
        .groupby(["rho", "delay_trace", "budget"], as_index=False)
        .agg(median_q=("q", "median"), median_b=("b", "median"))
    )
    low_q = float(q_summary[q_summary["rho"] == 0.0]["median_q"].median())
    high_q = float(q_summary[q_summary["rho"] == 0.9]["median_q"].median())
    zero_b = float(q_summary[q_summary["delay_trace"] == "zero"]["median_b"].median())
    delayed_b = float(q_summary[q_summary["delay_trace"] != "zero"]["median_b"].median())
    finite_budget_valid = bool(
        endpoints["finite"].all()
        and np.isfinite(
            endpoints[
                [
                    "terminal_prediction_mse",
                    "terminal_bellman_error",
                    "normalized_prediction_auc",
                    "wall_seconds",
                    "controller_wall_seconds",
                ]
            ].to_numpy(dtype=float)
        ).all()
        and (endpoints["messages"] <= endpoints["message_budget"]).all()
        and (endpoints["environment_steps"] <= endpoints["environment_budget"]).all()
    )
    metadata_valid = bool(
        all(item["information_only_taint_audit"]["passes"] for item in metadata)
        and all(len(item["bank_audit"]) == len(TASKS) * len(MIXING_PROFILES) for item in metadata)
    )
    mixing_valid = all(
        abs(float(spec["lambda_upper"]) + float(spec["gamma_certificate"]) - 1.0) < 1e-12
        for spec in MIXING_PROFILES.values()
    )
    controller_fraction = float(
        endpoints["controller_wall_seconds"].sum()
        / max(endpoints["wall_seconds"].sum(), 1.0e-12)
    )
    overall_primary_ratio = float(np.exp(np.mean(np.log(primary_ratios))))
    fixed_geometric_ratio = float(np.exp(np.mean(np.log(fixed_ratio))))
    fixed_tail_ratio = cvar90(learning["terminal_prediction_mse"]) / cvar90(
        learning["best_fixed_prediction_mse"]
    )
    required_values = {
        "tasks": set(endpoints["task"]) == set(TASKS),
        "mixing": set(endpoints["mixing"]) == set(MIXING_PROFILES),
        "correlations": set(endpoints["rho"]) == set(CORRELATIONS),
        "delays": set(endpoints["delay_trace"]) == set(DELAY_TRACES),
        "budgets": set(endpoints["budget"]) == set(BUDGETS),
        "policies": set(endpoints["policy"]) == set(POLICIES),
        "seeds": set(endpoints["seed"]) == set(PILOT_SEEDS),
    }
    gates = {
        "G1_all_finite_and_dual_budget_valid": finite_budget_valid,
        "G2_exact_registered_population": bool(population_complete and all(required_values.values())),
        "G3_known_mixing_certificates_valid": bool(mixing_valid),
        "G4_common_private_marginal_construction_recorded": bool(metadata_valid),
        "G5_information_only_taint_audit": bool(metadata_valid),
        "G6_communication_matched_budgets": bool(
            endpoints.groupby(KEY)["message_budget"].nunique().max() == 1
            and endpoints.groupby(KEY)["environment_budget"].nunique().max() == 1
        ),
        "G7_correlation_response_high_q_below_low_q": bool(high_q < low_q),
        "G8_delay_response_b_nondecreasing": bool(delayed_b >= zero_b),
        "G9_primary_high_correlation_delayed_improvement": bool(
            overall_primary_ratio <= 1.0 - PRIMARY_EFFECT_THRESHOLD
        ),
        "G10_both_standard_tasks_directionally_consistent": bool(
            all(value <= 1.05 for value in task_ratios.values())
        ),
        "G11_best_fixed_q_noninferiority_and_tail": bool(
            fixed_geometric_ratio <= PILOT_NONINFERIORITY_RATIO
            and fixed_tail_ratio <= PILOT_TAIL_RATIO
        ),
        "G12_controller_overhead_and_metrics_complete": bool(
            controller_fraction <= 0.10
            and (endpoints["server_ticks"] > 0).all()
            and (endpoints["agent_transitions"] > 0).all()
        ),
    }
    cell_summary = (
        endpoints.groupby(
            ["task", "mixing", "rho", "delay_trace", "budget", "policy"],
            as_index=False,
        )
        .agg(
            geometric_prediction_mse=(
                "terminal_prediction_mse",
                lambda x: float(np.exp(np.mean(np.log(x)))),
            ),
            mean_bellman_error=("terminal_bellman_error", "mean"),
            mean_normalized_auc=("normalized_prediction_auc", "mean"),
            mean_messages=("messages", "mean"),
            mean_environment_steps=("environment_steps", "mean"),
            mean_agent_transitions=("agent_transitions", "mean"),
            mean_wall_seconds=("wall_seconds", "mean"),
            mean_controller_wall_seconds=("controller_wall_seconds", "mean"),
        )
    )
    cell_summary.to_csv(root / "cell_summary.csv", index=False)
    best_fixed.to_csv(root / "pilot_selected_best_fixed_q.csv", index=False)
    endpoints.sort_values(KEY + ["policy"], kind="mergesort").to_csv(
        root / "endpoints.csv", index=False
    )
    trajectories.sort_values(KEY + ["policy", "block"], kind="mergesort").to_csv(
        root / "trajectories.csv", index=False
    )
    result = {
        "experiment": EXPERIMENT,
        "evidence_status": "implementation_only_gpu_pilot",
        "pilot_seeds_excluded_from_formal": list(PILOT_SEEDS),
        "endpoint_rows": len(endpoints),
        "trajectory_rows": len(trajectories),
        "primary_learning_over_information_geometric_ratio": overall_primary_ratio,
        "primary_relative_improvement": 1.0 - overall_primary_ratio,
        "task_primary_ratios": task_ratios,
        "learning_over_pilot_best_fixed_geometric_ratio": fixed_geometric_ratio,
        "learning_over_pilot_best_fixed_cvar90_ratio": fixed_tail_ratio,
        "low_rho_median_q": low_q,
        "high_rho_median_q": high_q,
        "zero_delay_median_b": zero_b,
        "nonzero_delay_median_b": delayed_b,
        "controller_wall_fraction": controller_fraction,
        "gates": gates,
        "all_mandatory_gates_pass": bool(all(gates.values())),
        "formal_authorized": bool(all(gates.values())),
        "negative_result_rule": "any failed gate stops formal without threshold, seed, or population changes",
    }
    (root / "pilot_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        name: sha256_file(root / name)
        for name in (
            "endpoints.csv",
            "trajectories.csv",
            "cell_summary.csv",
            "pilot_selected_best_fixed_q.csv",
            "pilot_summary.json",
        )
    }
    (root / "SHA256SUMS.json").write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**result, "artifact_sha256": artifact_hashes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(analyze(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
