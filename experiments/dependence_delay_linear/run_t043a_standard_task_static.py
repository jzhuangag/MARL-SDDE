"""Preregistered outcome-free standard-task feasibility runner for T-043A."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.t035_scalar_phase_theorem import (
    exact_scalar_risk,
)
from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    dual_budget_updates,
)
from experiments.dependence_delay_linear.t043a_standard_task_static import (
    build_projected_task,
    public_task_summary,
    stable_json_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs" / "t043a_standard_task_static_preregistration.json"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_tasks(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if config["software"]["gymnasium"] != "1.0.0":
        raise ValueError("T-043A is pinned to Gymnasium 1.0.0")
    return {
        name: build_projected_task(
            environment_id=task_config["environment_id"],
            kwargs=task_config["kwargs"],
            feature_dimension=task_config["feature_dimension"],
            discount=config["software"]["discount"],
        )
        for name, task_config in config["software"]["tasks"].items()
    }


def scenario_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["grid"]
    rows: list[dict[str, Any]] = []
    for task, ray, delay, rho in product(
        config["software"]["tasks"],
        grid["budget_rays"],
        grid["delay"],
        grid["rho"],
    ):
        ray_config = grid["budget_rays"][ray]
        varying_key = "message_budget" if isinstance(ray_config["message_budget"], list) else "environment_budget"
        budgets = ray_config[varying_key]
        for budget_index, budget in enumerate(budgets):
            message_budget = (
                budget if varying_key == "message_budget" else ray_config["message_budget"]
            )
            environment_budget = (
                budget if varying_key == "environment_budget" else ray_config["environment_budget"]
            )
            rows.append(
                {
                    "scenario_id": f"{task}-{ray}-B{budget_index}-D{delay}-R{rho:g}",
                    "task": task,
                    "ray": ray,
                    "budget": int(budget),
                    "budget_index": int(budget_index),
                    "message_budget": int(message_budget),
                    "environment_budget": int(environment_budget),
                    "delay": int(delay),
                    "rho": float(rho),
                }
            )
    return rows


def static_validate(
    config: dict[str, Any], tasks: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    task_data = build_tasks(config) if tasks is None else tasks
    scenarios = scenario_rows(config)
    action_count = len(config["grid"]["q"]) * len(config["grid"]["normalized_step"])
    expected = config["expected_workload"]
    if len(scenarios) != expected["scenarios"]:
        raise ValueError("scenario count differs from the frozen workload")
    if action_count != expected["actions_per_scenario"]:
        raise ValueError("action count differs from the frozen workload")
    if len(scenarios) * action_count != expected["rows"]:
        raise ValueError("row count differs from the frozen workload")
    if config["analysis"]["no_seeds_or_confidence_intervals"] is not True:
        raise ValueError("T-043A must remain outcome-free")
    if config["sampled_learning_trajectory_authorized"] is not False:
        raise ValueError("sampled trajectories are forbidden in T-043A")
    if config["formal_or_gpu_authorized"] is not False:
        raise ValueError("formal and GPU execution are forbidden")
    for task in task_data.values():
        np.testing.assert_allclose(task["continuing_transition"].sum(axis=1), 1.0)
        np.testing.assert_allclose(task["stationary"] @ task["continuing_transition"], task["stationary"])
        np.testing.assert_allclose(
            task["features"].T @ (task["stationary"][:, None] * task["features"]),
            np.eye(task["features"].shape[1]),
            atol=1e-10,
        )
    return {
        "experiment_id": config["experiment_id"],
        "tasks": len(task_data),
        "scenarios": len(scenarios),
        "actions_per_scenario": action_count,
        "rows": len(scenarios) * action_count,
        "sampled_trajectories": 0,
        "recommended_hardware": "local CPU",
    }


def estimate(config: dict[str, Any], tasks: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    validated = static_validate(config, tasks)
    maximum_horizon = 0
    for scenario in scenario_rows(config):
        for q in config["grid"]["q"]:
            maximum_horizon = max(
                maximum_horizon,
                dual_budget_updates(
                    message_budget=scenario["message_budget"],
                    environment_budget=scenario["environment_budget"],
                    message_cost=config["grid"]["message_overhead"] + q,
                    stride=1,
                    delay=scenario["delay"],
                ),
            )
    if maximum_horizon != config["expected_workload"]["maximum_scalar_horizon"]:
        raise ValueError("maximum horizon differs from frozen workload")
    return {**validated, "maximum_scalar_horizon": maximum_horizon}


@lru_cache(maxsize=None)
def unit_scalar_components(
    initial_error: float,
    drift_minimum: float,
    step_size: float,
    delay: int,
    updates: int,
    mixing_slem: float,
) -> tuple[float, float, float]:
    bias = exact_scalar_risk(
        initial_error=initial_error,
        mu=drift_minimum,
        step_size=step_size,
        delay=delay,
        updates=updates,
        single_variance=0.0,
        q=1,
        rho=0.0,
        markov_lambda=mixing_slem,
    )
    unit_noise = exact_scalar_risk(
        initial_error=0.0,
        mu=drift_minimum,
        step_size=step_size,
        delay=delay,
        updates=updates,
        single_variance=1.0,
        q=1,
        rho=0.0,
        markov_lambda=mixing_slem,
    )
    return bias["risk"], unit_noise["risk"], bias["spectral_radius"]


def evaluate_action(
    scenario: dict[str, Any],
    task: dict[str, Any],
    *,
    q: int,
    normalized_step: float,
    overhead: int,
) -> dict[str, Any]:
    updates = dual_budget_updates(
        message_budget=scenario["message_budget"],
        environment_budget=scenario["environment_budget"],
        message_cost=overhead + q,
        stride=1,
        delay=scenario["delay"],
    )
    step_size = normalized_step / task["drift_norm"]
    bias, unit_noise, spectral_radius = unit_scalar_components(
        task["initial_error"],
        task["drift_minimum"],
        step_size,
        scenario["delay"],
        updates,
        task["mixing_slem"],
    )
    dependence_factor = scenario["rho"] + (1.0 - scenario["rho"]) / q
    noise = unit_noise * task["single_agent_noise_second"] * dependence_factor
    risk = bias + noise
    return {
        **scenario,
        "q": int(q),
        "normalized_step": float(normalized_step),
        "step_size": float(step_size),
        "updates": int(updates),
        "spectral_radius": float(spectral_radius),
        "bias_risk": float(bias),
        "noise_risk": float(noise),
        "risk": float(risk),
    }


def _geometric_mean(values: list[float]) -> float:
    return float(math.exp(np.mean(np.log(np.maximum(values, 1e-300)))))


def analyze(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario_id"], []).append(row)
    best_q_rows: list[dict[str, Any]] = []
    for scenario_id, candidates in by_scenario.items():
        per_q = []
        for q in config["grid"]["q"]:
            q_rows = [row for row in candidates if row["q"] == q]
            per_q.append(min(q_rows, key=lambda row: (row["risk"], row["normalized_step"])))
        oracle = min(per_q, key=lambda row: (row["risk"], row["q"], row["normalized_step"]))
        enriched = dict(oracle)
        enriched["per_q"] = {int(row["q"]): row for row in per_q}
        best_q_rows.append(enriched)

    speed_cells = [row for row in best_q_rows if row["ray"] == "environment" and row["rho"] == 0.0]
    speed_fraction = float(
        np.mean([row["per_q"][16]["risk"] / row["per_q"][1]["risk"] <= 0.95 for row in speed_cells])
    )
    saturation_cells = [row for row in best_q_rows if row["ray"] == "environment" and row["rho"] == 1.0]
    saturation_max_value = float(
        max(1.0 - row["risk"] / row["per_q"][1]["risk"] for row in saturation_cells)
    )
    reversal_cells = [row for row in best_q_rows if row["ray"] == "message" and row["rho"] == 1.0]
    reversal_fraction = float(np.mean([row["q"] == 1 for row in reversal_cells]))
    support = {
        task: sorted({int(row["q"]) for row in best_q_rows if row["task"] == task})
        for task in config["software"]["tasks"]
    }

    fixed_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in best_q_rows:
        key = (row["task"], row["ray"], row["budget"], row["delay"])
        fixed_groups.setdefault(key, []).append(row)
    comparisons = []
    for group in fixed_groups.values():
        q_scores = {
            q: _geometric_mean([row["per_q"][q]["risk"] for row in group])
            for q in config["grid"]["q"]
        }
        fixed_q = min(q_scores, key=lambda q: (q_scores[q], q))
        for row in group:
            if row["ray"] != "message":
                continue
            fixed_risk = row["per_q"][fixed_q]["risk"]
            comparisons.append(
                {
                    "oracle_risk": row["risk"],
                    "fixed_risk": fixed_risk,
                    "ratio": row["risk"] / fixed_risk,
                    "strict": row["risk"] < fixed_risk * (1.0 - 1e-12),
                }
            )
    oracle_ratio = _geometric_mean([row["ratio"] for row in comparisons])
    strict_fraction = float(np.mean([row["strict"] for row in comparisons]))
    gates = {
        "S1": True,
        "S2": all(
            np.isfinite(row["risk"])
            and row["risk"] >= 0.0
            and row["spectral_radius"] < 1.0
            and row["updates"] >= 1
            for row in rows
        ),
        "S3": speed_fraction >= 0.80,
        "S4": saturation_max_value <= 0.01 + 1e-12,
        "S5": reversal_fraction >= 0.90,
        "S6": all(1 in values and 16 in values for values in support.values()),
        "S7": 1.0 - oracle_ratio >= 0.05,
        "S8": strict_fraction >= 0.40,
        "S9": True,
    }
    return {
        "experiment_id": config["experiment_id"],
        "rows": len(rows),
        "scenarios": len(by_scenario),
        "speedup_directional_fraction": speed_fraction,
        "saturation_max_oracle_value": saturation_max_value,
        "message_reversal_fraction": reversal_fraction,
        "best_q_support_by_task": support,
        "message_oracle_to_strong_fixed_geometric_ratio": oracle_ratio,
        "message_oracle_improvement": 1.0 - oracle_ratio,
        "message_strict_improvement_fraction": strict_fraction,
        "gates": gates,
        "pre_reproduction_gates_pass": all(gates.values()),
    }


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    tasks = build_tasks(config)
    static_validate(config, tasks)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows = [
        evaluate_action(
            scenario,
            tasks[scenario["task"]],
            q=q,
            normalized_step=normalized_step,
            overhead=config["grid"]["message_overhead"],
        )
        for scenario in scenario_rows(config)
        for q, normalized_step in product(config["grid"]["q"], config["grid"]["normalized_step"])
    ]
    row_path = output / "rows.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    task_path = output / "task_constants.json"
    task_path.write_bytes(stable_json_bytes({name: public_task_summary(task) for name, task in tasks.items()}))
    summary = analyze(rows, config)
    summary["configuration_sha256"] = sha256_file(DEFAULT_CONFIG)
    summary["runner_sha256"] = sha256_file(Path(__file__))
    summary["rows_sha256"] = sha256_file(row_path)
    summary["task_constants_sha256"] = sha256_file(task_path)
    summary_path = output / "summary.json"
    summary_path.write_bytes(stable_json_bytes(summary))
    summary["summary_sha256"] = sha256_file(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "t043a_standard_task_static",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = load_config(arguments.config)
    tasks = build_tasks(config)
    if arguments.mode == "validate":
        result = static_validate(config, tasks)
    elif arguments.mode == "estimate":
        result = estimate(config, tasks)
    else:
        result = run(config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
