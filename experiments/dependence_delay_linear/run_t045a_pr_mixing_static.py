"""Final outcome-free PR/mixing standard-task feasibility scan."""

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

from experiments.dependence_delay_linear.run_t043a_standard_task_static import (
    build_tasks,
    scenario_rows,
)
from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    dual_budget_updates,
)
from experiments.dependence_delay_linear.t043a_standard_task_static import (
    public_task_summary,
    stable_json_bytes,
)
from experiments.dependence_delay_linear.t044_pr_averaged_phase import (
    exact_pr_averaged_scalar_risk,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs" / "t045a_pr_mixing_static_preregistration.json"
CORE_SOURCE = Path(__file__).with_name("t044_pr_averaged_phase.py")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def t043_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "software": config["software"],
        "grid": {
            **config["grid"],
            "normalized_step": [0.04, 0.08, 0.12],
        },
        "analysis": {"no_seeds_or_confidence_intervals": True},
        "sampled_learning_trajectory_authorized": False,
        "formal_or_gpu_authorized": False,
        "expected_workload": {
            "scenarios": 144,
            "actions_per_scenario": 9,
            "rows": 1296,
            "maximum_scalar_horizon": 256,
        },
        "experiment_id": config["experiment_id"],
    }


def build_registered_tasks(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return build_tasks(t043_compatible_config(config))


def registered_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    return scenario_rows(t043_compatible_config(config))


def static_validate(
    config: dict[str, Any], tasks: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    task_data = build_registered_tasks(config) if tasks is None else tasks
    scenarios = registered_scenarios(config)
    actions = len(config["grid"]["q"]) * len(config["estimator"]["mixing_step_multiplier"])
    expected = config["expected_workload"]
    if len(scenarios) != expected["scenarios"] or actions != expected["actions_per_scenario"]:
        raise ValueError("frozen T-045A workload mismatch")
    if len(scenarios) * actions != expected["rows"]:
        raise ValueError("frozen row count mismatch")
    if config["analysis"]["T043A_results_as_inputs"] is not False:
        raise ValueError("T-043A outcomes are forbidden inputs")
    if config["sampled_learning_trajectory_authorized"] is not False:
        raise ValueError("sampled trajectories are forbidden")
    if config["formal_or_gpu_authorized"] is not False:
        raise ValueError("formal and GPU execution are forbidden")
    if config["estimator"]["burn_in_fraction"] != 0.5:
        raise ValueError("T-045A freezes half-horizon tail averaging")
    for task in task_data.values():
        if task["gymnasium_version"] != config["software"]["gymnasium"]:
            raise ValueError("Gymnasium version mismatch")
        if task["drift_minimum"] <= 0.0 or not 0.0 <= task["mixing_slem"] < 1.0:
            raise ValueError("invalid registered task constants")
    return {
        "experiment_id": config["experiment_id"],
        "tasks": len(task_data),
        "scenarios": len(scenarios),
        "actions_per_scenario": actions,
        "rows": len(scenarios) * actions,
        "sampled_trajectories": 0,
        "recommended_hardware": "local CPU",
    }


def estimate(config: dict[str, Any], tasks: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    validated = static_validate(config, tasks)
    maximum_horizon = 0
    for scenario in registered_scenarios(config):
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
        raise ValueError("maximum horizon mismatch")
    return {**validated, "maximum_scalar_horizon": maximum_horizon}


@lru_cache(maxsize=None)
def unit_pr_components(
    initial_error: float,
    drift_minimum: float,
    step_size: float,
    delay: int,
    updates: int,
    burn_in: int,
    mixing_slem: float,
) -> tuple[float, float, float]:
    bias = exact_pr_averaged_scalar_risk(
        initial_error=initial_error,
        mu=drift_minimum,
        step_size=step_size,
        delay=delay,
        updates=updates,
        burn_in=burn_in,
        single_variance=0.0,
        q=1,
        rho=0.0,
        markov_lambda=mixing_slem,
    )
    unit = exact_pr_averaged_scalar_risk(
        initial_error=0.0,
        mu=drift_minimum,
        step_size=step_size,
        delay=delay,
        updates=updates,
        burn_in=burn_in,
        single_variance=1.0,
        q=1,
        rho=0.0,
        markov_lambda=mixing_slem,
    )
    return bias["risk"], unit["risk"], bias["spectral_radius"]


def evaluate_action(
    scenario: dict[str, Any],
    task: dict[str, Any],
    *,
    q: int,
    multiplier: float,
    overhead: int,
) -> dict[str, Any]:
    updates = dual_budget_updates(
        message_budget=scenario["message_budget"],
        environment_budget=scenario["environment_budget"],
        message_cost=overhead + q,
        stride=1,
        delay=scenario["delay"],
    )
    burn_in = updates // 2
    step_size = multiplier * (1.0 - task["mixing_slem"]) / task["drift_norm"]
    bias, unit_noise, spectral_radius = unit_pr_components(
        task["initial_error"],
        task["drift_minimum"],
        step_size,
        scenario["delay"],
        updates,
        burn_in,
        task["mixing_slem"],
    )
    factor = scenario["rho"] + (1.0 - scenario["rho"]) / q
    noise = unit_noise * task["single_agent_noise_second"] * factor
    return {
        **scenario,
        "q": int(q),
        "mixing_step_multiplier": float(multiplier),
        "step_size": float(step_size),
        "updates": int(updates),
        "burn_in": int(burn_in),
        "averaged_count": int(updates - burn_in),
        "spectral_radius": float(spectral_radius),
        "bias_risk": float(bias),
        "noise_risk": float(noise),
        "risk": float(bias + noise),
    }


def _geometric_mean(values: list[float]) -> float:
    return float(math.exp(np.mean(np.log(np.maximum(values, 1e-300)))))


def analyze(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario_id"], []).append(row)
    best_q_rows = []
    for candidates in by_scenario.values():
        per_q = []
        for q in config["grid"]["q"]:
            q_rows = [row for row in candidates if row["q"] == q]
            per_q.append(
                min(q_rows, key=lambda row: (row["risk"], row["mixing_step_multiplier"]))
            )
        oracle = min(
            per_q,
            key=lambda row: (row["risk"], row["q"], row["mixing_step_multiplier"]),
        )
        enriched = dict(oracle)
        enriched["per_q"] = {int(row["q"]): row for row in per_q}
        best_q_rows.append(enriched)

    speed = [row for row in best_q_rows if row["ray"] == "message" and row["rho"] == 0.0]
    speed_fraction = float(
        np.mean([row["per_q"][16]["risk"] / row["per_q"][1]["risk"] <= 0.95 for row in speed])
    )
    reversal = [row for row in best_q_rows if row["ray"] == "message" and row["rho"] == 1.0]
    reversal_fraction = float(np.mean([row["q"] == 1 for row in reversal]))
    saturation = [row for row in best_q_rows if row["ray"] == "environment" and row["rho"] == 1.0]
    saturation_max_value = float(
        max(1.0 - row["risk"] / row["per_q"][1]["risk"] for row in saturation)
    )
    support = {
        task: sorted({int(row["q"]) for row in best_q_rows if row["task"] == task})
        for task in config["software"]["tasks"]
    }

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in best_q_rows:
        key = (row["task"], row["ray"], row["budget"], row["delay"])
        groups.setdefault(key, []).append(row)
    ratios = []
    for group in groups.values():
        q_scores = {
            q: _geometric_mean([row["per_q"][q]["risk"] for row in group])
            for q in config["grid"]["q"]
        }
        fixed_q = min(q_scores, key=lambda q: (q_scores[q], q))
        for row in group:
            if row["ray"] == "message":
                fixed_risk = row["per_q"][fixed_q]["risk"]
                ratios.append(row["risk"] / fixed_risk)
    oracle_ratio = _geometric_mean(ratios)
    strict_fraction = float(np.mean(np.asarray(ratios) < 1.0 - 1e-12))
    gates = {
        "M1": True,
        "M2": all(
            np.isfinite(row["risk"])
            and row["risk"] >= 0.0
            and row["spectral_radius"] < 1.0
            and row["averaged_count"] >= 1
            for row in rows
        ),
        "M3": speed_fraction >= 0.80,
        "M4": reversal_fraction >= 0.90,
        "M5": saturation_max_value <= 0.01 + 1e-12,
        "M6": all(1 in values and 16 in values for values in support.values()),
        "M7": 1.0 - oracle_ratio >= 0.05,
        "M8": strict_fraction >= 0.40,
        "M9": True,
    }
    return {
        "experiment_id": config["experiment_id"],
        "rows": len(rows),
        "scenarios": len(by_scenario),
        "message_speedup_fraction": speed_fraction,
        "message_reversal_fraction": reversal_fraction,
        "environment_saturation_max_oracle_value": saturation_max_value,
        "best_q_support_by_task": support,
        "message_oracle_to_strong_fixed_geometric_ratio": oracle_ratio,
        "message_oracle_improvement": 1.0 - oracle_ratio,
        "message_strict_improvement_fraction": strict_fraction,
        "gates": gates,
        "pre_reproduction_gates_pass": all(gates.values()),
    }


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    tasks = build_registered_tasks(config)
    static_validate(config, tasks)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows = [
        evaluate_action(
            scenario,
            tasks[scenario["task"]],
            q=q,
            multiplier=multiplier,
            overhead=config["grid"]["message_overhead"],
        )
        for scenario in registered_scenarios(config)
        for q, multiplier in product(
            config["grid"]["q"], config["estimator"]["mixing_step_multiplier"]
        )
    ]
    row_path = output / "rows.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    task_path = output / "task_constants.json"
    task_path.write_bytes(
        stable_json_bytes({name: public_task_summary(task) for name, task in tasks.items()})
    )
    summary = analyze(rows, config)
    summary.update(
        {
            "configuration_sha256": sha256_file(DEFAULT_CONFIG),
            "runner_sha256": sha256_file(Path(__file__)),
            "core_sha256": sha256_file(CORE_SOURCE),
            "rows_sha256": sha256_file(row_path),
            "task_constants_sha256": sha256_file(task_path),
        }
    )
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
        default=Path(__file__).resolve().parent / "results" / "t045a_pr_mixing_static",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = load_config(arguments.config)
    tasks = build_registered_tasks(config)
    if arguments.mode == "validate":
        result = static_validate(config, tasks)
    elif arguments.mode == "estimate":
        result = estimate(config, tasks)
    else:
        result = run(config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
