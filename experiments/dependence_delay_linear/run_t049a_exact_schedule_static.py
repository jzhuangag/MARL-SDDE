"""Preregistered exact vector schedule-value scan for T-049A."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    delayed_vector_companion,
)
from experiments.dependence_delay_linear.t048_pr_probe_oracle import (
    scheduled_pr_risk_affine_coefficients,
)
from experiments.dependence_delay_linear.t049_standard_task_exact import (
    build_exact_projected_task,
    exact_gradient_lag_covariances,
    public_exact_task_summary,
    stable_json_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs" / "t049a_exact_schedule_static_preregistration.json"
T048_CORE = Path(__file__).with_name("t048_pr_probe_oracle.py")
T049_CORE = Path(__file__).with_name("t049_standard_task_exact.py")


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
        raise ValueError("T-049A is pinned to Gymnasium 1.0.0")
    return {
        name: build_exact_projected_task(
            environment_id=specification["environment_id"],
            kwargs=specification["kwargs"],
            feature_dimension=specification["feature_dimension"],
            discount=config["software"]["discount"],
            policy_epsilon=config["software"]["policy_epsilon"],
        )
        for name, specification in config["software"]["tasks"].items()
    }


def schedule_specifications(config: dict[str, Any]) -> list[dict[str, Any]]:
    library = config["grid"]["schedule_library"]
    specifications = [
        {
            "name": f"fixed-q{q}",
            "kind": "fixed",
            "q_first": int(q),
            "q_second": int(q),
            "fraction": 1.0,
        }
        for q in library["fixed_q"]
    ]
    for pair, fraction in product(
        library["two_stage_pairs"], library["first_stage_fraction"]
    ):
        first, second = map(int, pair)
        specifications.append(
            {
                "name": f"q{first}-to-q{second}-f{fraction:g}",
                "kind": "two_stage",
                "q_first": first,
                "q_second": second,
                "fraction": float(fraction),
            }
        )
    return specifications


def scenario_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["grid"]
    rows = []
    for task, overhead, reference_updates, delay in product(
        config["software"]["tasks"],
        grid["message_overhead"],
        grid["reference_updates"],
        grid["delay"],
    ):
        message_budget = int(reference_updates * (overhead + grid["reference_q"]))
        for rho in grid["rho"]:
            rows.append(
                {
                    "scenario_id": (
                        f"{task}-H{overhead}-N{reference_updates}-D{delay}-R{rho:g}"
                    ),
                    "base_id": f"{task}-H{overhead}-N{reference_updates}-D{delay}",
                    "task": task,
                    "message_overhead": int(overhead),
                    "reference_updates": int(reference_updates),
                    "message_budget": message_budget,
                    "environment_budget": int(grid["environment_budget"]),
                    "delay": int(delay),
                    "rho": float(rho),
                }
            )
    return rows


def mixing_burn_in(task: dict[str, Any], target: float) -> int:
    rate = float(task["mixing_slem"])
    if not 0.0 < target < 1.0 or not 0.0 <= rate < 1.0:
        raise ValueError("invalid mixing target or rate")
    if rate == 0.0:
        return 1
    return max(1, int(math.ceil(math.log(target) / math.log(rate))))


def make_schedule(
    *,
    specification: dict[str, Any],
    message_budget: int,
    environment_budget: int,
    message_overhead: int,
    delay: int,
) -> tuple[int, ...]:
    """Maximize updates for a registered schedule shape under both budgets."""

    if message_budget <= 0 or environment_budget <= delay:
        return ()
    minimum_q = min(specification["q_first"], specification["q_second"])
    upper = min(
        (message_budget // (message_overhead + minimum_q)),
        environment_budget - delay,
    )

    def candidate(updates: int) -> tuple[int, ...]:
        if specification["kind"] == "fixed":
            return (specification["q_first"],) * updates
        first_count = max(
            1,
            min(
                updates - 1,
                int(math.floor(specification["fraction"] * updates)),
            ),
        )
        return (
            (specification["q_first"],) * first_count
            + (specification["q_second"],) * (updates - first_count)
        )

    best: tuple[int, ...] = ()
    lower = 1
    while lower <= upper:
        middle = (lower + upper) // 2
        schedule = candidate(middle)
        message = sum(message_overhead + q for q in schedule)
        environment = len(schedule) + delay
        if message <= message_budget and environment <= environment_budget:
            best = schedule
            lower = middle + 1
        else:
            upper = middle - 1
    return best


def slice_lag_covariances(
    full_lags: np.ndarray, *, full_horizon: int, horizon: int
) -> np.ndarray:
    if horizon < 1 or horizon > full_horizon:
        raise ValueError("invalid lag-covariance slice")
    center = full_horizon - 1
    return full_lags[center - horizon + 1 : center + horizon]


def static_validate(
    config: dict[str, Any], tasks: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    task_data = build_tasks(config) if tasks is None else tasks
    scenarios = scenario_rows(config)
    schedules = schedule_specifications(config)
    expected = config["expected_workload"]
    base_scenarios = len({row["base_id"] for row in scenarios})
    if len(task_data) != expected["tasks"]:
        raise ValueError("task count mismatch")
    if base_scenarios != expected["base_scenarios"]:
        raise ValueError("base scenario count mismatch")
    if len(scenarios) != expected["rho_cells"]:
        raise ValueError("rho-cell count mismatch")
    if len(schedules) != expected["schedules_per_cell"]:
        raise ValueError("schedule count mismatch")
    if len(scenarios) * len(schedules) != expected["rows"]:
        raise ValueError("row count mismatch")
    if config["sampled_learning_trajectory_authorized"] is not False:
        raise ValueError("sampled trajectories are forbidden")
    if config["formal_or_gpu_authorized"] is not False:
        raise ValueError("formal and GPU execution are forbidden")
    if config["analysis"]["T043A_T045A_results_as_inputs"] is not False:
        raise ValueError("earlier standard-task outcomes are forbidden inputs")
    for task in task_data.values():
        transition = task["continuing_transition"]
        stationary = task["stationary"]
        features = task["features"]
        np.testing.assert_allclose(transition.sum(axis=1), 1.0)
        np.testing.assert_allclose(stationary @ transition, stationary)
        np.testing.assert_allclose(
            features.T @ (stationary[:, None] * features),
            np.eye(features.shape[1]),
            atol=1e-10,
        )
        if task["drift_minimum"] <= 0.0 or not 0.0 <= task["mixing_slem"] < 1.0:
            raise ValueError("invalid exact task constants")
    return {
        "experiment_id": config["experiment_id"],
        "tasks": len(task_data),
        "base_scenarios": base_scenarios,
        "rho_cells": len(scenarios),
        "schedules_per_cell": len(schedules),
        "rows": len(scenarios) * len(schedules),
        "sampled_trajectories": 0,
        "recommended_hardware": "local CPU",
    }


def estimate(
    config: dict[str, Any], tasks: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    task_data = build_tasks(config) if tasks is None else tasks
    validated = static_validate(config, task_data)
    maximum_updates = 0
    specifications = schedule_specifications(config)
    for scenario in scenario_rows(config):
        for specification in specifications:
            schedule = make_schedule(
                specification=specification,
                message_budget=scenario["message_budget"],
                environment_budget=scenario["environment_budget"],
                message_overhead=scenario["message_overhead"],
                delay=scenario["delay"],
            )
            maximum_updates = max(maximum_updates, len(schedule))
    if maximum_updates != config["expected_workload"]["maximum_full_updates"]:
        raise ValueError(
            f"maximum update mismatch: {maximum_updates} != "
            f"{config['expected_workload']['maximum_full_updates']}"
        )
    return {
        **validated,
        "maximum_full_updates": maximum_updates,
        "task_constants": {
            name: public_exact_task_summary(task)
            for name, task in task_data.items()
        },
    }


def _geometric_mean(values: list[float]) -> float:
    return float(math.exp(np.mean(np.log(np.maximum(values, 1e-300)))))


def evaluate_base_scenario(
    base: dict[str, Any],
    task: dict[str, Any],
    config: dict[str, Any],
    specifications: list[dict[str, Any]],
    full_lags: np.ndarray,
    full_horizon: int,
) -> list[dict[str, Any]]:
    probe = config["probe_ceiling"]
    probe_burn_in = mixing_burn_in(
        task, probe["stationarity_total_variation_target"]
    )
    post_message_budget = base["message_budget"] - probe[
        "independent_restart_blocks"
    ] * (base["message_overhead"] + probe["q_probe"])
    post_environment_budget = base["environment_budget"] - (
        probe["independent_restart_blocks"] * probe_burn_in + base["delay"]
    )
    step_size = (
        config["estimator"]["step_multiplier"]
        * (1.0 - task["mixing_slem"])
        / task["drift_norm"]
    )
    spectral_radius = float(
        np.max(
            np.abs(
                np.linalg.eigvals(
                    delayed_vector_companion(
                        task["drift"], step_size, base["delay"]
                    )
                )
            )
        )
    )
    rows = []
    for specification in specifications:
        full_schedule = make_schedule(
            specification=specification,
            message_budget=base["message_budget"],
            environment_budget=base["environment_budget"],
            message_overhead=base["message_overhead"],
            delay=base["delay"],
        )
        post_schedule = make_schedule(
            specification=specification,
            message_budget=post_message_budget,
            environment_budget=post_environment_budget,
            message_overhead=base["message_overhead"],
            delay=base["delay"],
        )
        if not full_schedule or not post_schedule:
            raise RuntimeError("registered full or post-probe schedule is empty")

        def coefficients(schedule: tuple[int, ...]) -> tuple[float, float, float]:
            horizon = len(schedule)
            affine = scheduled_pr_risk_affine_coefficients(
                initial_history=np.tile(
                    -task["theta_star"], (base["delay"] + 1, 1)
                ),
                drift=task["drift"],
                step_size=step_size,
                delay=base["delay"],
                q_schedule=schedule,
                burn_in=horizon // 2,
                base_lag_covariances=slice_lag_covariances(
                    full_lags,
                    full_horizon=full_horizon,
                    horizon=horizon,
                ),
            )
            return affine.intercept, affine.slope, float(
                np.mean(np.asarray(schedule, dtype=float))
            )

        full_intercept, full_slope, full_mean_q = coefficients(full_schedule)
        post_intercept, post_slope, post_mean_q = coefficients(post_schedule)
        for rho in config["grid"]["rho"]:
            rows.append(
                {
                    **base,
                    "rho": float(rho),
                    "scenario_id": f"{base['base_id']}-R{rho:g}",
                    "schedule_name": specification["name"],
                    "schedule_kind": specification["kind"],
                    "full_updates": len(full_schedule),
                    "post_probe_updates": len(post_schedule),
                    "full_message_used": sum(
                        base["message_overhead"] + q for q in full_schedule
                    ),
                    "post_probe_message_used": sum(
                        base["message_overhead"] + q for q in post_schedule
                    ),
                    "full_environment_used": len(full_schedule) + base["delay"],
                    "post_probe_environment_used": len(post_schedule)
                    + base["delay"],
                    "post_probe_message_budget": post_message_budget,
                    "post_probe_environment_budget": post_environment_budget,
                    "full_mean_q": full_mean_q,
                    "post_probe_mean_q": post_mean_q,
                    "step_size": step_size,
                    "spectral_radius": spectral_radius,
                    "probe_burn_in": probe_burn_in,
                    "full_intercept": full_intercept,
                    "full_slope": full_slope,
                    "post_probe_intercept": post_intercept,
                    "post_probe_slope": post_slope,
                    "full_risk": full_intercept + full_slope * rho,
                    "post_probe_risk": post_intercept + post_slope * rho,
                }
            )
    return rows


def analyze(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    cells: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault(row["scenario_id"], []).append(row)
    fixed_names = {
        specification["name"]
        for specification in schedule_specifications(config)
        if specification["kind"] == "fixed"
    }

    fixed_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for candidates in cells.values():
        first = candidates[0]
        key = (
            first["task"],
            first["message_overhead"],
            first["reference_updates"],
            first["delay"],
        )
        fixed_groups.setdefault(key, []).extend(
            row for row in candidates if row["schedule_name"] in fixed_names
        )
    strong_fixed: dict[tuple[Any, ...], str] = {}
    for key, group in fixed_groups.items():
        scores = {
            name: _geometric_mean(
                [row["full_risk"] for row in group if row["schedule_name"] == name]
            )
            for name in fixed_names
        }
        strong_fixed[key] = min(scores, key=lambda name: (scores[name], name))

    comparisons = []
    oracle_support = set()
    adjacent_directions = []
    by_base: dict[str, list[dict[str, Any]]] = {}
    for scenario_id, candidates in cells.items():
        first = candidates[0]
        key = (
            first["task"],
            first["message_overhead"],
            first["reference_updates"],
            first["delay"],
        )
        baseline = next(
            row
            for row in candidates
            if row["schedule_name"] == strong_fixed[key]
        )
        post_oracle = min(
            candidates,
            key=lambda row: (row["post_probe_risk"], row["schedule_name"]),
        )
        full_oracle = min(
            candidates, key=lambda row: (row["full_risk"], row["schedule_name"])
        )
        fixed_oracle = min(
            (row for row in candidates if row["schedule_name"] in fixed_names),
            key=lambda row: (row["full_risk"], row["schedule_name"]),
        )
        nonconstant_oracle = min(
            (row for row in candidates if row["schedule_kind"] == "two_stage"),
            key=lambda row: (row["post_probe_risk"], row["schedule_name"]),
        )
        oracle_support.add(post_oracle["schedule_name"])
        comparison = {
            "base_id": first["base_id"],
            "rho": first["rho"],
            "post_oracle_ratio": post_oracle["post_probe_risk"]
            / baseline["full_risk"],
            "post_oracle_strict": post_oracle["post_probe_risk"]
            < baseline["full_risk"] * (1.0 - 1e-12),
            "nonconstant_ratio": nonconstant_oracle["post_probe_risk"]
            / fixed_oracle["full_risk"],
            "nonconstant_strict": nonconstant_oracle["post_probe_risk"]
            < fixed_oracle["full_risk"] * (1.0 - 1e-12),
            "post_schedule": post_oracle["schedule_name"],
            "post_mean_q": post_oracle["post_probe_mean_q"],
            "full_oracle_ratio": full_oracle["full_risk"] / baseline["full_risk"],
        }
        comparisons.append(comparison)
        by_base.setdefault(first["base_id"], []).append(comparison)
    for group in by_base.values():
        ordered = sorted(group, key=lambda row: row["rho"])
        adjacent_directions.extend(
            right["post_mean_q"] <= left["post_mean_q"] + 1e-12
            for left, right in zip(ordered, ordered[1:])
        )

    post_ratio = _geometric_mean(
        [comparison["post_oracle_ratio"] for comparison in comparisons]
    )
    post_strict = float(
        np.mean([comparison["post_oracle_strict"] for comparison in comparisons])
    )
    nonconstant_ratio = _geometric_mean(
        [comparison["nonconstant_ratio"] for comparison in comparisons]
    )
    nonconstant_strict = float(
        np.mean([comparison["nonconstant_strict"] for comparison in comparisons])
    )
    direction_fraction = float(np.mean(adjacent_directions))
    fixed_support = oracle_support & fixed_names
    dynamic_support = oracle_support - fixed_names
    gates = {
        "V1": True,
        "V2": all(
            np.isfinite(row["full_risk"])
            and np.isfinite(row["post_probe_risk"])
            and row["full_risk"] >= 0.0
            and row["post_probe_risk"] >= 0.0
            and row["full_updates"] >= 1
            and row["post_probe_updates"] >= 1
            and row["spectral_radius"] < 1.0
            and row["full_message_used"] <= row["message_budget"]
            and row["post_probe_message_used"]
            <= row["post_probe_message_budget"]
            and row["full_environment_used"] <= row["environment_budget"]
            and row["post_probe_environment_used"]
            <= row["post_probe_environment_budget"]
            for row in rows
        ),
        "V3": 1.0 - post_ratio >= 0.05,
        "V4": post_strict >= 0.50,
        "V5": 1.0 - nonconstant_ratio >= 0.01,
        "V6": nonconstant_strict >= 0.25,
        "V7": direction_fraction >= 0.80,
        "V8": len(fixed_support) >= 2 and len(dynamic_support) >= 1,
        "V9": True,
    }
    return {
        "experiment_id": config["experiment_id"],
        "rows": len(rows),
        "rho_cells": len(cells),
        "full_cost_oracle_to_strong_fixed_geometric_ratio": post_ratio,
        "full_cost_oracle_improvement": 1.0 - post_ratio,
        "full_cost_oracle_strict_fraction": post_strict,
        "nonconstant_post_probe_to_cellwise_fixed_geometric_ratio": (
            nonconstant_ratio
        ),
        "nonconstant_post_probe_improvement": 1.0 - nonconstant_ratio,
        "nonconstant_post_probe_strict_fraction": nonconstant_strict,
        "rho_direction_fraction": direction_fraction,
        "post_probe_oracle_support": sorted(oracle_support),
        "gates": gates,
        "pre_reproduction_gates_pass": all(gates.values()),
    }


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    tasks = build_tasks(config)
    estimate_result = estimate(config, tasks)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    specifications = schedule_specifications(config)
    full_horizon = estimate_result["maximum_full_updates"]
    task_lags = {
        name: exact_gradient_lag_covariances(task, horizon=full_horizon)
        for name, task in tasks.items()
    }
    bases: dict[str, dict[str, Any]] = {}
    for scenario in scenario_rows(config):
        bases.setdefault(
            scenario["base_id"],
            {key: value for key, value in scenario.items() if key != "rho"},
        )
    rows = []
    for base in bases.values():
        rows.extend(
            evaluate_base_scenario(
                base,
                tasks[base["task"]],
                config,
                specifications,
                task_lags[base["task"]],
                full_horizon,
            )
        )
    row_path = output / "rows.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    task_path = output / "task_constants.json"
    task_path.write_bytes(
        stable_json_bytes(
            {
                name: public_exact_task_summary(task)
                for name, task in tasks.items()
            }
        )
    )
    summary = analyze(rows, config)
    summary.update(
        {
            "configuration_sha256": sha256_file(DEFAULT_CONFIG),
            "runner_sha256": sha256_file(Path(__file__)),
            "t048_core_sha256": sha256_file(T048_CORE),
            "t049_core_sha256": sha256_file(T049_CORE),
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
        default=(
            Path(__file__).resolve().parent
            / "results"
            / "t049a_exact_schedule_static"
        ),
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
