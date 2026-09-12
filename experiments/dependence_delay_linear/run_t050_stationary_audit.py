"""Execute the outcome-free T-050 stationary theorem audit."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.run_t049a_exact_schedule_static import (
    build_tasks,
    load_config as load_t049_config,
)
from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    delayed_vector_companion,
)
from experiments.dependence_delay_linear.t049_standard_task_exact import (
    stable_json_bytes,
)
from experiments.dependence_delay_linear.t050_stationary_break_even import (
    asymptotic_participation_coefficient,
    contraction_burn_in_horizon,
    exact_edge_long_run_covariance,
    optimal_catalogue_q,
    pr_task_constant,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "docs" / "t050_stationary_audit_plan.json"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def geometric_mean(values: np.ndarray) -> float:
    return float(math.exp(float(np.mean(np.log(values)))))


def execute(plan: dict[str, Any]) -> dict[str, Any]:
    if plan["uses_t049a_outcome_rows"] is not False:
        raise ValueError("T-049A outcome rows are forbidden")
    if plan["sampled_trajectories"] != 0:
        raise ValueError("sampled trajectories are forbidden")
    if plan["gpu_authorized"] or plan["hpc4_authorized"]:
        raise ValueError("GPU and HPC4 are forbidden")
    t049_config = load_t049_config()
    tasks = build_tasks(t049_config)
    if sorted(tasks) != sorted(plan["tasks"]):
        raise ValueError("task list differs from the frozen plan")

    task_results: dict[str, Any] = {}
    for name, task in tasks.items():
        long_run = exact_edge_long_run_covariance(
            transition=task["continuing_transition"],
            stationary=task["stationary"],
            edge_gradient_sum=task["edge_gradient_sum"],
            conditional_gradient=task["conditional_gradient"],
            second_moment=task["gradient_second_moment"],
        )
        step_size = (
            t049_config["estimator"]["step_multiplier"]
            * (1.0 - task["mixing_slem"])
            / task["drift_norm"]
        )
        delay_rows: dict[str, Any] = {}
        for delay in plan["delays"]:
            radius = float(
                np.max(
                    np.abs(
                        np.linalg.eigvals(
                            delayed_vector_companion(
                                task["drift"], step_size, int(delay)
                            )
                        )
                    )
                )
            )
            delay_rows[str(delay)] = {
                "spectral_radius": radius,
                "contraction_horizons": {
                    f"{target:g}": contraction_burn_in_horizon(
                        spectral_radius=radius,
                        target=float(target),
                        averaging_fraction=plan["pr_burn_fraction"],
                    )
                    for target in plan["contraction_targets"]
                },
            }
        task_results[name] = {
            "kernel_sha256": task["kernel_sha256"],
            "step_size": step_size,
            "long_run_covariance_minimum_eigenvalue": float(
                np.min(np.linalg.eigvalsh(long_run))
            ),
            "long_run_covariance_trace": float(np.trace(long_run)),
            "pr_task_constant": pr_task_constant(
                drift=task["drift"], long_run_covariance=long_run
            ),
            "delays": delay_rows,
        }

    phase_results = []
    candidates = [int(q) for q in plan["participation_catalogue"]]
    correlations = [float(rho) for rho in plan["correlations"]]
    for overhead in plan["message_overheads"]:
        coefficient_table = np.asarray(
            [
                [
                    asymptotic_participation_coefficient(
                        q, overhead=float(overhead), rho=rho
                    )
                    for q in candidates
                ]
                for rho in correlations
            ]
        )
        fixed_geometric = np.asarray(
            [geometric_mean(coefficient_table[:, index]) for index in range(len(candidates))]
        )
        baseline_index = int(np.argmin(fixed_geometric))
        baseline = coefficient_table[:, baseline_index]
        oracle = np.min(coefficient_table, axis=1)
        phase_results.append(
            {
                "message_overhead": int(overhead),
                "strong_fixed_q": candidates[baseline_index],
                "oracle_geometric_improvement": 1.0
                - geometric_mean(oracle / baseline),
                "strict_improvement_fraction": float(
                    np.mean(oracle < baseline - 1e-12)
                ),
                "oracle_support": [
                    int(
                        optimal_catalogue_q(
                            candidates, overhead=float(overhead), rho=rho
                        )["q"]
                    )
                    for rho in correlations
                ],
            }
        )
    return {
        "audit_id": plan["audit_id"],
        "plan_sha256": sha256_file(DEFAULT_PLAN),
        "sampled_trajectories": 0,
        "task_results": task_results,
        "stationary_phase": phase_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    plan = load_plan(arguments.plan)
    result = execute(plan)
    data = stable_json_bytes(result)
    if arguments.output is None:
        print(data.decode("utf-8"), end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(data)


if __name__ == "__main__":
    main()
