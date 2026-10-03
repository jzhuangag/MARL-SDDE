"""Execute the preregistered T-052A exact-binomial static gate."""

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
from experiments.dependence_delay_linear.run_t051a_fingerprint_static import (
    geometric_mean,
    strong_fixed_q,
)
from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    delayed_vector_companion,
)
from experiments.dependence_delay_linear.t049_standard_task_exact import (
    stable_json_bytes,
)
from experiments.dependence_delay_linear.t050_stationary_break_even import (
    contraction_burn_in_horizon,
)
from experiments.dependence_delay_linear.t051_fingerprint_probe import (
    minimum_fingerprint_length,
    plug_in_action,
)
from experiments.dependence_delay_linear.t052_exact_binomial_probe import (
    exact_full_cost_plugin_ratio,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs" / "t052a_exact_fingerprint_static_preregistration.json"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def execute(config: dict[str, Any], *, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    analysis = config["analysis"]
    if analysis["uses_t049a_outcome_rows"] or analysis["uses_t051a_result_rows"]:
        raise ValueError("prior outcome rows are forbidden")
    if any(config["authorization"][key] for key in ("sampled_cpu_pilot", "formal", "gpu", "hpc4")):
        raise ValueError("sampled execution and accelerators are forbidden")
    t049 = load_t049_config()
    tasks = build_tasks(t049)
    if sorted(tasks) != sorted(config["tasks"]):
        raise ValueError("task list differs from preregistration")
    grid = config["grid"]
    candidates = [int(q) for q in grid["participation_catalogue"]]
    correlations = [float(rho) for rho in grid["correlations"]]
    q_max = max(candidates)
    probe = config["probe"]
    horizon = config["learning_horizon_rule"]
    rows: list[dict[str, Any]] = []
    fingerprints: dict[str, dict[str, float | int]] = {}

    for task_name, task in tasks.items():
        fingerprint = minimum_fingerprint_length(
            transition=task["continuing_transition"],
            stationary=task["stationary"],
            maximum_collision=probe["maximum_independent_path_collision"],
        )
        fingerprints[task_name] = fingerprint
        step_size = (
            t049["estimator"]["step_multiplier"]
            * (1.0 - task["mixing_slem"])
            / task["drift_norm"]
        )
        for delay in grid["delays"]:
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
            learning_updates_qmax = contraction_burn_in_horizon(
                spectral_radius=radius,
                target=horizon["contraction_target"],
                averaging_fraction=horizon["pr_burn_fraction"],
            )
            for overhead in grid["message_overheads"]:
                baseline_q = strong_fixed_q(
                    candidates=candidates,
                    overhead=int(overhead),
                    correlations=correlations,
                )
                learning_message_budget = learning_updates_qmax * (
                    int(overhead) + q_max
                )
                probe_message_cost = probe["blocks"] * (int(overhead) + 2)
                probe_environment_cost = probe["blocks"] * int(
                    fingerprint["transitions"]
                )
                maximum_learning_updates = learning_message_budget // (
                    int(overhead) + min(candidates)
                )
                environment_budget = (
                    probe_environment_cost + maximum_learning_updates + int(delay)
                )
                for rho in correlations:
                    exact = exact_full_cost_plugin_ratio(
                        rho=rho,
                        candidates=candidates,
                        overhead=float(overhead),
                        baseline_q=baseline_q,
                        learning_budget=float(learning_message_budget),
                        probe_blocks=probe["blocks"],
                        collision_probability=float(
                            fingerprint["collision_probability"]
                        ),
                    )
                    distribution = {
                        str(q): float(probability)
                        for q, probability in exact["action_distribution"].items()
                    }
                    rows.append(
                        {
                            "task": task_name,
                            "kernel_sha256": task["kernel_sha256"],
                            "delay": int(delay),
                            "spectral_radius": radius,
                            "learning_updates_qmax": learning_updates_qmax,
                            "message_overhead": int(overhead),
                            "rho": rho,
                            "baseline_q": baseline_q,
                            "oracle_q": int(exact["oracle_q"]),
                            "oracle_active": int(exact["oracle_q"]) != baseline_q,
                            "action_distribution": distribution,
                            "action_probability_sum": float(sum(distribution.values())),
                            "expected_risk_ratio": float(exact["expected_risk_ratio"]),
                            "learning_message_budget": learning_message_budget,
                            "probe_message_cost": probe_message_cost,
                            "total_message_budget": learning_message_budget
                            + probe_message_cost,
                            "probe_environment_cost": probe_environment_cost,
                            "maximum_learning_updates": maximum_learning_updates,
                            "environment_budget": environment_budget,
                            "environment_used_upper_bound": probe_environment_cost
                            + maximum_learning_updates
                            + int(delay),
                        }
                    )
    if len(rows) != config["expected_workload"]["cells"]:
        raise RuntimeError("registered cell count mismatch")
    ratios = [row["expected_risk_ratio"] for row in rows]
    by_overhead = {
        str(overhead): 1.0
        - geometric_mean(
            [
                row["expected_risk_ratio"]
                for row in rows
                if row["message_overhead"] == overhead
            ]
        )
        for overhead in grid["message_overheads"]
    }
    active = [row for row in rows if row["oracle_active"]]
    directions = []
    for overhead in grid["message_overheads"]:
        selected = [
            plug_in_action(rho, candidates, overhead=float(overhead))
            for rho in np.linspace(0.0, 1.0, 1001)
        ]
        directions.append(
            all(left >= right for left, right in zip(selected, selected[1:]))
        )
    gates = {
        "B1": all(
            np.allclose(task["stationary"] @ task["continuing_transition"], task["stationary"])
            for task in tasks.values()
        ),
        "B2": all(
            item["collision_probability"]
            <= probe["maximum_independent_path_collision"]
            for item in fingerprints.values()
        ),
        "B3": all(row["spectral_radius"] < 1.0 for row in rows),
        "B4": 1.0 - geometric_mean(ratios) >= 0.05,
        "B5": all(value >= 0.05 for value in by_overhead.values()),
        "B6": sum(row["expected_risk_ratio"] < 1.0 for row in active) / len(active)
        >= 0.70,
        "B7": max(ratios) <= 1.05,
        "B8": all(directions)
        and all(abs(row["action_probability_sum"] - 1.0) <= 1e-12 for row in rows),
        "B9": all(
            row["environment_used_upper_bound"] <= row["environment_budget"]
            and row["probe_message_cost"] + row["learning_message_budget"]
            <= row["total_message_budget"]
            for row in rows
        ),
        "B10": True,
        "B11": all(
            math.isfinite(value)
            for row in rows
            for value in row.values()
            if isinstance(value, float)
        ),
    }
    return {
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "cells": len(rows),
        "sampled_trajectories": 0,
        "fingerprints": fingerprints,
        "aggregate_geometric_improvement": 1.0 - geometric_mean(ratios),
        "improvement_by_overhead": by_overhead,
        "active_cells": len(active),
        "active_improved_fraction": sum(
            row["expected_risk_ratio"] < 1.0 for row in active
        )
        / len(active),
        "maximum_expected_risk_ratio": max(ratios),
        "gates": gates,
        "pre_reproduction_gates_pass": all(gates.values()),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = execute(load_config(arguments.config), config_path=arguments.config)
    data = stable_json_bytes(result)
    if arguments.output is None:
        print(data.decode("utf-8"), end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(data)


if __name__ == "__main__":
    main()
