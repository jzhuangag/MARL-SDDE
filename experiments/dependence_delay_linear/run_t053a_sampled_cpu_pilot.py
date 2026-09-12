"""Run and analyze the preregistered T-053A sampled CPU pilot."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.run_t049a_exact_schedule_static import (
    build_tasks,
    load_config as load_t049_config,
)
from experiments.dependence_delay_linear.run_t051a_fingerprint_static import (
    geometric_mean,
)
from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    delayed_vector_companion,
)
from experiments.dependence_delay_linear.t050_stationary_break_even import (
    contraction_burn_in_horizon,
    optimal_catalogue_q,
)
from experiments.dependence_delay_linear.t051_fingerprint_probe import (
    minimum_fingerprint_length,
    plug_in_action,
)
from experiments.dependence_delay_linear.t053_sampled_standard_task import (
    build_task_sampling_table,
    delayed_pr_risk,
    prefix_aggregate_innovations,
    sample_fingerprint_matches,
    sample_gradient_paths,
    verify_sampling_table,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs" / "t053a_sampled_cpu_pilot_preregistration.json"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "dependence_delay_linear"
    / "results"
    / "t053a_sampled_cpu_pilot"
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_seed(master: int, *labels: object) -> int:
    payload = "|".join([str(master), *(str(label) for label in labels)])
    value = int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "little")
    return int(value % (2**31 - 2) + 1)


def scenario_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in config["tasks"]:
        for delay in config["grid"]["delays"]:
            for overhead in config["grid"]["message_overheads"]:
                for rho in config["grid"]["correlations"]:
                    rows.append(
                        {
                            "task": task,
                            "delay": int(delay),
                            "message_overhead": int(overhead),
                            "rho": float(rho),
                            "cell_id": f"{task}-D{delay}-H{overhead}-R{rho:g}",
                        }
                    )
    return rows


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    if config["analysis"]["uses_prior_outcome_rows"] is not False:
        raise ValueError("prior outcome rows are forbidden")
    t049 = load_t049_config()
    tasks = build_tasks(t049)
    if sorted(tasks) != sorted(config["tasks"]):
        raise ValueError("task registry mismatch")
    prepared = {}
    for name, task in tasks.items():
        if task["kernel_sha256"] != config["kernel_sha256"][name]:
            raise ValueError("kernel hash mismatch")
        table = build_task_sampling_table(
            task, discount=t049["software"]["discount"]
        )
        verification = verify_sampling_table(task, table)
        if verification["transition_max_error"] > 1e-14:
            raise ValueError("sampling transition residual exceeds tolerance")
        if verification["mean_norm"] > 1e-8:
            raise ValueError("sampling innovation mean exceeds tolerance")
        if verification["second_moment_max_error"] > 1e-10:
            raise ValueError("sampling second moment residual exceeds tolerance")
        fingerprint = minimum_fingerprint_length(
            transition=task["continuing_transition"],
            stationary=task["stationary"],
            maximum_collision=config["probe"]["maximum_independent_path_collision"],
        )
        step_size = (
            t049["estimator"]["step_multiplier"]
            * (1.0 - task["mixing_slem"])
            / task["drift_norm"]
        )
        delays = {}
        for delay in config["grid"]["delays"]:
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
            delays[str(delay)] = {
                "spectral_radius": radius,
                "learning_updates_qmax": contraction_burn_in_horizon(
                    spectral_radius=radius,
                    target=config["learning"]["contraction_target"],
                    averaging_fraction=config["learning"]["pr_burn_fraction"],
                ),
            }
        prepared[name] = {
            "task": task,
            "table": table,
            "verification": verification,
            "fingerprint": fingerprint,
            "step_size": step_size,
            "delays": delays,
        }
    return prepared


def validate(config: dict[str, Any]) -> dict[str, Any]:
    prepared = prepare(config)
    scenarios = scenario_rows(config)
    seeds = config["pilot_seeds"]
    if len(seeds) != len(set(seeds)):
        raise ValueError("pilot seeds must be unique")
    if len(scenarios) != config["expected_workload"]["cells"]:
        raise ValueError("cell count mismatch")
    if len(scenarios) * len(seeds) != config["expected_workload"]["endpoints"]:
        raise ValueError("endpoint count mismatch")
    return {
        "experiment_id": config["experiment_id"],
        "tasks": len(prepared),
        "cells": len(scenarios),
        "seeds": len(seeds),
        "endpoints": len(scenarios) * len(seeds),
        "recommended_hardware": "local CPU",
        "sampling_verification": {
            name: value["verification"] for name, value in prepared.items()
        },
    }


def estimate(config: dict[str, Any]) -> dict[str, Any]:
    validated = validate(config)
    prepared = prepare(config)
    maximum_horizon = 0
    total_long_paths = 0
    for scenario in scenario_rows(config):
        item = prepared[scenario["task"]]
        updates = item["delays"][str(scenario["delay"])]["learning_updates_qmax"]
        overhead = scenario["message_overhead"]
        learning_budget = updates * (overhead + 16)
        total_budget = learning_budget + config["probe"]["blocks"] * (overhead + 2)
        maximum_horizon = max(maximum_horizon, total_budget // (overhead + 1))
        total_long_paths += 17 * len(config["pilot_seeds"])
    return {
        **validated,
        "maximum_learning_horizon": maximum_horizon,
        "long_learning_trajectories": total_long_paths,
        "stored_full_trajectories": 0,
    }


def run_endpoint(
    *,
    config: dict[str, Any],
    prepared: dict[str, Any],
    scenario: dict[str, Any],
    master_seed: int,
) -> dict[str, Any]:
    name = scenario["task"]
    item = prepared[name]
    task = item["task"]
    overhead = scenario["message_overhead"]
    delay = scenario["delay"]
    rho = scenario["rho"]
    candidates = tuple(config["grid"]["participation_catalogue"])
    probe_blocks = config["probe"]["blocks"]
    fingerprint = item["fingerprint"]
    matches = sample_fingerprint_matches(
        transition=task["continuing_transition"],
        stationary=task["stationary"],
        transitions=int(fingerprint["transitions"]),
        blocks=probe_blocks,
        rho=rho,
        seed=stable_seed(master_seed, scenario["cell_id"], "probe"),
    )
    match_count = int(np.sum(matches))
    collision = float(fingerprint["collision_probability"])
    raw_estimate = (match_count / probe_blocks - collision) / (1.0 - collision)
    rho_hat = float(np.clip(raw_estimate, 0.0, 1.0))
    selected_q = plug_in_action(rho_hat, candidates, overhead=float(overhead))
    baseline_q = int(config["comparators"]["strong_fixed_q"][f"overhead_{overhead}"])
    oracle_q = int(
        optimal_catalogue_q(candidates, overhead=float(overhead), rho=rho)["q"]
    )
    updates_qmax = item["delays"][str(delay)]["learning_updates_qmax"]
    learning_budget = updates_qmax * (overhead + max(candidates))
    probe_message_cost = probe_blocks * (overhead + 2)
    total_budget = learning_budget + probe_message_cost
    controller_updates = learning_budget // (overhead + selected_q)
    fixed_updates = {q: total_budget // (overhead + q) for q in candidates}
    maximum_horizon = max(controller_updates, *fixed_updates.values())
    bank = sample_gradient_paths(
        item["table"],
        paths=max(candidates) + 1,
        horizon=maximum_horizon,
        seed=stable_seed(master_seed, scenario["cell_id"], "learning-paths"),
    )
    aggregates = prefix_aggregate_innovations(
        bank,
        rho=rho,
        candidates=candidates,
        seed=stable_seed(master_seed, scenario["cell_id"], "switches"),
    )
    initial_error = -np.asarray(task["theta_star"], dtype=float)

    def risk(q: int, updates: int) -> float:
        return delayed_pr_risk(
            aggregates[q][:updates],
            drift=task["drift"],
            step_size=item["step_size"],
            delay=delay,
            initial_error=initial_error,
        )

    fixed_risks = {q: risk(q, fixed_updates[q]) for q in candidates}
    controller_risk = risk(selected_q, controller_updates)
    probe_environment = probe_blocks * int(fingerprint["transitions"])
    maximum_learning_updates = total_budget // (overhead + min(candidates))
    environment_budget = probe_environment + maximum_learning_updates + delay
    expected_match = collision + (1.0 - collision) * rho
    variance = max(probe_blocks * expected_match * (1.0 - expected_match), 1e-12)
    return {
        **scenario,
        "seed": int(master_seed),
        "kernel_sha256": task["kernel_sha256"],
        "fingerprint_transitions": int(fingerprint["transitions"]),
        "collision_probability": collision,
        "probe_blocks": probe_blocks,
        "match_count": match_count,
        "expected_match_probability": expected_match,
        "standardized_match_residual": (
            match_count - probe_blocks * expected_match
        )
        / math.sqrt(variance),
        "rho_hat": rho_hat,
        "selected_q": selected_q,
        "baseline_q": baseline_q,
        "oracle_q": oracle_q,
        "learning_updates_qmax": updates_qmax,
        "controller_updates": controller_updates,
        "fixed_q1_updates": fixed_updates[1],
        "fixed_q4_updates": fixed_updates[4],
        "fixed_q16_updates": fixed_updates[16],
        "learning_message_budget": learning_budget,
        "probe_message_cost": probe_message_cost,
        "total_message_budget": total_budget,
        "probe_environment_cost": probe_environment,
        "environment_budget": environment_budget,
        "environment_used_upper_bound": probe_environment
        + maximum_learning_updates
        + delay,
        "controller_risk": controller_risk,
        "strong_fixed_risk": fixed_risks[baseline_q],
        "true_rho_oracle_risk": fixed_risks[oracle_q],
        "sample_path_oracle_risk": min(fixed_risks.values()),
        "fixed_q1_risk": fixed_risks[1],
        "fixed_q4_risk": fixed_risks[4],
        "fixed_q16_risk": fixed_risks[16],
    }


def analyze(config: dict[str, Any], endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    expected = config["expected_workload"]
    if len(endpoints) != expected["endpoints"]:
        raise ValueError("endpoint count mismatch")
    cells: dict[str, list[dict[str, Any]]] = {}
    for row in endpoints:
        cells.setdefault(row["cell_id"], []).append(row)
    if len(cells) != expected["cells"] or any(
        len(rows) != len(config["pilot_seeds"]) for rows in cells.values()
    ):
        raise ValueError("cell or seed coverage mismatch")
    cell_rows = []
    for cell_id, rows in sorted(cells.items()):
        controller = float(np.mean([row["controller_risk"] for row in rows]))
        strong = float(np.mean([row["strong_fixed_risk"] for row in rows]))
        oracle = float(np.mean([row["true_rho_oracle_risk"] for row in rows]))
        first = rows[0]
        cell_rows.append(
            {
                "cell_id": cell_id,
                "task": first["task"],
                "delay": first["delay"],
                "message_overhead": first["message_overhead"],
                "rho": first["rho"],
                "oracle_active": first["oracle_q"] != first["baseline_q"],
                "controller_mean_risk": controller,
                "strong_fixed_mean_risk": strong,
                "true_rho_oracle_mean_risk": oracle,
                "controller_to_strong_ratio": controller / strong,
                "controller_to_oracle_ratio": controller / oracle,
                "median_selected_q": float(np.median([row["selected_q"] for row in rows])),
            }
        )
    ratios = [row["controller_to_strong_ratio"] for row in cell_rows]
    active = [row for row in cell_rows if row["oracle_active"]]
    inactive = [row for row in cell_rows if not row["oracle_active"]]

    def grouped_ratio(field: str) -> dict[str, float]:
        values = {}
        for level in sorted({row[field] for row in cell_rows}):
            values[str(level)] = geometric_mean(
                [
                    row["controller_to_strong_ratio"]
                    for row in cell_rows
                    if row[field] == level
                ]
            )
        return values

    direction = True
    for overhead in config["grid"]["message_overheads"]:
        medians = []
        for rho in config["grid"]["correlations"]:
            medians.append(
                float(
                    np.median(
                        [
                            row["selected_q"]
                            for row in endpoints
                            if row["message_overhead"] == overhead
                            and row["rho"] == rho
                        ]
                    )
                )
            )
        direction &= all(
            left >= right for left, right in zip(medians, medians[1:])
        )
    task_ratios = grouped_ratio("task")
    delay_ratios = grouped_ratio("delay")
    match_rmse = float(
        np.sqrt(np.mean([row["standardized_match_residual"] ** 2 for row in endpoints]))
    )
    gates = {
        "P1": True,
        "P2": all(
            row["environment_used_upper_bound"] <= row["environment_budget"]
            and row["probe_message_cost"] + row["learning_message_budget"]
            == row["total_message_budget"]
            and all(
                math.isfinite(row[key])
                for key in (
                    "controller_risk",
                    "strong_fixed_risk",
                    "true_rho_oracle_risk",
                    "sample_path_oracle_risk",
                )
            )
            for row in endpoints
        ),
        "P3": geometric_mean(ratios) <= 0.95,
        "P4": all(value <= 0.97 for value in task_ratios.values()),
        "P5": all(value <= 0.97 for value in delay_ratios.values()),
        "P6": sum(row["controller_to_strong_ratio"] < 1.0 for row in active)
        / len(active)
        >= 0.60,
        "P7": geometric_mean(
            [row["controller_to_strong_ratio"] for row in inactive]
        )
        <= 1.05,
        "P8": geometric_mean(
            [row["controller_to_oracle_ratio"] for row in cell_rows]
        )
        <= 1.20,
        "P9": bool(direction),
        "P10": match_rmse <= 1.5,
        "P11": len(endpoints) == 672 and len(cells) == 84,
    }
    return {
        "experiment_id": config["experiment_id"],
        "endpoints": len(endpoints),
        "cells": len(cell_rows),
        "aggregate_controller_to_strong_ratio": geometric_mean(ratios),
        "aggregate_controller_to_oracle_ratio": geometric_mean(
            [row["controller_to_oracle_ratio"] for row in cell_rows]
        ),
        "task_ratios": task_ratios,
        "delay_ratios": delay_ratios,
        "active_cells": len(active),
        "active_improved_fraction": sum(
            row["controller_to_strong_ratio"] < 1.0 for row in active
        )
        / len(active),
        "inactive_cells": len(inactive),
        "inactive_geometric_ratio": geometric_mean(
            [row["controller_to_strong_ratio"] for row in inactive]
        ),
        "fingerprint_standardized_rmse": match_rmse,
        "gates": gates,
        "pre_reproduction_gates_pass": all(gates.values()),
        "cell_rows": cell_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    prepared = prepare(config)
    started = time.perf_counter()
    endpoints = []
    for scenario in scenario_rows(config):
        for seed in config["pilot_seeds"]:
            endpoints.append(
                run_endpoint(
                    config=config,
                    prepared=prepared,
                    scenario=scenario,
                    master_seed=int(seed),
                )
            )
    summary = analyze(config, endpoints)
    runtime_seconds = time.perf_counter() - started
    cell_rows = summary.pop("cell_rows")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cell_rows)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {**summary, "runtime_seconds": runtime_seconds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.mode == "validate":
        result = validate(config)
    elif arguments.mode == "estimate":
        result = estimate(config)
    else:
        result = run(config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
