"""Run the preregistered T-065A discrete joint-control CPU mechanism pilot."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.t064_joint_clf_optimizer import (
    JointAction,
    JointDriftParameters,
    exact_integer_joint_action,
    joint_drift_score,
    optimal_gain_for_participation,
)
from experiments.dependence_delay_linear.t065_discrete_joint_certificate import (
    ar1_block_mean_variance_factor,
    paired_residual_statistics,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t065a_joint_mechanism_cpu_preregistration.json"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t065a_joint_mechanism_cpu"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(master: int, *labels: object) -> int:
    payload = "|".join([str(master), *(str(label) for label in labels)])
    return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "little")


def scenario_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    grid = config["grid"]
    for rho in grid["rho"]:
        for signal in grid["state_signal"]:
            for noise in grid["noise_scale"]:
                for delay in grid["delay_curvature"]:
                    for price in grid["resource_price"]:
                        cell_id = f"R{rho:g}-S{signal:g}-N{noise:g}-D{delay:g}-P{price:g}"
                        rows.append(
                            {
                                "cell_id": cell_id,
                                "rho": float(rho),
                                "state_signal": float(signal),
                                "noise_scale": float(noise),
                                "delay_curvature": float(delay),
                                "resource_price": float(price),
                            }
                        )
    return rows


def validate(config: dict[str, Any]) -> dict[str, Any]:
    scenarios = scenario_rows(config)
    seeds = config["pilot_seeds"]
    expected = config["expected_workload"]
    if config["analysis"]["uses_prior_outcome_rows"] is not False:
        raise ValueError("outcome-tainted scenario selection is forbidden")
    if config["sensor"]["disjoint_from_evaluation"] is not True:
        raise ValueError("sensor/evaluation separation must be frozen true")
    if config["sensor"]["fully_charged"] is not True:
        raise ValueError("all sensing must be fully charged")
    if len(seeds) != len(set(seeds)):
        raise ValueError("pilot seeds must be unique")
    if len(scenarios) != expected["cells"]:
        raise ValueError("cell count mismatch")
    if len(scenarios) * len(seeds) != expected["endpoints"]:
        raise ValueError("endpoint count mismatch")
    return {
        "experiment_id": config["experiment_id"],
        "cells": len(scenarios),
        "seeds": len(seeds),
        "endpoints": len(scenarios) * len(seeds),
        "recommended_hardware": expected["recommended_hardware"],
    }


def _ar1_block_means(
    rng: np.random.Generator,
    *,
    scale: float,
    correlation: float,
    length: int,
    replicates: int,
    dimension: int,
) -> np.ndarray:
    state = rng.normal(scale=math.sqrt(scale), size=(replicates, dimension))
    total = np.zeros_like(state)
    innovation = math.sqrt(scale * (1.0 - correlation * correlation))
    for _ in range(length):
        state = correlation * state + innovation * rng.normal(size=state.shape)
        total += state
    return total / length


def _parameters(
    config: dict[str, Any], scenario: dict[str, Any], *, signal: float, noise: float, rho: float
) -> JointDriftParameters:
    action = config["action"]
    price = scenario["resource_price"]
    return JointDriftParameters(
        contraction=action["contraction"],
        state_signal=max(0.0, signal),
        delay_curvature=scenario["delay_curvature"],
        noise_coefficient=action["noise_coefficient"],
        noise_scale=max(0.0, noise),
        rho_upper=float(np.clip(rho, 0.0, 1.0)),
        message_price=0.5 * price,
        environment_price=0.5 * price,
        overhead=action["message_overhead"],
        eta_min=action["eta_min"],
        eta_max=action["eta_max"],
    )


def _q_only_action(config: dict[str, Any], parameters: JointDriftParameters) -> JointAction:
    action = config["action"]
    eta = float(action["fixed_eta"])
    evaluated = [
        (joint_drift_score(float(q), eta, parameters), q)
        for q in range(action["q_min"], action["q_max"] + 1)
    ]
    score, q = min(evaluated, key=lambda row: (row[0], row[1]))
    return JointAction(q, eta, score, float(q), (q,))


def _eta_only_action(config: dict[str, Any], parameters: JointDriftParameters) -> JointAction:
    q = int(config["action"]["fixed_q"])
    eta = optimal_gain_for_participation(float(q), parameters)
    score = joint_drift_score(float(q), eta, parameters)
    return JointAction(q, eta, score, float(q), (q,))


def _fixed_action(config: dict[str, Any], parameters: JointDriftParameters) -> JointAction:
    q = int(config["action"]["fixed_q"])
    eta = float(config["action"]["fixed_eta"])
    return JointAction(q, eta, joint_drift_score(float(q), eta, parameters), float(q), (q,))


def run_endpoint(config: dict[str, Any], scenario: dict[str, Any], seed: int) -> dict[str, Any]:
    sensor = config["sensor"]
    rng = np.random.default_rng(stable_seed(seed, scenario["cell_id"], "sensor"))
    direction = np.asarray([1.0, -0.5, 0.25])
    direction /= np.linalg.norm(direction)
    mean_field = math.sqrt(scenario["state_signal"]) * direction
    block_args = {
        "scale": scenario["noise_scale"],
        "correlation": sensor["markov_correlation"],
        "length": sensor["residual_block_length"],
        "replicates": sensor["paired_replicates"],
        "dimension": sensor["dimension"],
    }
    first = mean_field + _ar1_block_means(rng, **block_args)
    second = mean_field + _ar1_block_means(rng, **block_args)
    paired = [paired_residual_statistics(a, b) for a, b in zip(first, second)]
    signal_hat = max(0.0, float(np.mean([row.signal for row in paired])))
    block_factor = ar1_block_mean_variance_factor(
        sensor["markov_correlation"], sensor["residual_block_length"]
    )
    raw_noise = float(np.mean([row.noise for row in paired]))
    noise_hat = max(0.0, raw_noise / (sensor["dimension"] * block_factor))

    collision = sensor["independent_collision"]
    match_probability = collision + (1.0 - collision) * scenario["rho"]
    match_count = int(rng.binomial(sensor["fingerprint_blocks"], match_probability))
    rho_hat = float(
        np.clip(
            (match_count / sensor["fingerprint_blocks"] - collision) / (1.0 - collision),
            0.0,
            1.0,
        )
    )
    true_parameters = _parameters(
        config,
        scenario,
        signal=scenario["state_signal"],
        noise=scenario["noise_scale"],
        rho=scenario["rho"],
    )
    observed_parameters = _parameters(
        config, scenario, signal=signal_hat, noise=noise_hat, rho=rho_hat
    )
    action = config["action"]
    oracle = exact_integer_joint_action(
        q_min=action["q_min"], q_max=action["q_max"], parameters=true_parameters
    )
    observable = exact_integer_joint_action(
        q_min=action["q_min"], q_max=action["q_max"], parameters=observed_parameters
    )
    q_only = _q_only_action(config, true_parameters)
    eta_only = _eta_only_action(config, true_parameters)
    fixed = _fixed_action(config, true_parameters)
    observable_true_score = joint_drift_score(
        observable.participation, observable.gain, true_parameters
    )

    residual_actor_cost = 2 * sensor["paired_replicates"] * sensor["residual_block_length"]
    fingerprint_actor_cost = (
        sensor["q_fingerprint"]
        * sensor["fingerprint_blocks"]
        * sensor["fingerprint_length"]
    )
    sensor_actor_cost = residual_actor_cost + fingerprint_actor_cost
    residual_message_cost = 2 * sensor["paired_replicates"] * (
        action["message_overhead"] + 1
    )
    fingerprint_message_cost = sensor["fingerprint_blocks"] * (
        action["message_overhead"] + sensor["q_fingerprint"]
    )
    sensor_message_cost = int(residual_message_cost + fingerprint_message_cost)
    updates = config["accounting"]["learning_updates"]
    environment_used = sensor_actor_cost + updates * observable.participation
    message_used = sensor_message_cost + updates * (
        action["message_overhead"] + observable.participation
    )
    environment_budget = sensor_actor_cost + updates * action["q_max"]
    message_budget = sensor_message_cost + updates * (
        action["message_overhead"] + action["q_max"]
    )
    denominator = (
        abs(oracle.drift_score)
        + action["contraction"] * scenario["state_signal"] * action["eta_max"]
        + scenario["resource_price"] * action["q_max"]
    )
    return {
        **scenario,
        "seed": seed,
        "signal_hat": signal_hat,
        "noise_hat": noise_hat,
        "rho_hat": rho_hat,
        "match_count": match_count,
        "oracle_q": oracle.participation,
        "oracle_eta": oracle.gain,
        "oracle_score": oracle.drift_score,
        "observable_q": observable.participation,
        "observable_eta": observable.gain,
        "observable_true_score": observable_true_score,
        "normalized_regret": (observable_true_score - oracle.drift_score) / denominator,
        "q_only_score": q_only.drift_score,
        "eta_only_score": eta_only.drift_score,
        "fixed_score": fixed.drift_score,
        "sensor_actor_cost": sensor_actor_cost,
        "sensor_message_cost": sensor_message_cost,
        "environment_used": environment_used,
        "environment_budget": environment_budget,
        "message_used": message_used,
        "message_budget": message_budget,
    }


def _direction_fraction(cell_rows: list[dict[str, Any]], varied: str, outcome: str, increasing: bool) -> float:
    other = [
        name
        for name in ("rho", "state_signal", "noise_scale", "delay_curvature", "resource_price")
        if name != varied
    ]
    groups: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for row in cell_rows:
        groups.setdefault(tuple(row[name] for name in other), []).append(row)
    matches = 0
    for rows in groups.values():
        ordered = sorted(rows, key=lambda row: row[varied])
        values = [row[outcome] for row in ordered]
        matches += int(
            all(
                (right >= left - 1e-12) if increasing else (right <= left + 1e-12)
                for left, right in zip(values, values[1:])
            )
        )
    return matches / len(groups)


def analyze(config: dict[str, Any], endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    validate(config)
    expected = config["expected_workload"]
    if len(endpoints) != expected["endpoints"]:
        raise ValueError("endpoint count mismatch")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in endpoints:
        groups.setdefault(row["cell_id"], []).append(row)
    if len(groups) != expected["cells"] or any(
        len(rows) != len(config["pilot_seeds"]) for rows in groups.values()
    ):
        raise ValueError("cell/seed coverage mismatch")
    cells = []
    for cell_id, rows in sorted(groups.items()):
        first = rows[0]
        cells.append(
            {
                "cell_id": cell_id,
                **{name: first[name] for name in ("rho", "state_signal", "noise_scale", "delay_curvature", "resource_price")},
                "mean_normalized_regret": float(np.mean([row["normalized_regret"] for row in rows])),
                "mean_observable_score": float(np.mean([row["observable_true_score"] for row in rows])),
                "oracle_score": first["oracle_score"],
                "q_only_score": first["q_only_score"],
                "eta_only_score": first["eta_only_score"],
                "fixed_score": first["fixed_score"],
                "median_observable_q": float(np.median([row["observable_q"] for row in rows])),
                "median_observable_eta": float(np.median([row["observable_eta"] for row in rows])),
            }
        )
    regrets = [row["mean_normalized_regret"] for row in cells]
    observable_fixed_fraction = float(np.mean([row["mean_observable_score"] < row["fixed_score"] for row in cells]))
    oracle_joint_fraction = float(np.mean([
        row["oracle_score"] < min(row["q_only_score"], row["eta_only_score"]) - 1e-12
        for row in cells
    ]))
    observable_ablation_fraction = float(np.mean([
        row["mean_observable_score"] < min(row["q_only_score"], row["eta_only_score"])
        for row in cells
    ]))
    q_direction = _direction_fraction(cells, "rho", "median_observable_q", increasing=False)
    eta_direction = _direction_fraction(cells, "state_signal", "median_observable_eta", increasing=True)
    finite = all(
        math.isfinite(float(row[key]))
        for row in endpoints
        for key in ("signal_hat", "noise_hat", "rho_hat", "observable_true_score", "normalized_regret")
    )
    budget_valid = all(
        row["environment_used"] <= row["environment_budget"]
        and row["message_used"] <= row["message_budget"]
        for row in endpoints
    )
    metrics = {
        "cell_count": len(cells),
        "endpoint_count": len(endpoints),
        "finite": finite,
        "nonnegative_noise": all(row["noise_hat"] >= 0.0 for row in endpoints),
        "median_cell_normalized_regret": float(np.median(regrets)),
        "p90_cell_normalized_regret": float(np.quantile(regrets, 0.9)),
        "observable_beats_fixed_fraction": observable_fixed_fraction,
        "oracle_beats_both_one_dimensional_fraction": oracle_joint_fraction,
        "observable_beats_both_one_dimensional_fraction": observable_ablation_fraction,
        "q_nonincreasing_rho_path_fraction": q_direction,
        "eta_nondecreasing_signal_path_fraction": eta_direction,
        "budget_valid": budget_valid,
    }
    gates = {
        "G1": len(cells) == 324 and len(endpoints) == 20736,
        "G2": finite and metrics["nonnegative_noise"],
        "G3": metrics["median_cell_normalized_regret"] <= 0.10,
        "G4": metrics["p90_cell_normalized_regret"] <= 0.35,
        "G5": observable_fixed_fraction >= 0.60,
        "G6": oracle_joint_fraction >= 0.30,
        "G7": observable_ablation_fraction >= 0.50,
        "G8": q_direction >= 0.90,
        "G9": eta_direction >= 0.90,
        "G10": budget_valid,
        "G11": True,
        "G12": False,
    }
    return {"experiment_id": config["experiment_id"], "metrics": metrics, "gates": gates, "cells": cells}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    endpoints = [
        run_endpoint(config, scenario, seed)
        for scenario in scenario_rows(config)
        for seed in config["pilot_seeds"]
    ]
    summary = analyze(config, endpoints)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", summary["cells"])
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        if args.command == "estimate":
            result["ar1_scalar_updates"] = (
                result["endpoints"]
                * config["sensor"]["paired_replicates"]
                * config["sensor"]["residual_block_length"]
                * 2
                * config["sensor"]["dimension"]
            )
    else:
        result = execute(config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
