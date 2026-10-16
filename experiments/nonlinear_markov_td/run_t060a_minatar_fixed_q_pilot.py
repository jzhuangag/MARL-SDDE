"""Run the preregistered T-060A CPU fixed-q MinAtar value pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from numba import njit
import numpy as np
import torch

from experiments.nonlinear_markov_td.t059_minatar_fixed_encoder import (
    FrozenConvEncoder,
    ReferenceMoments,
    coupled_prefix_indices,
    encoded_stream,
    reference_moments,
    sample_stream,
    stationary_cost_coefficient,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs" / "t060a_minatar_fixed_q_pilot_preregistration.json"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "nonlinear_markov_td"
    / "results"
    / "t060a_minatar_fixed_q_pilot"
)


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = dict(config)
    payload.pop("configuration_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    observed = canonical_config_hash(config)
    if observed != config["configuration_sha256"]:
        raise RuntimeError(f"configuration hash mismatch: {observed}")
    return config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(master_seed: int, *labels: object) -> int:
    text = "|".join((str(master_seed), *(str(label) for label in labels)))
    value = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    return int(value % (2**31 - 1))


def action_updates(
    config: dict[str, Any], *, overhead: int, q: int, delay: int
) -> int:
    learning = config["learning"]
    q_max = max(config["grid"]["participation"])
    target = int(learning["target_updates_qmax"])
    message_budget = (overhead + q_max) * target
    environment_budget = q_max * target
    synchronized_rounds = min(message_budget // (overhead + q), environment_budget // q)
    return max(0, int(synchronized_rounds - delay))


def lifted_spectral_radius(drift: np.ndarray, *, step_size: float, delay: int) -> float:
    dimension = drift.shape[0]
    if delay == 0:
        return float(max(abs(np.linalg.eigvals(np.eye(dimension) - step_size * drift))))
    lifted = np.zeros(((delay + 1) * dimension, (delay + 1) * dimension))
    lifted[:dimension, :dimension] = np.eye(dimension)
    lifted[:dimension, delay * dimension :] = -step_size * drift
    for block in range(1, delay + 1):
        lifted[block * dimension : (block + 1) * dimension,
               (block - 1) * dimension : block * dimension] = np.eye(dimension)
    return float(max(abs(np.linalg.eigvals(lifted))))


def combine_references(values: list[ReferenceMoments]) -> ReferenceMoments:
    if not values:
        raise ValueError("reference list must be nonempty")
    drift = np.mean([value.drift for value in values], axis=0)
    reward = np.mean([value.reward_vector for value in values], axis=0)
    covariance = np.mean([value.feature_covariance for value in values], axis=0)
    fixed = np.linalg.solve(drift, reward)
    symmetric = 0.5 * (drift + drift.T)
    return ReferenceMoments(
        drift=drift,
        reward_vector=reward,
        feature_covariance=covariance,
        fixed_point=fixed,
        symmetric_min_eigenvalue=float(np.linalg.eigvalsh(symmetric)[0]),
        spectral_norm=float(np.linalg.svd(drift, compute_uv=False)[0]),
    )


def reference_for_game(
    config: dict[str, Any], game: str
) -> tuple[ReferenceMoments, dict[str, Any]]:
    task = config["tasks"][game]
    encoder = FrozenConvEncoder(
        int(task["channels"]),
        seed=int(task["encoder_seed"]),
        filters=int(config["encoder"]["filters"]),
        output_features=int(config["encoder"]["output_features"]),
    )
    banks: list[ReferenceMoments] = []
    for bank, seeds in enumerate(task["reference_seeds"]):
        stream = sample_stream(
            game,
            transitions=int(config["reference"]["transitions_per_bank"]),
            environment_seed=int(seeds["environment"]),
            policy_seed=int(seeds["policy"]),
            sticky_action_probability=float(config["environment"]["sticky_action_probability"]),
            difficulty_ramping=bool(config["environment"]["difficulty_ramping"]),
        )
        encoded = encoded_stream(
            encoder, stream, batch_size=int(config["encoder"]["batch_size"])
        )
        banks.append(
            reference_moments(
                [encoded],
                discount=float(config["learning"]["discount"]),
                regularization=float(config["learning"]["regularization"]),
            )
        )
    combined = combine_references(banks)
    drift_relative = float(
        np.linalg.norm(banks[0].drift - banks[1].drift)
        / max(np.linalg.norm(combined.drift), 1e-15)
    )
    prediction_scale = float(
        np.sqrt(
            max(
                combined.fixed_point
                @ combined.feature_covariance
                @ combined.fixed_point,
                0.0,
            )
        )
    )
    difference = banks[0].fixed_point - banks[1].fixed_point
    fixed_point_relative = float(
        np.sqrt(max(difference @ combined.feature_covariance @ difference, 0.0))
        / max(prediction_scale, float(config["reference"]["prediction_scale_floor"]))
    )
    step_size = min(
        float(config["learning"]["maximum_step_size"]),
        float(config["learning"]["spectral_step_fraction"])
        / combined.spectral_norm,
    )
    radii = {
        str(delay): lifted_spectral_radius(combined.drift, step_size=step_size, delay=int(delay))
        for delay in config["grid"]["delays"]
    }
    condition = float(np.linalg.cond(combined.drift))
    diagnostics = {
        "game": game,
        "encoder_sha256": encoder.fingerprint(),
        "drift_relative_disagreement": drift_relative,
        "fixed_point_prediction_relative_disagreement": fixed_point_relative,
        "symmetric_min_eigenvalue": combined.symmetric_min_eigenvalue,
        "drift_condition_number": condition,
        "step_size": step_size,
        "lifted_spectral_radius": radii,
        "fixed_point_prediction_rms": prediction_scale,
    }
    return combined, diagnostics


def reference_to_json(reference: ReferenceMoments, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        **diagnostics,
        "drift": reference.drift.tolist(),
        "reward_vector": reference.reward_vector.tolist(),
        "feature_covariance": reference.feature_covariance.tolist(),
        "fixed_point": reference.fixed_point.tolist(),
    }


def maximum_path_horizons(config: dict[str, Any]) -> dict[int, int]:
    result = {}
    for q in config["grid"]["participation"]:
        result[int(q)] = max(
            action_updates(config, overhead=int(overhead), q=int(q), delay=int(delay)) + int(delay)
            for overhead in config["grid"]["overheads"]
            for delay in config["grid"]["delays"]
        )
    return result


def learning_bank(
    config: dict[str, Any], *, game: str, master_seed: int
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], str]:
    task = config["tasks"][game]
    encoder = FrozenConvEncoder(
        int(task["channels"]),
        seed=int(task["encoder_seed"]),
        filters=int(config["encoder"]["filters"]),
        output_features=int(config["encoder"]["output_features"]),
    )
    horizons = maximum_path_horizons(config)
    bank: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    q_max = max(config["grid"]["participation"])
    for path in range(q_max + 1):
        if path <= 1:
            horizon = horizons[1]
        elif path <= 4:
            horizon = horizons[4]
        else:
            horizon = horizons[16]
        stream = sample_stream(
            game,
            transitions=horizon,
            environment_seed=derived_seed(master_seed, game, "environment", path),
            policy_seed=derived_seed(master_seed, game, "policy", path),
            sticky_action_probability=float(config["environment"]["sticky_action_probability"]),
            difficulty_ramping=bool(config["environment"]["difficulty_ramping"]),
        )
        bank.append(
            encoded_stream(
                encoder, stream, batch_size=int(config["encoder"]["batch_size"])
            )
        )
    return bank, encoder.fingerprint()


@njit(cache=True)
def _delayed_regularized_td(
    features: np.ndarray,
    successors: np.ndarray,
    rewards: np.ndarray,
    fixed_point: np.ndarray,
    covariance: np.ndarray,
    drift: np.ndarray,
    reward_vector: np.ndarray,
    step_size: float,
    discount: float,
    regularization: float,
    delay: int,
) -> tuple[float, float, np.ndarray]:
    actors, updates, dimension = features.shape
    weights = np.zeros((updates + delay + 1, dimension))
    burn = updates // 2
    average = np.zeros(dimension)
    count = 0
    for time in range(updates):
        current = weights[delay + time]
        stale = weights[time]
        gradient = -regularization * stale.copy()
        for actor in range(actors):
            phi = features[actor, time]
            successor = successors[actor, time]
            delta = rewards[actor, time] + discount * np.dot(successor, stale) - np.dot(phi, stale)
            gradient += phi * delta / actors
        weights[delay + time + 1] = current + step_size * gradient
        if time >= burn:
            average += weights[delay + time + 1]
            count += 1
    average /= count
    error = average - fixed_point
    prediction_risk = float(error @ covariance @ error)
    residual = drift @ average - reward_vector
    bellman_residual = float(residual @ residual)
    return prediction_risk, bellman_residual, average


def run_fixed_q(
    *,
    config: dict[str, Any],
    reference: ReferenceMoments,
    diagnostics: dict[str, Any],
    bank: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    game: str,
    master_seed: int,
    overhead: int,
    rho: float,
    delay: int,
    q: int,
) -> dict[str, Any]:
    updates = action_updates(config, overhead=overhead, q=q, delay=delay)
    indices = coupled_prefix_indices(
        rho=rho,
        q_max=max(config["grid"]["participation"]),
        seed=derived_seed(master_seed, game, "coupling", rho),
    )[:q]
    features = np.stack([bank[int(index)][0][:updates] for index in indices])
    successors = np.stack([bank[int(index)][1][:updates] for index in indices])
    rewards = np.stack([bank[int(index)][2][:updates] for index in indices])
    prediction_risk, residual, average = _delayed_regularized_td(
        features,
        successors,
        rewards,
        reference.fixed_point,
        reference.feature_covariance,
        reference.drift,
        reference.reward_vector,
        float(diagnostics["step_size"]),
        float(config["learning"]["discount"]),
        float(config["learning"]["regularization"]),
        int(delay),
    )
    message_budget = (overhead + max(config["grid"]["participation"])) * int(
        config["learning"]["target_updates_qmax"]
    )
    environment_budget = max(config["grid"]["participation"]) * int(
        config["learning"]["target_updates_qmax"]
    )
    return {
        "master_seed": int(master_seed),
        "split": "selection" if master_seed in config["selection_seeds"] else "validation",
        "game": game,
        "overhead": int(overhead),
        "rho": float(rho),
        "delay": int(delay),
        "q": int(q),
        "updates": int(updates),
        "message_budget": int(message_budget),
        "message_used": int((updates + delay) * (overhead + q)),
        "environment_budget": int(environment_budget),
        "environment_used": int((updates + delay) * q),
        "prediction_risk": prediction_risk,
        "bellman_residual": residual,
        "average_weight_norm": float(np.linalg.norm(average)),
        "selected_common_actors": int(np.sum(indices == 0)),
        "encoder_sha256": diagnostics["encoder_sha256"],
    }


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires finite positive values")
    return float(np.exp(np.mean(np.log(array))))


def cell_means(endpoints: list[dict[str, Any]], split: str) -> dict[tuple, float]:
    grouped: dict[tuple, list[float]] = {}
    for row in endpoints:
        if row["split"] != split:
            continue
        key = (row["game"], row["overhead"], row["rho"], row["delay"], row["q"])
        grouped.setdefault(key, []).append(float(row["prediction_risk"]))
    return {key: float(np.mean(value)) for key, value in grouped.items()}


def analyze(
    config: dict[str, Any],
    endpoints: list[dict[str, Any]],
    reference_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    selection = cell_means(endpoints, "selection")
    validation = cell_means(endpoints, "validation")
    actions = [int(q) for q in config["grid"]["participation"]]
    strong: dict[tuple[str, int], int] = {}
    oracle: dict[tuple[str, int, float, int], int] = {}
    for game in config["tasks"]:
        for overhead in config["grid"]["overheads"]:
            strong[(game, int(overhead))] = min(
                actions,
                key=lambda q: geometric_mean(
                    [
                        selection[(game, int(overhead), float(rho), int(delay), q)]
                        for rho in config["grid"]["correlations"]
                        for delay in config["grid"]["delays"]
                    ]
                ),
            )
            for rho in config["grid"]["correlations"]:
                for delay in config["grid"]["delays"]:
                    key = (game, int(overhead), float(rho), int(delay))
                    oracle[key] = min(
                        actions,
                        key=lambda q: selection[(*key, q)],
                    )
    cells = []
    ratios = []
    for game in config["tasks"]:
        for overhead in config["grid"]["overheads"]:
            for rho in config["grid"]["correlations"]:
                for delay in config["grid"]["delays"]:
                    base_q = strong[(game, int(overhead))]
                    oracle_q = oracle[(game, int(overhead), float(rho), int(delay))]
                    base = validation[(game, int(overhead), float(rho), int(delay), base_q)]
                    adapted = validation[(game, int(overhead), float(rho), int(delay), oracle_q)]
                    ratio = adapted / base
                    ratios.append(ratio)
                    cells.append(
                        {
                            "game": game,
                            "overhead": int(overhead),
                            "rho": float(rho),
                            "delay": int(delay),
                            "strong_q": int(base_q),
                            "oracle_q": int(oracle_q),
                            "strong_validation_risk": base,
                            "oracle_validation_risk": adapted,
                            "ratio": ratio,
                        }
                    )
    task_ratios = {
        game: geometric_mean([row["ratio"] for row in cells if row["game"] == game])
        for game in config["tasks"]
    }
    directional_paths = 0
    directional_matches = 0
    for game in config["tasks"]:
        for overhead in config["grid"]["overheads"]:
            for delay in config["grid"]["delays"]:
                choices = [
                    oracle[(game, int(overhead), float(rho), int(delay))]
                    for rho in config["grid"]["correlations"]
                ]
                directional_paths += 1
                directional_matches += int(all(a >= b for a, b in zip(choices, choices[1:])))
    expected_rows = (
        len(config["pilot_seeds"])
        * len(config["tasks"])
        * len(config["grid"]["overheads"])
        * len(config["grid"]["correlations"])
        * len(config["grid"]["delays"])
        * len(config["grid"]["participation"])
    )
    reference_gates = []
    for item in reference_diagnostics:
        reference_gates.append(
            item["drift_relative_disagreement"]
            <= config["gates"]["maximum_reference_drift_disagreement"]
            and item["fixed_point_prediction_relative_disagreement"]
            <= config["gates"]["maximum_reference_prediction_disagreement"]
            and item["symmetric_min_eigenvalue"] > 0.0
            and item["drift_condition_number"] <= config["gates"]["maximum_condition_number"]
            and max(item["lifted_spectral_radius"].values()) < 1.0
        )
    finite_and_budgeted = all(
        np.isfinite(row["prediction_risk"])
        and row["prediction_risk"] > 0.0
        and row["message_used"] <= row["message_budget"]
        and row["environment_used"] <= row["environment_budget"]
        for row in endpoints
    )
    aggregate_ratio = geometric_mean(ratios)
    strict_fraction = float(np.mean([row["ratio"] < 1.0 for row in cells]))
    gates = {
        "V1_complete_unique": len(endpoints) == expected_rows
        and len({(row["master_seed"], row["game"], row["overhead"], row["rho"], row["delay"], row["q"]) for row in endpoints}) == expected_rows,
        "V2_finite_dual_budget": finite_and_budgeted,
        "V3_reference_stable": all(reference_gates),
        "V4_heldout_oracle_gain": aggregate_ratio <= config["gates"]["maximum_oracle_strong_ratio"],
        "V5_heldout_directional_breadth": strict_fraction >= config["gates"]["minimum_strict_cell_fraction"],
        "V6_taskwise_value": all(value <= config["gates"]["maximum_task_oracle_strong_ratio"] for value in task_ratios.values()),
        "V7_rho_direction": directional_matches / directional_paths >= config["gates"]["minimum_directional_path_fraction"],
        "V8_split_isolation": not set(config["selection_seeds"]) & set(config["validation_seeds"])
        and set(config["selection_seeds"]) | set(config["validation_seeds"]) == set(config["pilot_seeds"]),
        "V9_no_controller": all("controller" not in row for row in endpoints),
    }
    return {
        "experiment_id": config["experiment_id"],
        "configuration_sha256": config["configuration_sha256"],
        "endpoints": len(endpoints),
        "cells": len(cells),
        "selection_seeds": len(config["selection_seeds"]),
        "validation_seeds": len(config["validation_seeds"]),
        "strong_fixed_q": {f"{game}|{overhead}": q for (game, overhead), q in strong.items()},
        "heldout_oracle_strong_geometric_ratio": aggregate_ratio,
        "heldout_oracle_improvement": 1.0 - aggregate_ratio,
        "heldout_strict_cell_fraction": strict_fraction,
        "task_ratios": task_ratios,
        "rho_directional_paths": f"{directional_matches}/{directional_paths}",
        "reference_diagnostics": reference_diagnostics,
        "gates": gates,
        "pre_reproduction_value_gate_pass": all(gates.values()),
        "controller_pilot_authorized": False,
        "cell_rows": cells,
    }


def validate(config: dict[str, Any]) -> dict[str, Any]:
    actions = [int(q) for q in config["grid"]["participation"]]
    theoretical_ratios = []
    strict = 0
    total = 0
    for overhead in config["grid"]["overheads"]:
        strong = min(
            actions,
            key=lambda q: geometric_mean(
                [stationary_cost_coefficient(overhead=overhead, q=q, rho=rho) for rho in config["grid"]["correlations"]]
            ),
        )
        for rho in config["grid"]["correlations"]:
            oracle = min(actions, key=lambda q: stationary_cost_coefficient(overhead=overhead, q=q, rho=rho))
            ratio = stationary_cost_coefficient(overhead=overhead, q=oracle, rho=rho) / stationary_cost_coefficient(overhead=overhead, q=strong, rho=rho)
            theoretical_ratios.extend([ratio] * len(config["grid"]["delays"]))
            strict += int(ratio < 1.0) * len(config["grid"]["delays"])
            total += len(config["grid"]["delays"])
    return {
        "configuration_sha256": canonical_config_hash(config),
        "configuration_hash_matches": canonical_config_hash(config) == config["configuration_sha256"],
        "pilot_seeds_unique": len(set(config["pilot_seeds"])) == len(config["pilot_seeds"]),
        "splits_valid": not set(config["selection_seeds"]) & set(config["validation_seeds"])
        and set(config["selection_seeds"]) | set(config["validation_seeds"]) == set(config["pilot_seeds"]),
        "theoretical_oracle_strong_ratio": geometric_mean(theoretical_ratios),
        "theoretical_strict_fraction": strict / total,
        "theoretical_value_gate": geometric_mean(theoretical_ratios) <= 0.95 and strict / total >= 0.60,
    }


def estimate(config: dict[str, Any]) -> dict[str, Any]:
    horizons = maximum_path_horizons(config)
    per_game_seed = 2 * horizons[1] + 3 * horizons[4] + 12 * horizons[16]
    reference = (
        len(config["tasks"])
        * int(config["reference"]["banks"])
        * int(config["reference"]["transitions_per_bank"])
    )
    learning = len(config["tasks"]) * len(config["pilot_seeds"]) * per_game_seed
    endpoints = (
        len(config["tasks"])
        * len(config["pilot_seeds"])
        * len(config["grid"]["overheads"])
        * len(config["grid"]["correlations"])
        * len(config["grid"]["delays"])
        * len(config["grid"]["participation"])
    )
    return {
        "reference_environment_transitions": reference,
        "learning_bank_environment_transitions": learning,
        "total_generated_environment_transitions": reference + learning,
        "endpoints": endpoints,
        "estimated_peak_gib": 1.5,
        "recommended_device": "local CPU",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = sorted(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(config["encoder"]["torch_threads"]))
    references: dict[str, ReferenceMoments] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    reference_rows = {}
    for game in config["tasks"]:
        reference, audit = reference_for_game(config, game)
        references[game] = reference
        diagnostics[game] = audit
        reference_rows[game] = reference_to_json(reference, audit)
    started = time.perf_counter()
    endpoints: list[dict[str, Any]] = []
    for game in config["tasks"]:
        for master_seed in config["pilot_seeds"]:
            bank, encoder_hash = learning_bank(config, game=game, master_seed=int(master_seed))
            if encoder_hash != diagnostics[game]["encoder_sha256"]:
                raise RuntimeError("encoder fingerprint changed")
            for overhead in config["grid"]["overheads"]:
                for rho in config["grid"]["correlations"]:
                    for delay in config["grid"]["delays"]:
                        for q in config["grid"]["participation"]:
                            endpoints.append(
                                run_fixed_q(
                                    config=config,
                                    reference=references[game],
                                    diagnostics=diagnostics[game],
                                    bank=bank,
                                    game=game,
                                    master_seed=int(master_seed),
                                    overhead=int(overhead),
                                    rho=float(rho),
                                    delay=int(delay),
                                    q=int(q),
                                )
                            )
    summary = analyze(config, endpoints, list(diagnostics.values()))
    cell_rows = summary.pop("cell_rows")
    runtime_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cell_rows)
    with (output / "reference_moments.json").open("w", encoding="utf-8") as handle:
        json.dump(reference_rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
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
