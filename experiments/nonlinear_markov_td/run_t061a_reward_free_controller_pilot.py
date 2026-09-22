"""Run the prospective T-061A reward-free MinAtar controller pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from experiments.nonlinear_markov_td.run_t060a_minatar_fixed_q_pilot import (
    _delayed_regularized_td,
    canonical_config_hash,
    geometric_mean,
    learning_bank,
    run_fixed_q,
)
from experiments.nonlinear_markov_td.t059_minatar_fixed_encoder import ReferenceMoments
from experiments.nonlinear_markov_td.t061_reward_free_fingerprint import (
    action_from_match_count,
    controller_updates,
    phase_action,
    probe_match_count,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs" / "t061a_reward_free_controller_pilot_preregistration.json"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "nonlinear_markov_td"
    / "results"
    / "t061a_reward_free_controller_pilot"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    observed = canonical_config_hash(config)
    if observed != config["configuration_sha256"]:
        raise RuntimeError(f"configuration hash mismatch: {observed}")
    reference_path = ROOT / config["reference"]["path"]
    if file_sha256(reference_path) != config["reference"]["sha256"]:
        raise RuntimeError("reference moment artifact hash mismatch")
    return config


def load_references(
    config: dict[str, Any]
) -> tuple[dict[str, ReferenceMoments], dict[str, dict[str, Any]]]:
    path = ROOT / config["reference"]["path"]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    references = {}
    diagnostics = {}
    for game, row in payload.items():
        references[game] = ReferenceMoments(
            drift=np.asarray(row["drift"], dtype=float),
            reward_vector=np.asarray(row["reward_vector"], dtype=float),
            feature_covariance=np.asarray(row["feature_covariance"], dtype=float),
            fixed_point=np.asarray(row["fixed_point"], dtype=float),
            symmetric_min_eigenvalue=float(row["symmetric_min_eigenvalue"]),
            spectral_norm=float(np.linalg.svd(np.asarray(row["drift"], dtype=float), compute_uv=False)[0]),
        )
        diagnostics[game] = {
            key: value
            for key, value in row.items()
            if key
            not in {"drift", "reward_vector", "feature_covariance", "fixed_point"}
        }
    return references, diagnostics


def run_controller(
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
    probe = config["probe"]
    cost = controller_updates(
        overhead=overhead,
        q=q,
        delay=delay,
        target_qmax_updates=int(config["learning"]["target_updates_qmax"]),
        probe_blocks=int(probe["blocks"]),
        probe_q=int(probe["q"]),
        fingerprint_length=int(probe["length"]),
    )
    from experiments.nonlinear_markov_td.run_t060a_minatar_fixed_q_pilot import (
        coupled_prefix_indices,
        derived_seed,
    )

    indices = coupled_prefix_indices(
        rho=rho,
        q_max=16,
        seed=derived_seed(master_seed, game, "coupling", rho),
    )[:q]
    updates = int(cost["updates"])
    features = np.stack([bank[int(index)][0][:updates] for index in indices])
    successors = np.stack([bank[int(index)][1][:updates] for index in indices])
    rewards = np.stack([bank[int(index)][2][:updates] for index in indices])
    risk, residual, average = _delayed_regularized_td(
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
    return {
        **cost,
        "prediction_risk": float(risk),
        "bellman_residual": float(residual),
        "average_weight_norm": float(np.linalg.norm(average)),
    }


def run_endpoint(
    *,
    config: dict[str, Any],
    reference: ReferenceMoments,
    diagnostics: dict[str, Any],
    bank: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    game: str,
    master_seed: int,
    rho: float,
    match_count: int,
    overhead: int,
    delay: int,
) -> dict[str, Any]:
    selected_q = action_from_match_count(
        matches=match_count,
        blocks=int(config["probe"]["blocks"]),
        overhead=overhead,
    )
    strong_q = int(config["comparators"]["strong_fixed_q"][f"{game}|{overhead}"])
    theory_q = phase_action(overhead=overhead, rho_estimate=rho)
    controller = run_controller(
        config=config,
        reference=reference,
        diagnostics=diagnostics,
        bank=bank,
        game=game,
        master_seed=master_seed,
        overhead=overhead,
        rho=rho,
        delay=delay,
        q=selected_q,
    )
    fixed_cache = {}
    fixed_config = {**config, "selection_seeds": config.get("selection_seeds", [])}
    for label, q in (("strong", strong_q), ("theory", theory_q)):
        if q not in fixed_cache:
            fixed_cache[q] = run_fixed_q(
                config=fixed_config,
                reference=reference,
                diagnostics=diagnostics,
                bank=bank,
                game=game,
                master_seed=master_seed,
                overhead=overhead,
                rho=rho,
                delay=delay,
                q=q,
            )
    strong = fixed_cache[strong_q]
    theory = fixed_cache[theory_q]
    return {
        "master_seed": int(master_seed),
        "game": game,
        "rho": float(rho),
        "overhead": int(overhead),
        "delay": int(delay),
        "match_count": int(match_count),
        "selected_q": int(selected_q),
        "strong_q": int(strong_q),
        "true_rho_q": int(theory_q),
        "controller_updates": int(controller["updates"]),
        "controller_risk": float(controller["prediction_risk"]),
        "strong_risk": float(strong["prediction_risk"]),
        "true_rho_full_budget_risk": float(theory["prediction_risk"]),
        "controller_bellman_residual": float(controller["bellman_residual"]),
        "strong_bellman_residual": float(strong["bellman_residual"]),
        "message_budget": int(controller["message_budget"]),
        "environment_budget": int(controller["environment_budget"]),
        "probe_message": int(controller["probe_message"]),
        "probe_environment": int(controller["probe_environment"]),
        "learning_message": int(controller["learning_message"]),
        "learning_environment": int(controller["learning_environment"]),
        "encoder_sha256": diagnostics["encoder_sha256"],
    }


def cell_means(endpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in endpoints:
        key = (row["game"], row["rho"], row["overhead"], row["delay"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        controller = float(np.mean([row["controller_risk"] for row in rows]))
        strong = float(np.mean([row["strong_risk"] for row in rows]))
        theory = float(np.mean([row["true_rho_full_budget_risk"] for row in rows]))
        result.append(
            {
                "game": key[0],
                "rho": key[1],
                "overhead": key[2],
                "delay": key[3],
                "controller_mean_risk": controller,
                "strong_mean_risk": strong,
                "true_rho_full_budget_mean_risk": theory,
                "controller_strong_ratio": controller / strong,
                "controller_true_rho_ratio": controller / theory,
                "median_selected_q": float(np.median([row["selected_q"] for row in rows])),
                "mean_match_count": float(np.mean([row["match_count"] for row in rows])),
            }
        )
    return result


def analyze(config: dict[str, Any], endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    cells = cell_means(endpoints)
    aggregate = geometric_mean([row["controller_strong_ratio"] for row in cells])
    task_ratios = {
        game: geometric_mean(
            [row["controller_strong_ratio"] for row in cells if row["game"] == game]
        )
        for game in config["tasks"]
    }
    delay_ratios = {
        str(delay): geometric_mean(
            [row["controller_strong_ratio"] for row in cells if row["delay"] == delay]
        )
        for delay in config["grid"]["delays"]
    }
    strict_fraction = float(
        np.mean([row["controller_strong_ratio"] < 1.0 for row in cells])
    )
    oracle_ratio = geometric_mean([row["controller_true_rho_ratio"] for row in cells])
    directional_paths = 0
    directional_matches = 0
    for game in config["tasks"]:
        for overhead in config["grid"]["overheads"]:
            for delay in config["grid"]["delays"]:
                choices = [
                    next(
                        row["median_selected_q"]
                        for row in cells
                        if row["game"] == game
                        and row["overhead"] == overhead
                        and row["delay"] == delay
                        and row["rho"] == rho
                    )
                    for rho in config["grid"]["correlations"]
                ]
                directional_paths += 1
                directional_matches += int(
                    all(first >= second for first, second in zip(choices, choices[1:]))
                )
    unique_probe = {}
    for row in endpoints:
        unique_probe[(row["master_seed"], row["game"], row["rho"])] = row[
            "match_count"
        ]
    standardized = []
    rho_zero = []
    blocks = int(config["probe"]["blocks"])
    for (seed, game, rho), matches in unique_probe.items():
        if rho == 0.0:
            rho_zero.append(matches / blocks)
        if 0.0 < rho < 1.0:
            standardized.append((matches - blocks * rho) / math.sqrt(blocks * rho * (1.0 - rho)))
    fingerprint_rmse = float(np.sqrt(np.mean(np.square(standardized))))
    expected_rows = (
        len(config["pilot_seeds"])
        * len(config["tasks"])
        * len(config["grid"]["correlations"])
        * len(config["grid"]["overheads"])
        * len(config["grid"]["delays"])
    )
    finite_budgeted = all(
        np.isfinite(row["controller_risk"])
        and row["controller_risk"] > 0.0
        and row["probe_message"] + row["learning_message"] <= row["message_budget"]
        and row["probe_environment"] + row["learning_environment"] <= row["environment_budget"]
        for row in endpoints
    )
    gates = {
        "P1_complete_unique": len(endpoints) == expected_rows
        and len({(row["master_seed"], row["game"], row["rho"], row["overhead"], row["delay"]) for row in endpoints}) == expected_rows,
        "P2_finite_full_cost": finite_budgeted,
        "P3_aggregate_gain": aggregate <= config["gates"]["maximum_aggregate_ratio"],
        "P4_taskwise_gain": all(value <= config["gates"]["maximum_task_ratio"] for value in task_ratios.values()),
        "P5_delay_gain": all(value <= config["gates"]["maximum_delay_ratio"] for value in delay_ratios.values()),
        "P6_directional_breadth": strict_fraction >= config["gates"]["minimum_strict_cell_fraction"],
        "P7_true_rho_proximity": oracle_ratio <= config["gates"]["maximum_true_rho_ratio"],
        "P8_participation_direction": directional_matches / directional_paths >= config["gates"]["minimum_directional_path_fraction"],
        "P9_fingerprint_calibration": fingerprint_rmse <= config["gates"]["maximum_fingerprint_rmse"],
        "P10_independent_collision": max(rho_zero, default=0.0) <= config["gates"]["maximum_seed_level_rho0_match_rate"],
        "P11_new_seed_coverage": len(set(config["pilot_seeds"])) == len(config["pilot_seeds"]),
    }
    return {
        "experiment_id": config["experiment_id"],
        "configuration_sha256": config["configuration_sha256"],
        "endpoints": len(endpoints),
        "cells": len(cells),
        "aggregate_controller_strong_ratio": aggregate,
        "aggregate_improvement": 1.0 - aggregate,
        "task_ratios": task_ratios,
        "delay_ratios": delay_ratios,
        "strict_cell_fraction": strict_fraction,
        "controller_true_rho_full_budget_ratio": oracle_ratio,
        "rho_directional_paths": f"{directional_matches}/{directional_paths}",
        "fingerprint_standardized_rmse": fingerprint_rmse,
        "maximum_seed_level_rho0_match_rate": max(rho_zero, default=0.0),
        "gates": gates,
        "pre_reproduction_pilot_gate_pass": all(gates.values()),
        "formal_authorized": False,
        "gpu_authorized": False,
        "hpc4_authorized": False,
        "cell_rows": cells,
    }


def validate(config: dict[str, Any]) -> dict[str, Any]:
    probe = config["probe"]
    actions = {
        str(overhead): [
            action_from_match_count(
                matches=matches,
                blocks=int(probe["blocks"]),
                overhead=int(overhead),
            )
            for matches in range(int(probe["blocks"]) + 1)
        ]
        for overhead in config["grid"]["overheads"]
    }
    return {
        "configuration_hash_matches": canonical_config_hash(config)
        == config["configuration_sha256"],
        "pilot_seeds_unique": len(set(config["pilot_seeds"]))
        == len(config["pilot_seeds"]),
        "action_tables_nonincreasing": all(
            all(first >= second for first, second in zip(values, values[1:]))
            for values in actions.values()
        ),
        "reference_hash_matches": file_sha256(ROOT / config["reference"]["path"])
        == config["reference"]["sha256"],
    }


def estimate(config: dict[str, Any]) -> dict[str, Any]:
    from experiments.nonlinear_markov_td.run_t060a_minatar_fixed_q_pilot import (
        maximum_path_horizons,
    )

    horizons = maximum_path_horizons(config)
    learning_per_game_seed = 2 * horizons[1] + 3 * horizons[4] + 12 * horizons[16]
    learning = len(config["tasks"]) * len(config["pilot_seeds"]) * learning_per_game_seed
    probe = (
        len(config["tasks"])
        * len(config["pilot_seeds"])
        * len(config["grid"]["correlations"])
        * int(config["probe"]["blocks"])
        * 3
        * int(config["probe"]["length"])
    )
    endpoints = (
        len(config["tasks"])
        * len(config["pilot_seeds"])
        * len(config["grid"]["correlations"])
        * len(config["grid"]["overheads"])
        * len(config["grid"]["delays"])
    )
    return {
        "learning_bank_transitions": learning,
        "generated_probe_transitions": probe,
        "total_generated_transitions": learning + probe,
        "endpoints": endpoints,
        "estimated_peak_gib": 1.5,
        "recommended_device": "local CPU",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    references, diagnostics = load_references(config)
    endpoints = []
    started = time.perf_counter()
    for game in config["tasks"]:
        for master_seed in config["pilot_seeds"]:
            bank, encoder_hash = learning_bank(
                config, game=game, master_seed=int(master_seed)
            )
            if encoder_hash != diagnostics[game]["encoder_sha256"]:
                raise RuntimeError("encoder fingerprint changed")
            for rho in config["grid"]["correlations"]:
                matches = probe_match_count(
                    game=game,
                    rho=float(rho),
                    blocks=int(config["probe"]["blocks"]),
                    length=int(config["probe"]["length"]),
                    master_seed=int(master_seed),
                    sticky_action_probability=float(config["environment"]["sticky_action_probability"]),
                    difficulty_ramping=bool(config["environment"]["difficulty_ramping"]),
                )
                for overhead in config["grid"]["overheads"]:
                    for delay in config["grid"]["delays"]:
                        endpoints.append(
                            run_endpoint(
                                config=config,
                                reference=references[game],
                                diagnostics=diagnostics[game],
                                bank=bank,
                                game=game,
                                master_seed=int(master_seed),
                                rho=float(rho),
                                match_count=int(matches),
                                overhead=int(overhead),
                                delay=int(delay),
                            )
                        )
    summary = analyze(config, endpoints)
    cells = summary.pop("cell_rows")
    runtime = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cells)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {**summary, "runtime_seconds": runtime}


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
