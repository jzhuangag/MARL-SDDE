"""CPU development scan for passive clocked-secant geometry sensing."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .clocked_optimism_phase import (
    heterogeneous_clock_metric,
    rotational_optimism_threshold,
)
from .dual_use_fingerprint import (
    BinaryGeometryBelief,
    expected_binary_log_gain,
    predict_binary_geometry,
    update_binary_geometry,
)
from .passive_secant_fingerprint import passive_secant_fingerprint
from .run_dual_use_sensor_development import (
    _fixed_masks,
    _paths,
    _phase_log_gains,
    _transition_table,
)


EXPECTED_CONFIG_SHA256 = (
    "fb467ec266003718fc6885a8fa909ffdce83c940e5911bf1cb8c54b09f32bad0"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    if _sha256(path).lower() != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("LCO-P0 development configuration hash mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment"] != "LCO-P0-PASSIVE-SECANT-DEVELOPMENT":
        raise RuntimeError("unexpected passive-secant experiment")
    if config["formal_evidence"] or int(config["horizon"]) % 4 != 0:
        raise RuntimeError("development must be nonformal with period-aligned horizon")
    return config


def _specifications(config: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = range(
        int(config["seeds"]["start"]),
        int(config["seeds"]["start"]) + int(config["seeds"]["count"]),
    )
    return [
        {
            "seed": seed,
            "step": step,
            "arrival": arrival,
            "persistence": persistence,
            "rotation_fraction": rotation_fraction,
            "budget": budget,
            "gradient_noise": noise,
        }
        for seed in seeds
        for step in config["normalized_steps"]
        for arrival in config["first_agent_arrival_probabilities"]
        for persistence in config["phase_persistence"]
        for rotation_fraction in config["rotation_stationary_fractions"]
        for budget in config["optimism_budgets"]
        for noise in config["mandatory_gradient_noise_standard_deviations"]
    ]


def _operator(rotation: bool) -> np.ndarray:
    return (
        np.asarray([[0.0, 1.0], [-1.0, 0.0]])
        if rotation
        else np.eye(2)
    )


def _run_one(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    horizon = int(config["horizon"])
    period = int(config["fixed_schedule_period"])
    step = float(spec["step"])
    arrival = float(spec["arrival"])
    persistence = float(spec["persistence"])
    rotation_fraction = float(spec["rotation_fraction"])
    budget = float(spec["budget"])
    noise = float(spec["gradient_noise"])
    allowance = int(math.floor(budget * horizon + 1e-12))
    phases, agents, initial, gradient_noise, _ = _paths(
        seed=int(spec["seed"]),
        horizon=horizon,
        persistence=persistence,
        rotation_fraction=rotation_fraction,
        arrival=arrival,
    )
    metric = np.diag(heterogeneous_clock_metric(arrival))
    initial /= math.sqrt(float(initial @ metric @ initial))
    potential_gain, rotation_gain = _phase_log_gains(step, arrival, metric)
    table = _transition_table(step)
    masks = _fixed_masks(budget, period)
    names = ["passive", "exact_phase", "never"] + [
        "mask_" + "".join(map(str, mask)) if mask else "mask_none"
        for mask in masks
    ]
    states = np.broadcast_to(initial, (len(names), 2)).copy()
    accumulated = np.zeros(len(names))
    passive_debt = 0.0
    exact_debt = 0.0
    passive_calls = 0
    exact_calls = 0
    passive_potential_calls = 0
    informative_secants = 0
    correct_phase_classifications = 0
    belief = BinaryGeometryBelief(rotation_fraction)
    previous_state: np.ndarray | None = None
    previous_gradient: np.ndarray | None = None
    potential_to_rotation = (1.0 - persistence) * rotation_fraction
    rotation_to_potential = (1.0 - persistence) * (1.0 - rotation_fraction)
    likelihood_sigma = max(
        float(config["likelihood_sigma_floor"]),
        float(config["likelihood_sigma_noise_multiplier"]) * noise / step,
    )
    lyapunov_v = math.sqrt(horizon)

    for event, (rotation, agent) in enumerate(zip(phases, agents)):
        if event > 0:
            belief = predict_binary_geometry(
                belief,
                potential_to_rotation=potential_to_rotation,
                rotation_to_potential=rotation_to_potential,
            )
        passive_state = states[0].copy()
        current_exact_gradient = _operator(bool(rotation)) @ passive_state
        current_observed_gradient = (
            current_exact_gradient + noise * gradient_noise[event]
        )
        if previous_state is not None and previous_gradient is not None:
            fingerprint = passive_secant_fingerprint(
                previous_state,
                previous_gradient,
                passive_state,
                current_observed_gradient,
                minimum_displacement_energy=float(
                    config["minimum_displacement_energy"]
                ),
            )
            if fingerprint.informative:
                score = (
                    fingerprint.rotational_residual
                    - fingerprint.symmetric_alignment
                )
                belief = update_binary_geometry(
                    belief,
                    observed_score=score,
                    observation_standard_deviation=likelihood_sigma,
                )
                informative_secants += 1
                correct_phase_classifications += int(
                    (belief.rotation_probability >= 0.5) == bool(rotation)
                )

        expected_gain = expected_binary_log_gain(
            belief,
            potential_log_gain=potential_gain,
            rotational_log_gain=rotation_gain,
        )
        passive_anchor = bool(
            passive_calls < allowance
            and lyapunov_v * expected_gain > passive_debt
        )
        true_gain = rotation_gain if rotation else potential_gain
        exact_anchor = bool(
            exact_calls < allowance and lyapunov_v * true_gain > exact_debt
        )
        passive_calls += int(passive_anchor)
        exact_calls += int(exact_anchor)
        passive_potential_calls += int(passive_anchor and not rotation)
        passive_debt = max(0.0, passive_debt + float(passive_anchor) - budget)
        exact_debt = max(0.0, exact_debt + float(exact_anchor) - budget)

        actions = np.asarray(
            [passive_anchor, exact_anchor, False]
            + [event % period in mask for mask in masks],
            dtype=int,
        )
        matrices = table[int(rotation), actions, int(agent)]
        states = np.einsum("nij,nj->ni", matrices, states)
        energies = np.einsum("ni,ij,nj->n", states, metric, states)
        if np.any(~np.isfinite(energies)) or np.any(energies <= 0.0):
            raise RuntimeError("nonfinite or nonpositive clocked-game energy")
        accumulated += np.log(energies)
        states /= np.sqrt(energies)[:, None]
        previous_state = passive_state
        previous_gradient = current_observed_gradient

    rates = accumulated / horizon
    mask_rates = {
        name: float(rates[index])
        for index, name in enumerate(names)
        if name.startswith("mask_")
    }
    threshold = rotational_optimism_threshold(step)
    margin = float(config["dynamic_separation_margin"])
    separated_dynamic = bool(
        0.0 < rotation_fraction < 1.0
        and budget / rotation_fraction >= threshold + margin
        and budget <= threshold - margin
    )
    eligible_secants = horizon - 1
    return {
        **spec,
        "passive_log_energy_rate": float(rates[0]),
        "exact_phase_log_energy_rate": float(rates[1]),
        "never_log_energy_rate": float(rates[2]),
        "fixed_mask_log_energy_rates": mask_rates,
        "passive_calls": passive_calls,
        "exact_phase_calls": exact_calls,
        "passive_potential_calls": passive_potential_calls,
        "informative_secants": informative_secants,
        "eligible_secants": eligible_secants,
        "correct_phase_classifications": correct_phase_classifications,
        "allowance": allowance,
        "separated_dynamic": separated_dynamic,
    }


def _run_payload(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return _run_one(*payload)


def _cells(rows: list[dict[str, Any]], horizon: int) -> list[dict[str, Any]]:
    identity_keys = (
        "step",
        "arrival",
        "persistence",
        "rotation_fraction",
        "budget",
        "gradient_noise",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in identity_keys), []).append(row)
    cells = []
    for identity, selected in sorted(groups.items()):
        masks = sorted(selected[0]["fixed_mask_log_energy_rates"])
        mask_means = {
            mask: float(
                np.mean([row["fixed_mask_log_energy_rates"][mask] for row in selected])
            )
            for mask in masks
        }
        best_mask = min(mask_means, key=mask_means.get)
        cell = {key: value for key, value in zip(identity_keys, identity)}
        for metric in (
            "passive_log_energy_rate",
            "exact_phase_log_energy_rate",
            "never_log_energy_rate",
        ):
            cell[metric] = float(np.mean([row[metric] for row in selected]))
        cell["best_fixed_mask"] = best_mask
        cell["best_fixed_log_energy_rate"] = mask_means[best_mask]
        cell["passive_call_fraction"] = float(
            np.mean([row["passive_calls"] for row in selected]) / horizon
        )
        cell["passive_potential_call_fraction"] = float(
            np.mean([row["passive_potential_calls"] for row in selected]) / horizon
        )
        informative = sum(row["informative_secants"] for row in selected)
        eligible = sum(row["eligible_secants"] for row in selected)
        correct = sum(row["correct_phase_classifications"] for row in selected)
        cell["informative_fraction"] = informative / eligible
        cell["phase_accuracy"] = correct / informative if informative else 0.0
        cell["separated_dynamic"] = bool(selected[0]["separated_dynamic"])
        exact_gain = (
            cell["best_fixed_log_energy_rate"]
            - cell["exact_phase_log_energy_rate"]
        )
        passive_gain = (
            cell["best_fixed_log_energy_rate"] - cell["passive_log_energy_rate"]
        )
        cell["passive_exact_gain_capture"] = (
            passive_gain / exact_gain if exact_gain > 1e-15 else 1.0
        )
        cells.append(cell)
    return cells


def _mean_gain(cells: list[dict[str, Any]]) -> float:
    return float(
        np.mean(
            [
                cell["best_fixed_log_energy_rate"]
                - cell["passive_log_energy_rate"]
                for cell in cells
            ]
        )
    )


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    horizon = int(config["horizon"])
    cells = _cells(rows, horizon)
    dynamic = [cell for cell in cells if cell["separated_dynamic"]]
    noise_metrics: dict[str, Any] = {}
    for noise in config["mandatory_gradient_noise_standard_deviations"]:
        selected = [cell for cell in dynamic if cell["gradient_noise"] == noise]
        gains = np.asarray(
            [cell["best_fixed_log_energy_rate"] - cell["passive_log_energy_rate"] for cell in selected]
        )
        noise_metrics[str(noise)] = {
            "dynamic_cell_count": len(selected),
            "mean_dynamic_gain": float(np.mean(gains)),
            "dynamic_improvement_fraction": float(np.mean(gains > 0.0)),
            "dynamic_contraction_fraction": float(
                np.mean([cell["passive_log_energy_rate"] < 0.0 for cell in selected])
            ),
            "median_exact_gain_capture": float(
                np.median([cell["passive_exact_gain_capture"] for cell in selected])
            ),
            "mean_phase_accuracy": float(np.mean([cell["phase_accuracy"] for cell in selected])),
            "mean_informative_fraction": float(
                np.mean([cell["informative_fraction"] for cell in selected])
            ),
        }
    arrival_gains = {
        str(arrival): _mean_gain([cell for cell in dynamic if cell["arrival"] == arrival])
        for arrival in config["first_agent_arrival_probabilities"]
    }
    low_persistence = min(config["phase_persistence"])
    low_budget = min(config["optimism_budgets"])
    low_persistence_gain = _mean_gain(
        [cell for cell in dynamic if cell["persistence"] == low_persistence]
    )
    low_budget_gain = _mean_gain(
        [cell for cell in dynamic if cell["budget"] == low_budget]
    )
    potential = [cell for cell in cells if cell["rotation_fraction"] == 0.0]
    potential_loss = max(
        cell["passive_log_energy_rate"] - cell["never_log_energy_rate"]
        for cell in potential
    )
    maximum_overshoot = max(
        row[f"{prefix}_calls"] - row["allowance"]
        for row in rows
        for prefix in ("passive", "exact_phase")
    )
    thresholds = config["development_survival_gates"]
    gates = {
        "P1_finite_and_accounted": all(
            math.isfinite(row[metric])
            for row in rows
            for metric in (
                "passive_log_energy_rate",
                "exact_phase_log_energy_rate",
                "never_log_energy_rate",
            )
        ),
        "P2_each_noise_gain": all(
            value["mean_dynamic_gain"]
            >= thresholds["minimum_each_noise_mean_dynamic_gain"]
            for value in noise_metrics.values()
        ),
        "P3_each_noise_capture": all(
            value["median_exact_gain_capture"]
            >= thresholds["minimum_each_noise_median_exact_gain_capture"]
            for value in noise_metrics.values()
        ),
        "P4_each_noise_cells": all(
            value["dynamic_improvement_fraction"]
            >= thresholds["minimum_each_noise_dynamic_cell_improvement_fraction"]
            for value in noise_metrics.values()
        ),
        "P5_each_arrival": all(
            gain >= thresholds["minimum_each_arrival_mean_dynamic_gain"]
            for gain in arrival_gains.values()
        ),
        "P6_low_persistence": low_persistence_gain
        >= thresholds["minimum_low_persistence_mean_dynamic_gain"],
        "P7_low_budget": low_budget_gain
        >= thresholds["minimum_low_budget_mean_dynamic_gain"],
        "P8_phase_accuracy": all(
            value["mean_phase_accuracy"] >= thresholds["minimum_each_noise_phase_accuracy"]
            for value in noise_metrics.values()
        ),
        "P9_informative_coverage": all(
            value["mean_informative_fraction"]
            >= thresholds["minimum_each_noise_informative_fraction"]
            for value in noise_metrics.values()
        ),
        "P10_stationary_potential": potential_loss
        <= thresholds["maximum_stationary_potential_log_rate_loss"],
        "P11_budget": maximum_overshoot <= thresholds["maximum_budget_overshoot"],
        "P12_development_only": not bool(config["formal_evidence"]),
    }
    return {
        "row_count": len(rows),
        "cell_count": len(cells),
        "dynamic_cell_count": len(dynamic),
        "metrics_by_noise": noise_metrics,
        "arrival_group_gains": arrival_gains,
        "low_persistence_mean_dynamic_gain": low_persistence_gain,
        "low_budget_mean_dynamic_gain": low_budget_gain,
        "stationary_potential_max_log_rate_loss": potential_loss,
        "maximum_budget_overshoot": maximum_overshoot,
        "gates": gates,
        "development_survives": all(gates.values()),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = _load_config(args.config)
    specifications = _specifications(config)
    if args.command == "validate":
        print(f"config_sha256={_sha256(args.config)}")
        print("validation=pass")
        return
    if args.command == "estimate":
        print(f"paths={len(specifications)}")
        print(f"coordinate_events={len(specifications) * int(config['horizon'])}")
        return
    if args.output_dir is None:
        raise ValueError("run requires --output-dir")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    payloads = ((specification, config) for specification in specifications)
    if args.workers == 1:
        rows = [_run_payload(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_run_payload, payloads, chunksize=8))
    summary = _summarize(rows, config)
    payload = {
        "experiment": config["experiment"],
        "config_sha256": _sha256(args.config),
        "rows": rows,
        "summary": summary,
        "formal_evidence": False,
        "gpu_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "summary.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "cells"}, indent=2, sort_keys=True))
    print(f"summary_sha256={_sha256(output)}")


if __name__ == "__main__":
    main()
