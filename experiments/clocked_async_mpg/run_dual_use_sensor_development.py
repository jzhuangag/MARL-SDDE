"""CPU development scan for the causal dual-use optimism sensor."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .clocked_optimism_phase import (
    choose_log_drift_anchor,
    expected_quadratic_multiplier,
    heterogeneous_clock_metric,
    rotational_optimism_threshold,
)
from .dual_use_fingerprint import (
    BinaryGeometryBelief,
    directional_geometry_fingerprint,
    expected_binary_log_gain,
    predict_binary_geometry,
    update_binary_geometry,
)


EXPECTED_CONFIG_SHA256 = (
    "43fac4f2ec52b478b729c61066a66afba287b8958415994a9cdd4b2bbf0f724d"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    if _sha256(path).lower() != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("LCO-S0 development configuration hash mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment"] != "LCO-S0-DEVELOPMENT":
        raise RuntimeError("unexpected development experiment")
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
            "fingerprint_noise": noise,
            "probe_period": probe_period,
        }
        for seed in seeds
        for step in config["normalized_steps"]
        for arrival in config["first_agent_arrival_probabilities"]
        for persistence in config["phase_persistence"]
        for rotation_fraction in config["rotation_stationary_fractions"]
        for budget in config["optimism_budgets"]
        for noise in config["fingerprint_noise_standard_deviations"]
        for probe_period in config["probe_periods"]
    ]


def _fixed_masks(budget: float, period: int) -> tuple[tuple[int, ...], ...]:
    maximum = int(math.floor(budget * period + 1e-12))
    return tuple(
        subset
        for size in range(maximum + 1)
        for subset in itertools.combinations(range(period), size)
    )


def _paths(
    *,
    seed: int,
    horizon: int,
    persistence: float,
    rotation_fraction: float,
    arrival: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase_rng, arrival_rng, state_rng, noise_rng = (
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(4)
    )
    phases = np.zeros(horizon, dtype=bool)
    state = bool(phase_rng.random() < rotation_fraction)
    potential_to_rotation = (1.0 - persistence) * rotation_fraction
    rotation_to_potential = (1.0 - persistence) * (1.0 - rotation_fraction)
    uniforms = phase_rng.random(horizon)
    for event in range(horizon):
        phases[event] = state
        if state and uniforms[event] < rotation_to_potential:
            state = False
        elif not state and uniforms[event] < potential_to_rotation:
            state = True
    agents = (arrival_rng.random(horizon) >= arrival).astype(int)
    initial = state_rng.normal(size=2)
    first_noise = noise_rng.normal(size=(horizon, 2))
    second_noise = noise_rng.normal(size=(horizon, 2))
    return phases, agents, initial, first_noise, second_noise


def _operator(rotation: bool) -> np.ndarray:
    return (
        np.asarray([[0.0, 1.0], [-1.0, 0.0]])
        if rotation
        else np.eye(2)
    )


def _transition_table(step: float) -> np.ndarray:
    table = np.empty((2, 2, 2, 2, 2), dtype=float)
    identity = np.eye(2)
    for rotation in (False, True):
        operator = _operator(rotation)
        for optimism in (False, True):
            direction = operator @ (identity - step * operator) if optimism else operator
            for agent in (0, 1):
                selector = np.zeros((2, 2))
                selector[agent, agent] = 1.0
                table[int(rotation), int(optimism), agent] = (
                    identity - step * selector @ direction
                )
    return table


def _phase_log_gains(step: float, arrival: float, metric: np.ndarray) -> tuple[float, float]:
    table = _transition_table(step)
    probabilities = (arrival, 1.0 - arrival)
    gains = []
    for rotation in (False, True):
        multipliers = []
        for optimism in (False, True):
            multipliers.append(
                expected_quadratic_multiplier(
                    metric,
                    tuple(table[int(rotation), int(optimism), agent] for agent in (0, 1)),
                    probabilities,
                )
            )
        gains.append(math.log(multipliers[0]) - math.log(multipliers[1]))
    return float(gains[0]), float(gains[1])


def _run_one(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    horizon = int(config["horizon"])
    period = int(config["fixed_schedule_period"])
    step = float(spec["step"])
    arrival = float(spec["arrival"])
    persistence = float(spec["persistence"])
    rotation_fraction = float(spec["rotation_fraction"])
    budget = float(spec["budget"])
    noise = float(spec["fingerprint_noise"])
    probe_period = int(spec["probe_period"])
    allowance = int(math.floor(budget * horizon + 1e-12))
    phases, agents, initial, first_noise, second_noise = _paths(
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
    names = ["sensor", "exact_phase", "never"] + [
        "mask_" + "".join(map(str, mask)) if mask else "mask_none"
        for mask in masks
    ]
    states = np.broadcast_to(initial, (len(names), 2)).copy()
    accumulated = np.zeros(len(names))
    sensor_debt = 0.0
    exact_debt = 0.0
    sensor_calls = 0
    exact_calls = 0
    forced_probes = 0
    potential_calls = 0
    informative_fingerprints = 0
    belief = BinaryGeometryBelief(rotation_probability=rotation_fraction)
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
        expected_gain = expected_binary_log_gain(
            belief,
            potential_log_gain=potential_gain,
            rotational_log_gain=rotation_gain,
        )
        probe_due = bool(
            0.0 < rotation_fraction < 1.0 and event % probe_period == 0
        )
        sensor_value_call = lyapunov_v * expected_gain > sensor_debt
        sensor_anchor = bool(
            sensor_calls < allowance and (probe_due or sensor_value_call)
        )
        forced_probes += int(sensor_anchor and probe_due and not sensor_value_call)
        sensor_calls += int(sensor_anchor)
        potential_calls += int(sensor_anchor and not rotation)
        sensor_debt = max(0.0, sensor_debt + float(sensor_anchor) - budget)

        true_gain = rotation_gain if rotation else potential_gain
        exact_anchor = bool(
            exact_calls < allowance and lyapunov_v * true_gain > exact_debt
        )
        exact_calls += int(exact_anchor)
        exact_debt = max(0.0, exact_debt + float(exact_anchor) - budget)
        actions = np.asarray(
            [sensor_anchor, exact_anchor, False]
            + [event % period in mask for mask in masks],
            dtype=int,
        )
        sensor_state = states[0].copy()
        matrices = table[int(rotation), actions, int(agent)]
        states = np.einsum("nij,nj->ni", matrices, states)
        energies = np.einsum("ni,ij,nj->n", states, metric, states)
        if np.any(~np.isfinite(energies)) or np.any(energies <= 0.0):
            raise RuntimeError("nonfinite or nonpositive clocked-game energy")
        accumulated += np.log(energies)
        states /= np.sqrt(energies)[:, None]

        if sensor_anchor:
            operator = _operator(bool(rotation))
            current = operator @ sensor_state + noise * first_noise[event]
            lookahead = (
                operator @ (sensor_state - step * (operator @ sensor_state))
                + noise * second_noise[event]
            )
            fingerprint = directional_geometry_fingerprint(
                current,
                lookahead,
                lookahead_step=step,
                minimum_gradient_energy=float(config["minimum_gradient_energy"]),
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
                informative_fingerprints += 1

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
    return {
        **spec,
        "sensor_log_energy_rate": float(rates[0]),
        "exact_phase_log_energy_rate": float(rates[1]),
        "never_log_energy_rate": float(rates[2]),
        "fixed_mask_log_energy_rates": mask_rates,
        "sensor_calls": sensor_calls,
        "exact_phase_calls": exact_calls,
        "forced_probes": forced_probes,
        "potential_calls": potential_calls,
        "informative_fingerprints": informative_fingerprints,
        "allowance": allowance,
        "separated_dynamic": separated_dynamic,
    }


def _run_payload(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    specification, config = payload
    return _run_one(specification, config)


def _cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity_keys = (
        "step",
        "arrival",
        "persistence",
        "rotation_fraction",
        "budget",
        "fingerprint_noise",
        "probe_period",
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
            "sensor_log_energy_rate",
            "exact_phase_log_energy_rate",
            "never_log_energy_rate",
        ):
            cell[metric] = float(np.mean([row[metric] for row in selected]))
        cell["best_fixed_mask"] = best_mask
        cell["best_fixed_log_energy_rate"] = mask_means[best_mask]
        cell["sensor_anchor_fraction"] = float(
            np.mean([row["sensor_calls"] for row in selected])
            / int(selected[0]["allowance"])
            * float(cell["budget"])
        )
        cell["forced_probe_fraction"] = float(
            np.mean([row["forced_probes"] for row in selected]) / 1024.0
        )
        cell["potential_call_fraction"] = float(
            np.mean([row["potential_calls"] for row in selected]) / 1024.0
        )
        cell["separated_dynamic"] = bool(selected[0]["separated_dynamic"])
        exact_gain = (
            cell["best_fixed_log_energy_rate"]
            - cell["exact_phase_log_energy_rate"]
        )
        sensor_gain = (
            cell["best_fixed_log_energy_rate"] - cell["sensor_log_energy_rate"]
        )
        cell["exact_gain_capture"] = (
            sensor_gain / exact_gain if exact_gain > 1e-15 else 1.0
        )
        cells.append(cell)
    return cells


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    cells = _cells(rows)
    metrics: dict[str, Any] = {}
    for probe_period in config["probe_periods"]:
        for noise in config["fingerprint_noise_standard_deviations"]:
            selected = [
                cell
                for cell in cells
                if cell["probe_period"] == probe_period
                and cell["fingerprint_noise"] == noise
            ]
            dynamic = [cell for cell in selected if cell["separated_dynamic"]]
            gains = np.asarray(
                [
                    cell["best_fixed_log_energy_rate"]
                    - cell["sensor_log_energy_rate"]
                    for cell in dynamic
                ]
            )
            captures = np.asarray([cell["exact_gain_capture"] for cell in dynamic])
            key = f"probe={probe_period}|noise={noise}"
            metrics[key] = {
                "dynamic_cell_count": len(dynamic),
                "mean_dynamic_gain": float(np.mean(gains)),
                "dynamic_improvement_fraction": float(np.mean(gains > 0.0)),
                "dynamic_contraction_fraction": float(
                    np.mean([cell["sensor_log_energy_rate"] < 0.0 for cell in dynamic])
                ),
                "median_exact_gain_capture": float(np.median(captures)),
                "mean_forced_probe_fraction": float(
                    np.mean([cell["forced_probe_fraction"] for cell in dynamic])
                ),
            }
    minimum_noise_gain = {
        int(probe): min(
            metrics[f"probe={probe}|noise={noise}"]["mean_dynamic_gain"]
            for noise in config["fingerprint_noise_standard_deviations"]
        )
        for probe in config["probe_periods"]
    }
    selected_probe = max(
        (int(probe) for probe in config["probe_periods"]),
        key=lambda probe: (minimum_noise_gain[probe], probe),
    )
    selected_metrics = {
        str(noise): metrics[f"probe={selected_probe}|noise={noise}"]
        for noise in config["fingerprint_noise_standard_deviations"]
    }
    selected_dynamic = [
        cell
        for cell in cells
        if cell["probe_period"] == selected_probe and cell["separated_dynamic"]
    ]
    arrival_gains = {
        str(arrival): float(
            np.mean(
                [
                    cell["best_fixed_log_energy_rate"]
                    - cell["sensor_log_energy_rate"]
                    for cell in selected_dynamic
                    if cell["arrival"] == arrival
                ]
            )
        )
        for arrival in config["first_agent_arrival_probabilities"]
    }
    potential = [
        cell
        for cell in cells
        if cell["probe_period"] == selected_probe
        and cell["rotation_fraction"] == 0.0
    ]
    potential_loss = max(
        cell["sensor_log_energy_rate"] - cell["never_log_energy_rate"]
        for cell in potential
    )
    maximum_overshoot = max(row["sensor_calls"] - row["allowance"] for row in rows)
    thresholds = config["development_survival_gates"]
    gates = {
        "D1_finite_and_accounted": all(
            math.isfinite(row[metric])
            for row in rows
            for metric in (
                "sensor_log_energy_rate",
                "exact_phase_log_energy_rate",
                "never_log_energy_rate",
            )
        ),
        "D2_each_noise_gain": all(
            value["mean_dynamic_gain"]
            >= thresholds["minimum_each_noise_mean_dynamic_gain"]
            for value in selected_metrics.values()
        ),
        "D3_each_noise_capture": all(
            value["median_exact_gain_capture"]
            >= thresholds["minimum_each_noise_median_exact_gain_capture"]
            for value in selected_metrics.values()
        ),
        "D4_each_noise_cells": all(
            value["dynamic_improvement_fraction"]
            >= thresholds["minimum_each_noise_dynamic_cell_improvement_fraction"]
            for value in selected_metrics.values()
        ),
        "D5_each_arrival": all(
            gain >= thresholds["minimum_each_arrival_mean_dynamic_gain"]
            for gain in arrival_gains.values()
        ),
        "D6_stationary_potential": potential_loss
        <= thresholds["maximum_stationary_potential_log_rate_loss"],
        "D7_budget": maximum_overshoot
        <= thresholds["maximum_budget_overshoot"],
        "D8_development_only": not bool(config["formal_evidence"]),
    }
    return {
        "row_count": len(rows),
        "cell_count": len(cells),
        "selected_probe_period": selected_probe,
        "minimum_noise_gain_by_probe": minimum_noise_gain,
        "metrics_by_probe_and_noise": metrics,
        "selected_metrics_by_noise": selected_metrics,
        "selected_arrival_group_gains": arrival_gains,
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
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "cells"},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"summary_sha256={_sha256(output)}")


if __name__ == "__main__":
    main()
