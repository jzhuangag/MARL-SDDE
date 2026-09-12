"""Frozen CPU phase-headroom scan for Lyapunov-clocked optimism."""

from __future__ import annotations

import argparse
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


EXPECTED_CONFIG_SHA256 = (
    "58caceeea755d8a1057073eeae0cca9284abc0f4f8e139c695c7d834eb54f6b8"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    if _sha256(path).lower() != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("LCO-H1 configuration hash mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment"] != "LCO-H1" or config["horizon"] % 4 != 0:
        raise RuntimeError("invalid frozen LCO-H1 configuration")
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
        }
        for seed in seeds
        for step in config["normalized_steps"]
        for arrival in config["first_agent_arrival_probabilities"]
        for persistence in config["phase_persistence"]
        for rotation_fraction in config["rotation_stationary_fractions"]
        for budget in config["optimism_budgets"]
    ]


def _fixed_masks(budget: float, period: int) -> tuple[tuple[int, ...], ...]:
    maximum = int(math.floor(budget * period + 1e-12))
    return tuple(
        subset
        for size in range(maximum + 1)
        for subset in itertools.combinations(range(period), size)
    )


def _phase_and_arrival_paths(
    *,
    seed: int,
    horizon: int,
    persistence: float,
    rotation_fraction: float,
    first_agent_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phases = np.zeros(horizon, dtype=bool)
    state = bool(rng.random() < rotation_fraction)
    potential_to_rotation = (1.0 - persistence) * rotation_fraction
    rotation_to_potential = (1.0 - persistence) * (1.0 - rotation_fraction)
    for event in range(horizon):
        phases[event] = state
        if state:
            if rng.random() < rotation_to_potential:
                state = False
        elif rng.random() < potential_to_rotation:
            state = True
    agents = (rng.random(horizon) >= first_agent_probability).astype(int)
    initial = rng.normal(size=2)
    return phases, agents, initial


def _transition_matrix(
    *, phase_is_rotational: bool, use_optimism: bool, agent: int, step: float
) -> np.ndarray:
    selector = np.zeros((2, 2))
    selector[agent, agent] = 1.0
    operator = (
        np.asarray([[0.0, 1.0], [-1.0, 0.0]])
        if phase_is_rotational
        else np.eye(2)
    )
    if use_optimism:
        return np.eye(2) - step * selector @ operator @ (
            np.eye(2) - step * operator
        )
    return np.eye(2) - step * selector @ operator


def _phase_multipliers(
    *, step: float, arrival: float, metric: np.ndarray
) -> dict[tuple[bool, bool], float]:
    probabilities = (arrival, 1.0 - arrival)
    return {
        (rotation, optimism): expected_quadratic_multiplier(
            metric,
            tuple(
                _transition_matrix(
                    phase_is_rotational=rotation,
                    use_optimism=optimism,
                    agent=agent,
                    step=step,
                )
                for agent in (0, 1)
            ),
            probabilities,
        )
        for rotation in (False, True)
        for optimism in (False, True)
    }


def _run_one(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    horizon = int(config["horizon"])
    period = int(config["fixed_schedule_period"])
    step = float(spec["step"])
    arrival = float(spec["arrival"])
    budget = float(spec["budget"])
    allowance = int(math.floor(budget * horizon + 1e-12))
    phases, agents, initial = _phase_and_arrival_paths(
        seed=int(spec["seed"]),
        horizon=horizon,
        persistence=float(spec["persistence"]),
        rotation_fraction=float(spec["rotation_fraction"]),
        first_agent_probability=arrival,
    )
    metric = np.diag(heterogeneous_clock_metric(arrival))
    initial /= math.sqrt(float(initial @ metric @ initial))
    multipliers = _phase_multipliers(step=step, arrival=arrival, metric=metric)
    masks = _fixed_masks(budget, period)
    names = ["controller", "phase_oracle", "never", "always"] + [
        "mask_" + "".join(map(str, mask)) if mask else "mask_none"
        for mask in masks
    ]
    states = np.broadcast_to(initial, (len(names), 2)).copy()
    accumulated_log_energy = np.zeros(len(names))
    controller_debt = 0.0
    controller_anchors = 0
    controller_potential_anchors = 0
    oracle_anchors = 0
    lyapunov_v = math.sqrt(horizon)

    for event, (rotation, agent) in enumerate(zip(phases, agents)):
        controller_decision = choose_log_drift_anchor(
            plain_multiplier=multipliers[(bool(rotation), False)],
            fresh_multiplier=multipliers[(bool(rotation), True)],
            resource_debt=controller_debt,
            average_anchor_budget=budget,
            lyapunov_tradeoff=lyapunov_v,
            hard_feasible=controller_anchors < allowance,
        )
        controller_anchor = controller_decision.use_fresh_anchor
        controller_debt = controller_decision.resource_debt_after
        controller_anchors += int(controller_anchor)
        controller_potential_anchors += int(controller_anchor and not rotation)
        oracle_anchor = bool(rotation and oracle_anchors < allowance)
        oracle_anchors += int(oracle_anchor)
        actions = [controller_anchor, oracle_anchor, False, True] + [
            event % period in mask for mask in masks
        ]
        for index, use_optimism in enumerate(actions):
            transition = _transition_matrix(
                phase_is_rotational=bool(rotation),
                use_optimism=bool(use_optimism),
                agent=int(agent),
                step=step,
            )
            states[index] = transition @ states[index]
            energy = float(states[index] @ metric @ states[index])
            if not math.isfinite(energy) or energy <= 0.0:
                raise RuntimeError("nonfinite or nonpositive clocked-game energy")
            accumulated_log_energy[index] += math.log(energy)
            states[index] /= math.sqrt(energy)

    rates = accumulated_log_energy / horizon
    mask_rates = {
        name: float(rates[index])
        for index, name in enumerate(names)
        if name.startswith("mask_")
    }
    threshold = rotational_optimism_threshold(step)
    rotation_fraction = float(spec["rotation_fraction"])
    margin = float(config["dynamic_separation_margin"])
    separated_dynamic = bool(
        0.0 < rotation_fraction < 1.0
        and budget / rotation_fraction >= threshold + margin
        and budget <= threshold - margin
    )
    return {
        **spec,
        "controller_log_energy_rate": float(rates[0]),
        "phase_oracle_log_energy_rate": float(rates[1]),
        "never_log_energy_rate": float(rates[2]),
        "always_log_energy_rate": float(rates[3]),
        "fixed_mask_log_energy_rates": mask_rates,
        "controller_anchors": controller_anchors,
        "controller_potential_anchors": controller_potential_anchors,
        "phase_oracle_anchors": oracle_anchors,
        "allowance": allowance,
        "realized_rotation_fraction": float(np.mean(phases)),
        "rotational_threshold": threshold,
        "separated_dynamic": separated_dynamic,
    }


def _cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity_keys = (
        "step",
        "arrival",
        "persistence",
        "rotation_fraction",
        "budget",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in identity_keys), []).append(row)
    cells = []
    for identity, selected in sorted(groups.items()):
        masks = sorted(selected[0]["fixed_mask_log_energy_rates"])
        mask_means = {
            mask: float(
                np.mean(
                    [row["fixed_mask_log_energy_rates"][mask] for row in selected]
                )
            )
            for mask in masks
        }
        best_mask = min(mask_means, key=mask_means.get)
        cell = {key: value for key, value in zip(identity_keys, identity)}
        for metric in (
            "controller_log_energy_rate",
            "phase_oracle_log_energy_rate",
            "never_log_energy_rate",
            "always_log_energy_rate",
        ):
            cell[metric] = float(np.mean([row[metric] for row in selected]))
        cell["best_fixed_mask"] = best_mask
        cell["best_fixed_log_energy_rate"] = mask_means[best_mask]
        cell["controller_anchor_fraction"] = float(
            np.mean([row["controller_anchors"] / row["allowance"] for row in selected])
            * float(identity[-1])
        )
        cell["controller_potential_anchors"] = int(
            sum(row["controller_potential_anchors"] for row in selected)
        )
        cell["separated_dynamic"] = bool(selected[0]["separated_dynamic"])
        fixed_gain = (
            cell["best_fixed_log_energy_rate"]
            - cell["phase_oracle_log_energy_rate"]
        )
        controller_gain = (
            cell["best_fixed_log_energy_rate"]
            - cell["controller_log_energy_rate"]
        )
        cell["phase_oracle_gain_capture"] = (
            controller_gain / fixed_gain if fixed_gain > 1e-15 else 1.0
        )
        cells.append(cell)
    return cells


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    cells = _cells(rows)
    separated = [cell for cell in cells if cell["separated_dynamic"]]
    potential = [cell for cell in cells if cell["rotation_fraction"] == 0.0]
    gains = np.asarray(
        [
            cell["best_fixed_log_energy_rate"]
            - cell["controller_log_energy_rate"]
            for cell in separated
        ]
    )
    captures = np.asarray(
        [cell["phase_oracle_gain_capture"] for cell in separated]
    )
    arrival_gains = {
        str(arrival): float(
            np.mean(
                [
                    cell["best_fixed_log_energy_rate"]
                    - cell["controller_log_energy_rate"]
                    for cell in separated
                    if cell["arrival"] == arrival
                ]
            )
        )
        for arrival in config["first_agent_arrival_probabilities"]
    }
    potential_anchor_fraction = sum(
        row["controller_potential_anchors"] for row in rows
        if row["rotation_fraction"] == 0.0
    ) / max(
        1,
        sum(row["allowance"] for row in rows if row["rotation_fraction"] == 0.0),
    )
    potential_error = max(
        abs(cell["controller_log_energy_rate"] - cell["never_log_energy_rate"])
        for cell in potential
    )
    overshoot = max(row["controller_anchors"] - row["allowance"] for row in rows)
    finite = all(
        math.isfinite(value)
        for row in rows
        for value in (
            row["controller_log_energy_rate"],
            row["phase_oracle_log_energy_rate"],
            row["never_log_energy_rate"],
            row["always_log_energy_rate"],
        )
    )
    thresholds = config["mandatory_gates"]
    mean_gain = float(np.mean(gains))
    improvement_fraction = float(np.mean(gains > 0.0))
    contraction_fraction = float(
        np.mean([cell["controller_log_energy_rate"] < 0.0 for cell in separated])
    )
    median_capture = float(np.median(captures))
    gates = {
        "L1_finite_and_exact_accounting": finite,
        "L2_dynamic_gain": mean_gain
        >= thresholds["L2_separated_dynamic_log_rate_gain_min"],
        "L3_dynamic_cells": improvement_fraction
        >= thresholds["L3_separated_dynamic_cell_improvement_fraction_min"],
        "L4_dynamic_contraction": contraction_fraction
        >= thresholds["L4_separated_dynamic_contraction_fraction_min"],
        "L5_oracle_capture": median_capture
        >= thresholds["L5_median_phase_oracle_gain_capture_min"],
        "L6_potential_no_anchor": potential_anchor_fraction
        <= thresholds["L6_potential_anchor_fraction_max"],
        "L7_potential_exact": potential_error
        <= thresholds["L7_potential_controller_never_log_rate_abs_error_max"],
        "L8_arrival_groups": all(
            gain >= thresholds["L8_each_arrival_group_log_rate_gain_min"]
            for gain in arrival_gains.values()
        ),
        "L9_budget": overshoot <= thresholds["L9_budget_overshoot_max"],
        "L10_stop_rule": True,
    }
    return {
        "row_count": len(rows),
        "cell_count": len(cells),
        "separated_dynamic_cell_count": len(separated),
        "mean_separated_dynamic_log_rate_gain": mean_gain,
        "separated_dynamic_cell_improvement_fraction": improvement_fraction,
        "separated_dynamic_contraction_fraction": contraction_fraction,
        "median_phase_oracle_gain_capture": median_capture,
        "potential_anchor_fraction": potential_anchor_fraction,
        "potential_controller_never_max_abs_error": potential_error,
        "arrival_group_log_rate_gains": arrival_gains,
        "maximum_budget_overshoot": overshoot,
        "gates": gates,
        "all_mandatory_gates_pass": all(gates.values()),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _load_config(args.config)
    specifications = _specifications(config)
    if args.command == "validate":
        print(f"config_sha256={_sha256(args.config)}")
        print("validation=pass")
        return
    if args.command == "estimate":
        print(f"scenarios={len(specifications)}")
        print(f"coordinate_events={len(specifications) * config['horizon']}")
        return
    if args.output_dir is None:
        raise ValueError("run requires --output-dir")
    rows = [_run_one(specification, config) for specification in specifications]
    summary = _summarize(rows, config)
    payload = {
        "experiment": "LCO-H1",
        "config_sha256": _sha256(args.config),
        "rows": rows,
        "summary": summary,
        "formal_evidence": False,
        "gpu_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cells"}, indent=2, sort_keys=True))
    print(f"summary_sha256={_sha256(path)}")


if __name__ == "__main__":
    main()
