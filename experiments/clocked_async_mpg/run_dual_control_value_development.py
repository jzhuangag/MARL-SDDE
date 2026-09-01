"""CPU development run for Lyapunov-priced dual-use optimism."""

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
    choose_log_drift_anchor,
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
from .dual_use_value_of_information import choose_dual_use_lookahead
from .run_dual_use_sensor_development import (
    _fixed_masks,
    _paths,
    _phase_log_gains,
    _transition_table,
)


EXPECTED_CONFIG_SHA256 = (
    "2e9f3fc22961c13a73860a7a937d5d6428ef7683afd515436bce62f625706580"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    if _sha256(path).lower() != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("LCO-V0 development configuration hash mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment"] != "LCO-V0-DEVELOPMENT":
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
        }
        for seed in seeds
        for step in config["normalized_steps"]
        for arrival in config["first_agent_arrival_probabilities"]
        for persistence in config["phase_persistence"]
        for rotation_fraction in config["rotation_stationary_fractions"]
        for budget in config["optimism_budgets"]
        for noise in config["fingerprint_noise_standard_deviations"]
    ]


def _observe(
    *,
    state: np.ndarray,
    rotation: bool,
    step: float,
    noise: float,
    first_noise: np.ndarray,
    second_noise: np.ndarray,
    likelihood_sigma: float,
    belief: BinaryGeometryBelief,
    minimum_gradient_energy: float,
) -> tuple[BinaryGeometryBelief, bool]:
    operator = (
        np.asarray([[0.0, 1.0], [-1.0, 0.0]]) if rotation else np.eye(2)
    )
    exact_current = operator @ state
    current = exact_current + noise * first_noise
    lookahead = (
        operator @ (state - step * exact_current) + noise * second_noise
    )
    fingerprint = directional_geometry_fingerprint(
        current,
        lookahead,
        lookahead_step=step,
        minimum_gradient_energy=minimum_gradient_energy,
    )
    if not fingerprint.informative:
        return belief, False
    score = fingerprint.rotational_residual - fingerprint.symmetric_alignment
    return (
        update_binary_geometry(
            belief,
            observed_score=score,
            observation_standard_deviation=likelihood_sigma,
        ),
        True,
    )


def _run_one(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    horizon = int(config["horizon"])
    period = int(config["fixed_schedule_period"])
    step = float(spec["step"])
    arrival = float(spec["arrival"])
    persistence = float(spec["persistence"])
    rotation_fraction = float(spec["rotation_fraction"])
    budget = float(spec["budget"])
    noise = float(spec["fingerprint_noise"])
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
    names = ["dual", "myopic", "exact_phase", "never"] + [
        "mask_" + "".join(map(str, mask)) if mask else "mask_none"
        for mask in masks
    ]
    states = np.broadcast_to(initial, (len(names), 2)).copy()
    accumulated = np.zeros(len(names))
    debts = {"dual": 0.0, "myopic": 0.0, "exact_phase": 0.0}
    calls = {"dual": 0, "myopic": 0, "exact_phase": 0}
    potential_calls = {"dual": 0, "myopic": 0}
    informative = {"dual": 0, "myopic": 0}
    information_induced_calls = 0
    beliefs = {
        "dual": BinaryGeometryBelief(rotation_fraction),
        "myopic": BinaryGeometryBelief(rotation_fraction),
    }
    potential_to_rotation = (1.0 - persistence) * rotation_fraction
    rotation_to_potential = (1.0 - persistence) * (1.0 - rotation_fraction)
    likelihood_sigma = max(
        float(config["likelihood_sigma_floor"]),
        float(config["likelihood_sigma_noise_multiplier"]) * noise / step,
    )
    lyapunov_v = math.sqrt(horizon)

    for event, (rotation, agent) in enumerate(zip(phases, agents)):
        if event > 0:
            for name in beliefs:
                beliefs[name] = predict_binary_geometry(
                    beliefs[name],
                    potential_to_rotation=potential_to_rotation,
                    rotation_to_potential=rotation_to_potential,
                )

        dual_decision = choose_dual_use_lookahead(
            beliefs["dual"],
            potential_to_rotation=potential_to_rotation,
            rotation_to_potential=rotation_to_potential,
            potential_log_gain=potential_gain,
            rotational_log_gain=rotation_gain,
            observation_standard_deviation=likelihood_sigma,
            resource_debt=debts["dual"],
            average_optimism_budget=budget,
            lyapunov_tradeoff=lyapunov_v,
            hard_feasible=calls["dual"] < allowance,
        )
        dual_anchor = dual_decision.use_optimism
        immediate_dual_call = bool(
            calls["dual"] < allowance
            and lyapunov_v * dual_decision.immediate_expected_log_gain
            > debts["dual"]
        )
        information_induced_calls += int(dual_anchor and not immediate_dual_call)

        myopic_gain = expected_binary_log_gain(
            beliefs["myopic"],
            potential_log_gain=potential_gain,
            rotational_log_gain=rotation_gain,
        )
        myopic_anchor = bool(
            calls["myopic"] < allowance
            and lyapunov_v * myopic_gain > debts["myopic"]
        )
        true_gain = rotation_gain if rotation else potential_gain
        exact_decision = choose_log_drift_anchor(
            plain_multiplier=math.exp(true_gain),
            fresh_multiplier=1.0,
            resource_debt=debts["exact_phase"],
            average_anchor_budget=budget,
            lyapunov_tradeoff=lyapunov_v,
            hard_feasible=calls["exact_phase"] < allowance,
        )
        # choose_log_drift_anchor above sees log(q_plain/q_fresh)=true_gain.
        exact_anchor = exact_decision.use_fresh_anchor

        anchors = {
            "dual": dual_anchor,
            "myopic": myopic_anchor,
            "exact_phase": exact_anchor,
        }
        for name, anchor in anchors.items():
            calls[name] += int(anchor)
            debts[name] = max(0.0, debts[name] + float(anchor) - budget)
        potential_calls["dual"] += int(dual_anchor and not rotation)
        potential_calls["myopic"] += int(myopic_anchor and not rotation)

        actions = np.asarray(
            [dual_anchor, myopic_anchor, exact_anchor, False]
            + [event % period in mask for mask in masks],
            dtype=int,
        )
        pre_states = {"dual": states[0].copy(), "myopic": states[1].copy()}
        matrices = table[int(rotation), actions, int(agent)]
        states = np.einsum("nij,nj->ni", matrices, states)
        energies = np.einsum("ni,ij,nj->n", states, metric, states)
        if np.any(~np.isfinite(energies)) or np.any(energies <= 0.0):
            raise RuntimeError("nonfinite or nonpositive clocked-game energy")
        accumulated += np.log(energies)
        states /= np.sqrt(energies)[:, None]

        for name, anchor in (("dual", dual_anchor), ("myopic", myopic_anchor)):
            if anchor:
                beliefs[name], observed = _observe(
                    state=pre_states[name],
                    rotation=bool(rotation),
                    step=step,
                    noise=noise,
                    first_noise=first_noise[event],
                    second_noise=second_noise[event],
                    likelihood_sigma=likelihood_sigma,
                    belief=beliefs[name],
                    minimum_gradient_energy=float(config["minimum_gradient_energy"]),
                )
                informative[name] += int(observed)

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
        "dual_log_energy_rate": float(rates[0]),
        "myopic_log_energy_rate": float(rates[1]),
        "exact_phase_log_energy_rate": float(rates[2]),
        "never_log_energy_rate": float(rates[3]),
        "fixed_mask_log_energy_rates": mask_rates,
        "dual_calls": calls["dual"],
        "myopic_calls": calls["myopic"],
        "exact_phase_calls": calls["exact_phase"],
        "dual_potential_calls": potential_calls["dual"],
        "myopic_potential_calls": potential_calls["myopic"],
        "dual_informative_fingerprints": informative["dual"],
        "myopic_informative_fingerprints": informative["myopic"],
        "information_induced_calls": information_induced_calls,
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
        "fingerprint_noise",
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
            "dual_log_energy_rate",
            "myopic_log_energy_rate",
            "exact_phase_log_energy_rate",
            "never_log_energy_rate",
        ):
            cell[metric] = float(np.mean([row[metric] for row in selected]))
        cell["best_fixed_mask"] = best_mask
        cell["best_fixed_log_energy_rate"] = mask_means[best_mask]
        for prefix in ("dual", "myopic"):
            cell[f"{prefix}_call_fraction"] = float(
                np.mean([row[f"{prefix}_calls"] for row in selected]) / horizon
            )
            cell[f"{prefix}_potential_call_fraction"] = float(
                np.mean([row[f"{prefix}_potential_calls"] for row in selected])
                / horizon
            )
        cell["information_induced_call_fraction"] = float(
            np.mean([row["information_induced_calls"] for row in selected]) / horizon
        )
        cell["separated_dynamic"] = bool(selected[0]["separated_dynamic"])
        exact_gain = (
            cell["best_fixed_log_energy_rate"]
            - cell["exact_phase_log_energy_rate"]
        )
        for prefix in ("dual", "myopic"):
            gain = (
                cell["best_fixed_log_energy_rate"]
                - cell[f"{prefix}_log_energy_rate"]
            )
            cell[f"{prefix}_exact_gain_capture"] = (
                gain / exact_gain if exact_gain > 1e-15 else 1.0
            )
        cells.append(cell)
    return cells


def _mean_gain(cells: list[dict[str, Any]], method: str = "dual") -> float:
    return float(
        np.mean(
            [
                cell["best_fixed_log_energy_rate"]
                - cell[f"{method}_log_energy_rate"]
                for cell in cells
            ]
        )
    )


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    horizon = int(config["horizon"])
    cells = _cells(rows, horizon)
    dynamic = [cell for cell in cells if cell["separated_dynamic"]]
    noise_metrics: dict[str, Any] = {}
    for noise in config["fingerprint_noise_standard_deviations"]:
        selected = [cell for cell in dynamic if cell["fingerprint_noise"] == noise]
        gains = np.asarray(
            [cell["best_fixed_log_energy_rate"] - cell["dual_log_energy_rate"] for cell in selected]
        )
        dual_over_myopic = np.asarray(
            [cell["myopic_log_energy_rate"] - cell["dual_log_energy_rate"] for cell in selected]
        )
        noise_metrics[str(noise)] = {
            "dynamic_cell_count": len(selected),
            "mean_dynamic_gain": float(np.mean(gains)),
            "dynamic_improvement_fraction": float(np.mean(gains > 0.0)),
            "dynamic_contraction_fraction": float(
                np.mean([cell["dual_log_energy_rate"] < 0.0 for cell in selected])
            ),
            "median_exact_gain_capture": float(
                np.median([cell["dual_exact_gain_capture"] for cell in selected])
            ),
            "mean_gain_over_myopic": float(np.mean(dual_over_myopic)),
            "mean_information_induced_call_fraction": float(
                np.mean([cell["information_induced_call_fraction"] for cell in selected])
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
        cell["dual_log_energy_rate"] - cell["never_log_energy_rate"]
        for cell in potential
    )
    maximum_overshoot = max(
        row[f"{prefix}_calls"] - row["allowance"]
        for row in rows
        for prefix in ("dual", "myopic", "exact_phase")
    )
    thresholds = config["development_survival_gates"]
    gates = {
        "E1_finite_and_accounted": all(
            math.isfinite(row[metric])
            for row in rows
            for metric in (
                "dual_log_energy_rate",
                "myopic_log_energy_rate",
                "exact_phase_log_energy_rate",
                "never_log_energy_rate",
            )
        ),
        "E2_each_noise_gain": all(
            value["mean_dynamic_gain"]
            >= thresholds["minimum_each_noise_mean_dynamic_gain"]
            for value in noise_metrics.values()
        ),
        "E3_each_noise_capture": all(
            value["median_exact_gain_capture"]
            >= thresholds["minimum_each_noise_median_exact_gain_capture"]
            for value in noise_metrics.values()
        ),
        "E4_each_noise_cells": all(
            value["dynamic_improvement_fraction"]
            >= thresholds["minimum_each_noise_dynamic_cell_improvement_fraction"]
            for value in noise_metrics.values()
        ),
        "E5_each_arrival": all(
            gain >= thresholds["minimum_each_arrival_mean_dynamic_gain"]
            for gain in arrival_gains.values()
        ),
        "E6_low_persistence": low_persistence_gain
        >= thresholds["minimum_low_persistence_mean_dynamic_gain"],
        "E7_low_budget": low_budget_gain
        >= thresholds["minimum_low_budget_mean_dynamic_gain"],
        "E8_value_of_information": all(
            value["mean_gain_over_myopic"]
            >= thresholds["minimum_each_noise_gain_over_myopic"]
            for value in noise_metrics.values()
        ),
        "E9_information_nontrivial": min(
            value["mean_information_induced_call_fraction"]
            for value in noise_metrics.values()
        )
        >= thresholds["minimum_information_induced_call_fraction"],
        "E10_stationary_potential": potential_loss
        <= thresholds["maximum_stationary_potential_log_rate_loss"],
        "E11_budget": maximum_overshoot <= thresholds["maximum_budget_overshoot"],
        "E12_development_only": not bool(config["formal_evidence"]),
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
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2, sort_keys=True))
    print(f"summary_sha256={_sha256(output)}")


if __name__ == "__main__":
    main()
