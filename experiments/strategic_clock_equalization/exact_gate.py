"""Frozen exact-game gate for strategic-clock equalization.

The environment is a one-state two-agent cooperative Markov game with
Bernoulli policies parameterized by logits.  All asynchronous methods consume
the same actor-arrival and gradient-noise arrays.  The synchronous path is an
ideal reference and is never counted as a causal comparator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


METHODS = (
    "synchronous_reference",
    "raw_async",
    "barrier_fresh_sequential",
    "true_stationary_rate_inverse",
    "ewma_rate_inverse",
    "count_only_debt",
    "lyapunov_strategic_clock",
)

FIXED_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
LYAPUNOV_WEIGHT = 10.0
EWMA_RATE = 0.05


@dataclass(frozen=True)
class Scenario:
    reward: tuple[tuple[float, float], tuple[float, float]]
    initial: tuple[float, float]
    profile: str
    persistence: float
    noise: float
    population: str = "primary"

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.population,
            self.reward,
            self.initial,
            self.profile,
            self.persistence,
            self.noise,
        )


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def expected_reward(logits: np.ndarray, reward: np.ndarray) -> float:
    p, q = sigmoid(logits)
    return float(
        (1.0 - p) * (1.0 - q) * reward[0, 0]
        + (1.0 - p) * q * reward[0, 1]
        + p * (1.0 - q) * reward[1, 0]
        + p * q * reward[1, 1]
    )


def exact_logit_gradient(logits: np.ndarray, reward: np.ndarray) -> np.ndarray:
    p, q = sigmoid(logits)
    first_delta = (reward[1, 0] - reward[0, 0]) * (1.0 - q) + (
        reward[1, 1] - reward[0, 1]
    ) * q
    second_delta = (reward[0, 1] - reward[0, 0]) * (1.0 - p) + (
        reward[1, 1] - reward[1, 0]
    ) * p
    return np.asarray(
        (p * (1.0 - p) * first_delta, q * (1.0 - q) * second_delta),
        dtype=np.float64,
    )


def block_lipschitz_bound(reward: np.ndarray) -> float:
    # max_x x(1-x)|1-2x| = 1/(6 sqrt(3)); each unilateral
    # payoff difference is bounded by the full reward range.
    return float((reward.max() - reward.min()) / (6.0 * np.sqrt(3.0)))


def normalized_reward(logits: np.ndarray, reward: np.ndarray) -> float:
    lower = float(reward.min())
    width = float(reward.max() - lower)
    if width <= 0.0:
        raise ValueError("reward matrix must have positive range")
    return (expected_reward(logits, reward) - lower) / width


def _rng_for(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join((str(seed), *(str(part) for part in parts)))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def availability_path(
    profile: str, horizon: int, persistence: float, seed: int
) -> np.ndarray:
    rng = _rng_for(seed, profile, persistence, "arrivals")
    flip = seed % 2
    if profile == "iid_balanced_control":
        return rng.integers(0, 2, size=horizon, dtype=np.int8)
    if profile == "iid_stationary_80_20":
        preferred = flip
        return np.where(rng.random(horizon) < 0.8, preferred, 1 - preferred).astype(
            np.int8
        )
    if profile == "equal_count_early_burst_reversal":
        base = np.asarray(([0] * 4 + [1]) * (horizon // 10) + ([1] * 4 + [0]) * (horizon // 10), dtype=np.int8)
        if base.size != horizon:
            raise ValueError("horizon must be divisible by ten for reversal profile")
        return base if flip == 0 else 1 - base
    if profile == "symmetric_markov_bursts":
        preferred = flip
        path = np.empty(horizon, dtype=np.int8)
        for event in range(horizon):
            if event and rng.random() > persistence:
                preferred = 1 - preferred
            path[event] = preferred if rng.random() < 0.8 else 1 - preferred
        return path
    raise ValueError(f"unknown profile: {profile}")


def noise_path(
    seed: int, scenario: Scenario, horizon: int
) -> np.ndarray:
    rng = _rng_for(seed, scenario.key, "gradient-noise")
    return rng.normal(0.0, scenario.noise, size=(horizon, 2))


def stationary_rates(profile: str, seed: int) -> np.ndarray:
    if profile == "iid_stationary_80_20":
        return np.asarray((0.2, 0.8) if seed % 2 else (0.8, 0.2))
    return np.asarray((0.5, 0.5))


def _clip_mass(value: float, maximum: float) -> float:
    return float(np.clip(value, 0.0, maximum))


def _simulate(
    scenario: Scenario,
    seed: int,
    method: str,
    *,
    horizon: int,
    base_step: float,
    maximum_mass: float,
    fixed_multipliers: tuple[float, float] | None = None,
) -> dict[str, object]:
    reward = np.asarray(scenario.reward, dtype=np.float64)
    logits = np.log(np.asarray(scenario.initial) / (1.0 - np.asarray(scenario.initial)))
    arrivals = availability_path(scenario.profile, horizon, scenario.persistence, seed)
    noises = noise_path(seed, scenario, horizon)
    common_increment = base_step / 2.0
    debt = np.zeros(2, dtype=np.float64)
    rate_hat = np.full(2, 0.5, dtype=np.float64)
    pending_gradient: list[float | None] = [None, None]
    values = np.empty(horizon + 1, dtype=np.float64)
    values[0] = normalized_reward(logits, reward)
    masses = np.zeros(horizon, dtype=np.float64)
    debt_differences = 0
    performance_active = 0
    lipschitz = block_lipschitz_bound(reward)

    for event, actor_value in enumerate(arrivals):
        actor = int(actor_value)
        current_gradient = exact_logit_gradient(logits, reward)

        if method == "synchronous_reference":
            for index in range(2):
                logits[index] += common_increment * (
                    current_gradient[index] + noises[event, index]
                )
            masses[event] = base_step
        elif method == "barrier_fresh_sequential":
            pending_gradient[actor] = float(current_gradient[actor] + noises[event, actor])
            if pending_gradient[0] is not None and pending_gradient[1] is not None:
                for index in range(2):
                    logits[index] += base_step * float(pending_gradient[index])
                    masses[event] += base_step
                    pending_gradient[index] = None
        else:
            observed = float(current_gradient[actor] + noises[event, actor])
            if method == "raw_async":
                mass = base_step
            elif method == "true_stationary_rate_inverse":
                rates = stationary_rates(scenario.profile, seed)
                mass = base_step / (2.0 * float(rates[actor]))
            elif method == "ewma_rate_inverse":
                rate_hat *= 1.0 - EWMA_RATE
                rate_hat[actor] += EWMA_RATE
                mass = base_step / (2.0 * float(rate_hat[actor]))
            elif method == "fixed_block_scaling":
                if fixed_multipliers is None:
                    raise ValueError("fixed multipliers required")
                mass = base_step * fixed_multipliers[actor]
            elif method in {"count_only_debt", "lyapunov_strategic_clock"}:
                debt += common_increment
                count_mass = _clip_mass(debt[actor], maximum_mass)
                if method == "count_only_debt":
                    mass = count_mass
                else:
                    signal = max(observed * observed - scenario.noise**2, 0.0)
                    numerator = debt[actor] + LYAPUNOV_WEIGHT * signal
                    denominator = 1.0 + LYAPUNOV_WEIGHT * lipschitz * signal
                    mass = _clip_mass(numerator / denominator, maximum_mass)
                    if abs(mass - count_mass) > 1e-12:
                        debt_differences += 1
                    if signal > 0.0:
                        performance_active += 1
                debt[actor] -= mass
            else:
                raise ValueError(f"unknown method: {method}")
            mass = _clip_mass(mass, maximum_mass)
            logits[actor] += mass * observed
            masses[event] = mass

        values[event + 1] = normalized_reward(logits, reward)

    return {
        "auc": float(values.mean()),
        "terminal": float(values[-1]),
        "total_mass": float(masses.sum()),
        "maximum_mass": float(masses.max(initial=0.0)),
        "terminal_debt_l1": float(np.abs(debt).sum()),
        "mass_difference_fraction": float(debt_differences / horizon),
        "performance_active_fraction": float(performance_active / horizon),
        "arrival_counts": [int((arrivals == i).sum()) for i in range(2)],
    }


def scenarios_from_manifest(manifest: dict[str, object]) -> list[Scenario]:
    rewards = manifest["environment"]["reward_matrices"]
    initials = manifest["environment"]["initial_action1_probabilities"]
    profiles = manifest["availability"]["profiles"]
    persistences = manifest["availability"]["markov_persistence"]
    noises = manifest["gradient"]["noise_standard_deviation"]
    result: list[Scenario] = []
    for reward in rewards:
        reward_tuple = tuple(tuple(float(v) for v in row) for row in reward)
        for initial in initials:
            for profile in profiles:
                relevant_persistences = persistences if profile == "symmetric_markov_bursts" else (persistences[0],)
                for persistence in relevant_persistences:
                    for noise in noises:
                        population = (
                            "balanced_control"
                            if profile == "iid_balanced_control"
                            else "primary"
                        )
                        result.append(
                            Scenario(
                                reward=reward_tuple,
                                initial=tuple(float(v) for v in initial),
                                profile=str(profile),
                                persistence=float(persistence),
                                noise=float(noise),
                                population=population,
                            )
                        )

    # Same-side controls use the registered reward matrices and balanced plus
    # reversal profiles.  They are generated mechanically, not selected from
    # outcomes.
    for reward in rewards:
        reward_tuple = tuple(tuple(float(v) for v in row) for row in reward)
        for initial in ((0.35, 0.35), (0.65, 0.65)):
            for profile in ("iid_balanced_control", "equal_count_early_burst_reversal"):
                result.append(
                    Scenario(
                        reward=reward_tuple,
                        initial=initial,
                        profile=profile,
                        persistence=float(persistences[0]),
                        noise=0.0,
                        population="single_basin_control",
                    )
                )

    # Additive reward removes strategic coupling while retaining two policy
    # blocks and the same arrival process.
    additive = ((0.0, 1.0), (1.0, 2.0))
    for profile in ("iid_balanced_control", "equal_count_early_burst_reversal"):
        result.append(
            Scenario(
                reward=additive,
                initial=(0.35, 0.55),
                profile=profile,
                persistence=float(persistences[0]),
                noise=0.0,
                population="uncoupled_control",
            )
        )
    return result


def _mean(items: Iterable[float]) -> float:
    values = tuple(float(item) for item in items)
    return float(np.mean(values)) if values else float("nan")


def _median(items: Iterable[float]) -> float:
    values = tuple(float(item) for item in items)
    return float(np.median(values)) if values else float("nan")


def run_gate(manifest_path: Path) -> dict[str, object]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    horizon = int(manifest["availability"]["horizon_events"])
    seeds = tuple(int(seed) for seed in manifest["availability"]["seeds"])
    base_step = float(manifest["gradient"]["base_step"])
    maximum_mass = float(manifest["gradient"]["maximum_update_mass"])
    scenarios = scenarios_from_manifest(manifest)

    rows: list[dict[str, object]] = []
    fixed_candidates: dict[tuple[object, ...], tuple[float, float]] = {}
    for scenario in scenarios:
        candidate_means: list[tuple[float, tuple[float, float]]] = []
        for first in FIXED_MULTIPLIERS:
            for second in FIXED_MULTIPLIERS:
                aucs = []
                for seed in seeds:
                    result = _simulate(
                        scenario,
                        seed,
                        "fixed_block_scaling",
                        horizon=horizon,
                        base_step=base_step,
                        maximum_mass=maximum_mass,
                        fixed_multipliers=(first, second),
                    )
                    aucs.append(float(result["auc"]))
                candidate_means.append((_mean(aucs), (first, second)))
        fixed_candidates[scenario.key] = max(candidate_means, key=lambda item: item[0])[1]

        for seed in seeds:
            for method in METHODS:
                result = _simulate(
                    scenario,
                    seed,
                    method,
                    horizon=horizon,
                    base_step=base_step,
                    maximum_mass=maximum_mass,
                )
                rows.append(
                    {
                        "scenario": scenario.key,
                        "seed": seed,
                        "method": method,
                        **result,
                    }
                )
            fixed = fixed_candidates[scenario.key]
            result = _simulate(
                scenario,
                seed,
                "fixed_block_scaling",
                horizon=horizon,
                base_step=base_step,
                maximum_mass=maximum_mass,
                fixed_multipliers=fixed,
            )
            rows.append(
                {
                    "scenario": scenario.key,
                    "seed": seed,
                    "method": "best_fixed_block_scaling",
                    "fixed_multipliers": list(fixed),
                    **result,
                }
            )

    by_cell: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        by_cell.setdefault(tuple(row["scenario"]), []).append(row)

    cell_summaries: list[dict[str, object]] = []
    causal_methods = (
        "raw_async",
        "barrier_fresh_sequential",
        "true_stationary_rate_inverse",
        "ewma_rate_inverse",
        "count_only_debt",
        "best_fixed_block_scaling",
    )
    for key, cell_rows in by_cell.items():
        means = {
            method: _mean(
                float(row["auc"]) for row in cell_rows if row["method"] == method
            )
            for method in (*METHODS, "best_fixed_block_scaling")
        }
        strong_method = max(causal_methods, key=lambda method: means[method])
        loss = means["synchronous_reference"] - means["raw_async"]
        active = key[0] == "primary" and loss >= 0.05
        recovery = (
            (means["lyapunov_strategic_clock"] - means["raw_async"]) / loss
            if active
            else None
        )
        additional = (
            (means["lyapunov_strategic_clock"] - means[strong_method]) / loss
            if active
            else None
        )
        lyapunov_rows = [
            row for row in cell_rows if row["method"] == "lyapunov_strategic_clock"
        ]
        cell_summaries.append(
            {
                "scenario": key,
                "mean_auc": means,
                "strong_method": strong_method,
                "raw_to_sync_loss": loss,
                "headroom_active": active,
                "recovery": recovery,
                "additional_recovery": additional,
                "lyapunov_terminal_debt_l1": _mean(
                    float(row["terminal_debt_l1"]) for row in lyapunov_rows
                ),
                "mass_difference_fraction": _mean(
                    float(row["mass_difference_fraction"]) for row in lyapunov_rows
                ),
                "performance_active_fraction": _mean(
                    float(row["performance_active_fraction"]) for row in lyapunov_rows
                ),
            }
        )

    active_cells = [cell for cell in cell_summaries if cell["headroom_active"]]
    primary_cells = [cell for cell in cell_summaries if cell["scenario"][0] == "primary"]
    balanced_controls = [cell for cell in cell_summaries if cell["scenario"][0] == "balanced_control"]
    single_controls = [cell for cell in cell_summaries if cell["scenario"][0] in {"single_basin_control", "uncoupled_control"}]

    median_primary_loss = _median(cell["raw_to_sync_loss"] for cell in primary_cells)
    median_recovery = _median(cell["recovery"] for cell in active_cells)
    median_additional = _median(cell["additional_recovery"] for cell in active_cells)
    directional = _mean(
        float(cell["additional_recovery"] > 0.0) for cell in active_cells
    )
    homogeneous_loss = max(
        (
            max(cell["mean_auc"][method] for method in causal_methods)
            - cell["mean_auc"]["lyapunov_strategic_clock"]
            for cell in balanced_controls
        ),
        default=0.0,
    )
    single_loss = max(
        (
            max(cell["mean_auc"][method] for method in causal_methods)
            - cell["mean_auc"]["lyapunov_strategic_clock"]
            for cell in single_controls
        ),
        default=0.0,
    )
    difference_fraction = _mean(
        cell["mass_difference_fraction"] for cell in primary_cells
    )
    performance_fraction = _mean(
        cell["performance_active_fraction"] for cell in primary_cells
    )
    maximum_mass_seen = max(float(row["maximum_mass"]) for row in rows)
    max_debt_per_event = max(
        cell["lyapunov_terminal_debt_l1"] / horizon for cell in primary_cells
    )

    gates = {
        "G1_validity": bool(
            all(np.isfinite(float(row["auc"])) for row in rows)
            and maximum_mass_seen <= maximum_mass + 1e-12
        ),
        "G2_raw_headroom": bool(median_primary_loss >= 0.05),
        "G3_recovery": bool(active_cells and median_recovery >= 0.50),
        "G4_strong_baseline": bool(active_cells and median_additional >= 0.10),
        "G5_directionality": bool(active_cells and directional >= 0.60),
        "G6_homogeneous_control": bool(homogeneous_loss <= 0.01),
        "G7_single_basin_control": bool(single_loss <= 0.01),
        "G8_nontriviality": bool(
            difference_fraction >= 0.10 and performance_fraction > 0.0
        ),
        "G9_queue_feasibility": bool(
            maximum_mass_seen <= maximum_mass + 1e-12 and max_debt_per_event <= 0.01
        ),
    }
    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "scenario_count": len(scenarios),
        "row_count": len(rows),
        "active_cell_count": len(active_cells),
        "metrics": {
            "median_primary_raw_to_sync_loss": median_primary_loss,
            "median_active_recovery": median_recovery,
            "median_active_additional_recovery": median_additional,
            "active_directional_fraction": directional,
            "maximum_balanced_control_loss": homogeneous_loss,
            "maximum_single_or_uncoupled_control_loss": single_loss,
            "mean_primary_mass_difference_fraction": difference_fraction,
            "mean_primary_performance_active_fraction": performance_fraction,
            "maximum_mass_seen": maximum_mass_seen,
            "maximum_terminal_debt_per_event": max_debt_per_event,
        },
        "gates": gates,
        "all_pre_reproduction_gates_pass": all(gates.values()),
        "fixed_candidates": [
            {"scenario": key, "multipliers": value}
            for key, value in fixed_candidates.items()
        ],
        "cells": cell_summaries,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_gate(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("scenario_count", "row_count", "active_cell_count", "metrics", "gates", "all_pre_reproduction_gates_pass")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

