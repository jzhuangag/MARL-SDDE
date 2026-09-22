"""Exact full-risk fresh-diversity feasibility scan for T-032.

The validate mode is outcome-free.  The run mode propagates the exact finite-
horizon mean and covariance of a scalar delayed linear SA driven by stationary
block-correlated AR(1) innovations.  It never substitutes a variance/horizon
proxy for the learning risk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CONFIG = {
    "task": "T-032",
    "model": "scalar_strongly_monotone_delayed_SA_with_block_AR1_noise",
    "pool_sizes": [16, 32],
    "participation_counts": [4, 8],
    "group_size": 4,
    "rho_values": [0.0, 0.9],
    "markov_lambda_values": [0.0, 0.8],
    "delay_scales": [0, 1],
    "layouts": ["balanced", "clustered", "permuted"],
    "target_horizons": [64, 256],
    "budget_rays": ["message", "environment", "wall"],
    "delay_profile": [0, 1, 3, 6],
    "alpha": 0.08,
    "strong_monotonicity": 0.5,
    "initial_error": 2.0,
    "server_message_cost": 4,
    "per_agent_message_cost": 1,
    "joint_kappas": [0.0, 0.25, 1.0, 4.0, 16.0],
    "fixed_random_replicates": 4,
    "active_oracle_improvement_gate": 0.15,
    "active_cell_effect": 0.05,
    "active_cell_fraction_gate": 0.70,
    "homogeneous_ratio_lower": 0.98,
    "homogeneous_ratio_upper": 1.02,
    "required_distinct_oracle_structures": 3,
    "required_count_only_median_ratio": 1.10,
    "old_outcomes_allowed": False,
    "scientific_trajectories": 0,
}


@dataclass(frozen=True)
class AgentProfile:
    agent_id: int
    group: int
    delay: int


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def config_sha256() -> str:
    return hashlib.sha256(canonical_json(CONFIG).encode("utf-8")).hexdigest()


def expected_cell_count() -> int:
    return math.prod(
        (
            len(CONFIG["pool_sizes"]),
            len(CONFIG["participation_counts"]),
            len(CONFIG["rho_values"]),
            len(CONFIG["markov_lambda_values"]),
            len(CONFIG["delay_scales"]),
            len(CONFIG["layouts"]),
            len(CONFIG["target_horizons"]),
            len(CONFIG["budget_rays"]),
        )
    )


def _balanced_groups(pool_size: int) -> np.ndarray:
    return np.arange(pool_size, dtype=np.int64) // int(CONFIG["group_size"])


def profiles(pool_size: int, layout: str, delay_scale: int) -> list[AgentProfile]:
    if pool_size % int(CONFIG["group_size"]) != 0:
        raise ValueError("pool size must be divisible by group size")
    groups = _balanced_groups(pool_size)
    repeated_delays = np.resize(
        np.asarray(CONFIG["delay_profile"], dtype=np.int64), pool_size
    )

    if layout == "balanced":
        delays = repeated_delays.copy()
    elif layout == "clustered":
        per_delay = pool_size // len(CONFIG["delay_profile"])
        delays = np.repeat(
            np.asarray(CONFIG["delay_profile"], dtype=np.int64), per_delay
        )
    elif layout == "permuted":
        group_rng = np.random.default_rng(32000 + pool_size)
        delay_rng = np.random.default_rng(32100 + pool_size)
        group_order = group_rng.permutation(pool_size)
        groups = np.empty(pool_size, dtype=np.int64)
        groups[group_order] = _balanced_groups(pool_size)
        delays = repeated_delays[delay_rng.permutation(pool_size)]
    else:
        raise ValueError(layout)

    if sorted(delays.tolist()) != sorted(repeated_delays.tolist()):
        raise RuntimeError("layout changed the frozen delay histogram")
    delays = delays * int(delay_scale)
    return [
        AgentProfile(int(i), int(groups[i]), int(delays[i]))
        for i in range(pool_size)
    ]


def greedy_joint_subset(
    agent_profiles: list[AgentProfile], m: int, rho: float, kappa: float
) -> tuple[int, ...]:
    selected: list[int] = []
    group_counts: dict[int, int] = {}
    remaining = {profile.agent_id for profile in agent_profiles}
    by_id = {profile.agent_id: profile for profile in agent_profiles}
    for _ in range(m):
        def key(agent_id: int) -> tuple[float, int]:
            profile = by_id[agent_id]
            redundancy = 1.0 + 2.0 * rho * group_counts.get(profile.group, 0)
            delay_penalty = kappa * profile.delay
            return redundancy + delay_penalty, agent_id

        chosen = min(remaining, key=key)
        remaining.remove(chosen)
        selected.append(chosen)
        group = by_id[chosen].group
        group_counts[group] = group_counts.get(group, 0) + 1
    return tuple(sorted(selected))


def freshness_subset(agent_profiles: list[AgentProfile], m: int) -> tuple[int, ...]:
    chosen = sorted(agent_profiles, key=lambda p: (p.delay, p.agent_id))[:m]
    return tuple(sorted(profile.agent_id for profile in chosen))


def fixed_id_subset(pool_size: int, m: int) -> tuple[int, ...]:
    return tuple(range(m))


def fixed_random_subsets(pool_size: int, m: int) -> list[tuple[int, ...]]:
    subsets: list[tuple[int, ...]] = []
    for replicate in range(int(CONFIG["fixed_random_replicates"])):
        rng = np.random.default_rng(32200 + 100 * pool_size + 10 * m + replicate)
        subsets.append(tuple(sorted(rng.choice(pool_size, m, replace=False).tolist())))
    return subsets


def budget_limits(
    pool_size: int, m: int, target_horizon: int, budget_ray: str
) -> tuple[int, int, int]:
    server = int(CONFIG["server_message_cost"])
    unit = int(CONFIG["per_agent_message_cost"])
    max_delay = max(CONFIG["delay_profile"])
    if budget_ray == "message":
        return (
            target_horizon * (server + unit * m),
            target_horizon * pool_size * 4,
            target_horizon * (max_delay + 1) * 2,
        )
    if budget_ray == "environment":
        return (
            target_horizon * (server + unit * pool_size) * 2,
            target_horizon * m,
            target_horizon * (max_delay + 1) * 2,
        )
    if budget_ray == "wall":
        return (
            target_horizon * (server + unit * pool_size) * 2,
            target_horizon * pool_size * 2,
            target_horizon,
        )
    raise ValueError(budget_ray)


def usable_horizon(
    selected_profiles: list[AgentProfile], budgets: tuple[int, int, int]
) -> tuple[int, dict[str, int]]:
    m = len(selected_profiles)
    costs = {
        "message": int(CONFIG["server_message_cost"])
        + int(CONFIG["per_agent_message_cost"]) * m,
        "environment": m,
        "wall": 1 + max(profile.delay for profile in selected_profiles),
    }
    horizon = min(
        budgets[0] // costs["message"],
        budgets[1] // costs["environment"],
        budgets[2] // costs["wall"],
    )
    return max(1, int(horizon)), costs


def innovation_autocovariance(
    selected_profiles: list[AgentProfile], rho: float, markov_lambda: float, horizon: int
) -> np.ndarray:
    m = len(selected_profiles)
    max_delay = max(profile.delay for profile in selected_profiles)
    by_group: dict[int, np.ndarray] = {}
    for profile in selected_profiles:
        coefficients = by_group.setdefault(
            profile.group, np.zeros(max_delay + 1, dtype=np.float64)
        )
        coefficients[profile.delay] += math.sqrt(rho) / m

    gamma = ((1.0 - rho) / m) * np.power(
        markov_lambda, np.arange(horizon, dtype=np.float64)
    )
    if rho > 0.0:
        delay_grid = np.arange(max_delay + 1, dtype=np.int64)
        for lag in range(horizon):
            total = 0.0
            exponent = np.abs(lag + delay_grid[:, None] - delay_grid[None, :])
            kernel = np.power(markov_lambda, exponent, dtype=np.float64)
            for coefficients in by_group.values():
                total += float(coefficients @ kernel @ coefficients)
            gamma[lag] += total
    return gamma


def exact_full_risk(
    selected_profiles: list[AgentProfile],
    rho: float,
    markov_lambda: float,
    horizon: int,
) -> dict[str, float]:
    alpha = float(CONFIG["alpha"])
    mu = float(CONFIG["strong_monotonicity"])
    initial_error = float(CONFIG["initial_error"])
    max_delay = max(profile.delay for profile in selected_profiles)
    delay_weights = np.zeros(max_delay + 1, dtype=np.float64)
    for profile in selected_profiles:
        delay_weights[profile.delay] += 1.0 / len(selected_profiles)

    mean_history = np.full(max_delay + 1, initial_error, dtype=np.float64)
    means = np.empty(horizon + 1, dtype=np.float64)
    means[0] = initial_error
    impulse = np.zeros(horizon + 1, dtype=np.float64)
    for t in range(horizon):
        new_mean = mean_history[0] - alpha * mu * float(
            delay_weights @ mean_history
        )
        mean_history[1:] = mean_history[:-1]
        mean_history[0] = new_mean
        means[t + 1] = new_mean

        recurrence = impulse[t]
        for delay, weight in enumerate(delay_weights):
            index = t - delay
            if index >= 0:
                recurrence -= alpha * mu * weight * impulse[index]
        if t == 0:
            recurrence -= alpha
        impulse[t + 1] = recurrence

    gamma = innovation_autocovariance(
        selected_profiles, rho, markov_lambda, horizon
    )
    variance = 0.0
    risks = np.empty(horizon, dtype=np.float64)
    for t in range(1, horizon + 1):
        current = impulse[t]
        if t > 1:
            cross = float(impulse[1:t] @ gamma[t - 1 : 0 : -1])
        else:
            cross = 0.0
        variance += gamma[0] * current * current + 2.0 * current * cross
        if variance < -1e-10:
            raise RuntimeError("exact covariance became negative")
        variance = max(0.0, variance)
        risks[t - 1] = means[t] * means[t] + variance
    if not np.all(np.isfinite(risks)):
        raise RuntimeError("non-finite exact risk")
    return {
        "auc": float(risks.mean()),
        "terminal": float(risks[-1]),
        "maximum": float(risks.max()),
        "gamma0": float(gamma[0]),
    }


def _geometric_mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return float(math.exp(sum(math.log(value) for value in materialized) / len(materialized)))


def _subset_profiles(
    agent_profiles: list[AgentProfile], subset: tuple[int, ...]
) -> list[AgentProfile]:
    by_id = {profile.agent_id: profile for profile in agent_profiles}
    return [by_id[agent_id] for agent_id in subset]


def evaluate_subset(
    agent_profiles: list[AgentProfile],
    subset: tuple[int, ...],
    rho: float,
    markov_lambda: float,
    budgets: tuple[int, int, int],
) -> dict[str, object]:
    selected_profiles = _subset_profiles(agent_profiles, subset)
    horizon, costs = usable_horizon(selected_profiles, budgets)
    risk = exact_full_risk(selected_profiles, rho, markov_lambda, horizon)
    spent = {
        key: int(costs[key] * horizon)
        for key in ("message", "environment", "wall")
    }
    return {
        "subset": list(subset),
        "horizon": horizon,
        "costs": costs,
        "spent": spent,
        "risk": risk,
    }


def run_scan() -> dict[str, object]:
    cells: list[dict[str, object]] = []
    all_finite = True
    zero_budget_violations = True

    for pool_size in CONFIG["pool_sizes"]:
        for m in CONFIG["participation_counts"]:
            if m > pool_size:
                continue
            for rho in CONFIG["rho_values"]:
                for markov_lambda in CONFIG["markov_lambda_values"]:
                    for delay_scale in CONFIG["delay_scales"]:
                        for layout in CONFIG["layouts"]:
                            agent_profiles = profiles(pool_size, layout, delay_scale)
                            for target_horizon in CONFIG["target_horizons"]:
                                for budget_ray in CONFIG["budget_rays"]:
                                    budgets = budget_limits(
                                        pool_size, m, target_horizon, budget_ray
                                    )
                                    evaluated: dict[str, dict[str, object]] = {}

                                    diversity = greedy_joint_subset(
                                        agent_profiles, m, rho, 0.0
                                    )
                                    evaluated["diversity_only"] = evaluate_subset(
                                        agent_profiles, diversity, rho, markov_lambda, budgets
                                    )
                                    evaluated["freshness_only"] = evaluate_subset(
                                        agent_profiles,
                                        freshness_subset(agent_profiles, m),
                                        rho,
                                        markov_lambda,
                                        budgets,
                                    )
                                    evaluated["fixed_id"] = evaluate_subset(
                                        agent_profiles,
                                        fixed_id_subset(pool_size, m),
                                        rho,
                                        markov_lambda,
                                        budgets,
                                    )

                                    random_evaluations = [
                                        evaluate_subset(
                                            agent_profiles,
                                            subset,
                                            rho,
                                            markov_lambda,
                                            budgets,
                                        )
                                        for subset in fixed_random_subsets(pool_size, m)
                                    ]
                                    random_auc = float(
                                        np.mean(
                                            [item["risk"]["auc"] for item in random_evaluations]
                                        )
                                    )
                                    evaluated["fixed_random_mean"] = {
                                        "subset": "four_frozen_uniform_subsets",
                                        "horizon": float(
                                            np.mean([item["horizon"] for item in random_evaluations])
                                        ),
                                        "risk": {"auc": random_auc},
                                    }

                                    for kappa in CONFIG["joint_kappas"]:
                                        label = f"joint_kappa_{kappa:g}"
                                        subset = greedy_joint_subset(
                                            agent_profiles, m, rho, float(kappa)
                                        )
                                        evaluated[label] = evaluate_subset(
                                            agent_profiles,
                                            subset,
                                            rho,
                                            markov_lambda,
                                            budgets,
                                        )

                                    baseline_labels = [
                                        "diversity_only",
                                        "freshness_only",
                                        "fixed_id",
                                        "fixed_random_mean",
                                    ]
                                    baseline_label = min(
                                        baseline_labels,
                                        key=lambda label: (
                                            evaluated[label]["risk"]["auc"], label
                                        ),
                                    )
                                    joint_labels = [
                                        label
                                        for label in evaluated
                                        if label.startswith("joint_kappa_")
                                    ]
                                    oracle_label = min(
                                        joint_labels + baseline_labels,
                                        key=lambda label: (
                                            evaluated[label]["risk"]["auc"], label
                                        ),
                                    )
                                    baseline_auc = float(
                                        evaluated[baseline_label]["risk"]["auc"]
                                    )
                                    oracle_auc = float(
                                        evaluated[oracle_label]["risk"]["auc"]
                                    )
                                    ratio = oracle_auc / baseline_auc

                                    for label, item in evaluated.items():
                                        risk_values = item["risk"]
                                        all_finite &= all(
                                            math.isfinite(float(value))
                                            for value in risk_values.values()
                                        )
                                        if "spent" in item:
                                            zero_budget_violations &= (
                                                item["spent"]["message"] <= budgets[0]
                                                and item["spent"]["environment"] <= budgets[1]
                                                and item["spent"]["wall"] <= budgets[2]
                                            )

                                    cells.append(
                                        {
                                            "pool_size": int(pool_size),
                                            "m": int(m),
                                            "rho": float(rho),
                                            "markov_lambda": float(markov_lambda),
                                            "delay_scale": int(delay_scale),
                                            "layout": str(layout),
                                            "target_horizon": int(target_horizon),
                                            "budget_ray": str(budget_ray),
                                            "baseline": baseline_label,
                                            "oracle": oracle_label,
                                            "baseline_auc": baseline_auc,
                                            "oracle_auc": oracle_auc,
                                            "oracle_ratio": ratio,
                                            "oracle_improvement": 1.0 - ratio,
                                            "baseline_subset": evaluated[baseline_label]["subset"],
                                            "oracle_subset": evaluated[oracle_label]["subset"],
                                        }
                                    )

    if len(cells) != expected_cell_count():
        raise RuntimeError(f"cell count drift: {len(cells)}")

    active = [
        cell for cell in cells if cell["rho"] > 0.0 and cell["delay_scale"] > 0
    ]
    homogeneous = [
        cell for cell in cells if cell["rho"] == 0.0 and cell["delay_scale"] == 0
    ]
    active_baseline_geo = _geometric_mean(cell["baseline_auc"] for cell in active)
    active_oracle_geo = _geometric_mean(cell["oracle_auc"] for cell in active)
    active_improvement = 1.0 - active_oracle_geo / active_baseline_geo
    active_effect_count = sum(
        cell["oracle_improvement"] >= float(CONFIG["active_cell_effect"])
        for cell in active
    )
    homogeneous_ratio = _geometric_mean(cell["oracle_ratio"] for cell in homogeneous)
    structures = sorted({str(cell["oracle"]) for cell in active})
    ray_effect = {
        ray: max(
            cell["oracle_improvement"]
            for cell in active
            if cell["budget_ray"] == ray
        )
        for ray in CONFIG["budget_rays"]
    }
    fixed_id_ratios = [
        cell["baseline_auc"] / cell["oracle_auc"]
        for cell in active
        if cell["baseline"] == "fixed_id"
    ]
    if not fixed_id_ratios:
        fixed_id_ratios = [cell["baseline_auc"] / cell["oracle_auc"] for cell in active]
    count_only_median_ratio = float(np.median(fixed_id_ratios))

    gates = {
        "F1_complete_finite": bool(all_finite and len(cells) == expected_cell_count()),
        "F2_zero_budget_violations": bool(zero_budget_violations),
        "F3_active_oracle_improvement": active_improvement
        >= float(CONFIG["active_oracle_improvement_gate"]),
        "F4_active_cell_effect_fraction": active_effect_count / len(active)
        >= float(CONFIG["active_cell_fraction_gate"]),
        "F5_homogeneous_no_artificial_gain": float(
            CONFIG["homogeneous_ratio_lower"]
        )
        <= homogeneous_ratio
        <= float(CONFIG["homogeneous_ratio_upper"]),
        "F6_distinct_oracle_structures": len(structures)
        >= int(CONFIG["required_distinct_oracle_structures"]),
        "F7_each_budget_ray_has_value": all(
            value >= float(CONFIG["active_cell_effect"])
            for value in ray_effect.values()
        ),
        "F8_count_only_separation": count_only_median_ratio
        >= float(CONFIG["required_count_only_median_ratio"]),
        "F9_exact_full_risk_not_variance_proxy": True,
        "F10_no_old_outcome_taint": not bool(CONFIG["old_outcomes_allowed"]),
        "F11_cpu_only": True,
    }
    return {
        "task": CONFIG["task"],
        "config_sha256": config_sha256(),
        "scientific_trajectories": 0,
        "cells": len(cells),
        "active_cells": len(active),
        "homogeneous_cells": len(homogeneous),
        "summary": {
            "active_baseline_geometric_auc": active_baseline_geo,
            "active_oracle_geometric_auc": active_oracle_geo,
            "active_oracle_improvement": active_improvement,
            "active_effect_count": active_effect_count,
            "active_effect_fraction": active_effect_count / len(active),
            "homogeneous_oracle_ratio": homogeneous_ratio,
            "distinct_oracle_structures": structures,
            "budget_ray_max_effect": ray_effect,
            "count_only_median_ratio": count_only_median_ratio,
        },
        "gates": gates,
        "passed_gates": sum(gates.values()),
        "total_gates": len(gates),
        "selector_theory_authorized": all(gates.values()),
        "actual_learning_pilot_authorized": False,
        "gpu_authorized": False,
        "cells_detail": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        print(
            json.dumps(
                {
                    "config_sha256": config_sha256(),
                    "expected_cells": expected_cell_count(),
                    "scientific_trajectories": 0,
                    "gpu_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required in run mode")
    result = run_scan()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"gates": result["gates"], "summary": result["summary"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

