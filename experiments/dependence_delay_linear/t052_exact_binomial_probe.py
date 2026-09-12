"""Exact two-agent fingerprint decision risk under trajectory switches."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np

from experiments.dependence_delay_linear.t050_stationary_break_even import (
    asymptotic_participation_coefficient,
)
from experiments.dependence_delay_linear.t051_fingerprint_probe import (
    plug_in_action,
    trajectory_switch_match_probability,
)


def binomial_probabilities(trials: int, probability: float) -> np.ndarray:
    """Numerically stable complete Binomial(trials, probability) mass vector."""

    if trials < 1 or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid trials or probability")
    masses = np.zeros(trials + 1)
    if probability == 0.0:
        masses[0] = 1.0
        return masses
    if probability == 1.0:
        masses[-1] = 1.0
        return masses
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    for successes in range(trials + 1):
        log_mass = (
            math.lgamma(trials + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(trials - successes + 1)
            + successes * log_probability
            + (trials - successes) * log_complement
        )
        masses[successes] = math.exp(log_mass)
    masses /= np.sum(masses)
    return masses


def exact_plugin_action_distribution(
    *,
    rho: float,
    candidates: Iterable[int],
    overhead: float,
    blocks: int,
    collision_probability: float,
) -> dict[int, float]:
    """Exact distribution of the plug-in fixed-q action for q_probe=2."""

    if blocks < 1 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid blocks or rho")
    if not 0.0 <= collision_probability < 1.0:
        raise ValueError("invalid collision probability")
    actions = sorted({int(q) for q in candidates})
    if not actions or actions[0] < 1:
        raise ValueError("candidates must be positive")
    match_probability = trajectory_switch_match_probability(
        rho=rho, collision_probability=collision_probability
    )
    masses = binomial_probabilities(blocks, match_probability)
    distribution = {q: 0.0 for q in actions}
    for matches, mass in enumerate(masses):
        raw_estimate = (
            matches / blocks - collision_probability
        ) / (1.0 - collision_probability)
        estimate = float(np.clip(raw_estimate, 0.0, 1.0))
        selected = plug_in_action(estimate, actions, overhead=overhead)
        distribution[selected] += float(mass)
    return distribution


def exact_plugin_expected_coefficient(
    *,
    rho: float,
    candidates: Iterable[int],
    overhead: float,
    blocks: int,
    collision_probability: float,
) -> dict[str, float | int | dict[int, float]]:
    """Exact expected stationary coefficient of the two-agent plug-in rule."""

    actions = sorted({int(q) for q in candidates})
    distribution = exact_plugin_action_distribution(
        rho=rho,
        candidates=actions,
        overhead=overhead,
        blocks=blocks,
        collision_probability=collision_probability,
    )
    coefficients = {
        q: asymptotic_participation_coefficient(q, overhead=overhead, rho=rho)
        for q in actions
    }
    oracle_q = min(actions, key=lambda q: (coefficients[q], q))
    expected = sum(distribution[q] * coefficients[q] for q in actions)
    return {
        "oracle_q": int(oracle_q),
        "oracle_coefficient": float(coefficients[oracle_q]),
        "expected_coefficient": float(expected),
        "action_distribution": distribution,
    }


def exact_full_cost_plugin_ratio(
    *,
    rho: float,
    candidates: Iterable[int],
    overhead: float,
    baseline_q: int,
    learning_budget: float,
    probe_blocks: int,
    collision_probability: float,
) -> dict[str, float | int | dict[int, float]]:
    """Exact leading expected-risk ratio for a two-agent fingerprint probe."""

    if learning_budget <= 0.0 or probe_blocks < 1:
        raise ValueError("invalid learning budget or probe block count")
    result = exact_plugin_expected_coefficient(
        rho=rho,
        candidates=candidates,
        overhead=overhead,
        blocks=probe_blocks,
        collision_probability=collision_probability,
    )
    probe_message_cost = probe_blocks * (overhead + 2.0)
    total_message_budget = learning_budget + probe_message_cost
    baseline = asymptotic_participation_coefficient(
        baseline_q, overhead=overhead, rho=rho
    )
    ratio = (
        float(result["expected_coefficient"])
        / learning_budget
        / (baseline / total_message_budget)
    )
    return {
        **result,
        "baseline_q": int(baseline_q),
        "probe_message_cost": float(probe_message_cost),
        "total_message_budget": float(total_message_budget),
        "expected_risk_ratio": float(ratio),
    }
