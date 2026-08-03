"""Observable trajectory-fingerprint probe for trajectory-switch correlation.

Each probe block contains one common standard-task path, private independent
paths, and unobserved Bernoulli switches.  The server receives one fixed-size
fingerprint per agent.  Accidental collisions of independent paths are
computed from the public finite-state kernel rather than ignored.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

from experiments.dependence_delay_linear.t050_stationary_break_even import (
    asymptotic_participation_coefficient,
)


@dataclass(frozen=True)
class FingerprintCertificate:
    estimate: float
    lower: float
    upper: float
    radius: float
    collision_probability: float
    blocks: int
    alpha: float


def state_path_collision_probability(
    *, transition: np.ndarray, stationary: np.ndarray, transitions: int
) -> float:
    """Probability that two independent stationary state paths are identical."""

    probability = np.asarray(transition, dtype=float)
    invariant = np.asarray(stationary, dtype=float)
    states = probability.shape[0]
    if probability.shape != (states, states) or invariant.shape != (states,):
        raise ValueError("transition or stationary shape mismatch")
    if transitions < 0:
        raise ValueError("transitions must be nonnegative")
    if np.any(probability < 0.0) or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-11
    ):
        raise ValueError("transition must be stochastic")
    if np.any(invariant < 0.0) or not np.isclose(invariant.sum(), 1.0):
        raise ValueError("stationary must be a probability vector")
    if not np.allclose(invariant @ probability, invariant, atol=1e-10):
        raise ValueError("stationary must be invariant")
    joint_diagonal_mass = invariant**2
    squared_transition = probability**2
    for _ in range(transitions):
        joint_diagonal_mass = joint_diagonal_mass @ squared_transition
    return float(np.sum(joint_diagonal_mass))


def minimum_fingerprint_length(
    *,
    transition: np.ndarray,
    stationary: np.ndarray,
    maximum_collision: float,
    maximum_transitions: int = 10_000,
) -> dict[str, float | int]:
    """Shortest state-path fingerprint satisfying an accidental-collision cap."""

    if not 0.0 < maximum_collision < 1.0 or maximum_transitions < 0:
        raise ValueError("invalid collision target or transition limit")
    for transitions in range(maximum_transitions + 1):
        collision = state_path_collision_probability(
            transition=transition,
            stationary=stationary,
            transitions=transitions,
        )
        if collision <= maximum_collision:
            return {
                "transitions": transitions,
                "collision_probability": collision,
            }
    raise ValueError("collision target is not attained within the limit")


def trajectory_switch_match_probability(
    *, rho: float, collision_probability: float
) -> float:
    """Expected pairwise fingerprint match under trajectory switches."""

    if not 0.0 <= rho <= 1.0 or not 0.0 <= collision_probability < 1.0:
        raise ValueError("invalid rho or collision probability")
    return float(
        collision_probability + (1.0 - collision_probability) * rho
    )


def pairwise_fingerprint_match_rate(fingerprints: np.ndarray) -> float:
    """Fraction of unordered agent pairs with an identical fingerprint."""

    values = np.asarray(fingerprints)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("fingerprints must contain at least two agents")
    _, counts = np.unique(values, axis=0, return_counts=True)
    matching_pairs = int(np.sum(counts * (counts - 1) // 2))
    agents = values.shape[0]
    return float(matching_pairs / (agents * (agents - 1) / 2))


def fingerprint_correlation_certificate(
    block_match_rates: np.ndarray,
    *,
    collision_probability: float,
    alpha: float,
) -> FingerprintCertificate:
    """Fixed-block Hoeffding interval for trajectory-switch correlation."""

    values = np.asarray(block_match_rates, dtype=float)
    if values.ndim != 1 or values.size < 1 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("block match rates must be a nonempty vector in [0,1]")
    if not 0.0 <= collision_probability < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid collision probability or alpha")
    raw_match = float(np.mean(values))
    raw_rho = (raw_match - collision_probability) / (
        1.0 - collision_probability
    )
    radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * values.size)) / (
        1.0 - collision_probability
    )
    estimate = float(np.clip(raw_rho, 0.0, 1.0))
    return FingerprintCertificate(
        estimate=estimate,
        lower=float(np.clip(raw_rho - radius, 0.0, 1.0)),
        upper=float(np.clip(raw_rho + radius, 0.0, 1.0)),
        radius=radius,
        collision_probability=collision_probability,
        blocks=int(values.size),
        alpha=alpha,
    )


def catalogue_optimal_intervals(
    candidates: Iterable[int], *, overhead: float
) -> dict[int, tuple[float, float]]:
    """Closed intervals on which each catalogue action minimizes PR risk."""

    actions = sorted({int(q) for q in candidates})
    if not actions or actions[0] < 1 or overhead < 0.0:
        raise ValueError("invalid candidates or overhead")

    def affine(q: int) -> tuple[float, float]:
        intercept = (overhead + q) / q
        slope = (overhead + q) * (1.0 - 1.0 / q)
        return intercept, slope

    intervals: dict[int, tuple[float, float]] = {}
    for action in actions:
        lower, upper = 0.0, 1.0
        action_intercept, action_slope = affine(action)
        for other in actions:
            if other == action:
                continue
            other_intercept, other_slope = affine(other)
            intercept = action_intercept - other_intercept
            slope = action_slope - other_slope
            if abs(slope) <= 1e-15:
                if intercept > 0.0:
                    lower, upper = 1.0, 0.0
                    break
                continue
            crossing = -intercept / slope
            if slope > 0.0:
                upper = min(upper, crossing)
            else:
                lower = max(lower, crossing)
        lower, upper = max(0.0, lower), min(1.0, upper)
        if lower <= upper + 1e-14:
            intervals[action] = (lower, upper)
    return intervals


def plug_in_action(
    estimate: float, candidates: Iterable[int], *, overhead: float
) -> int:
    """Choose the leading-risk catalogue minimizer at an observed estimate."""

    if not 0.0 <= estimate <= 1.0:
        raise ValueError("estimate must lie in [0,1]")
    actions = sorted({int(q) for q in candidates})
    if not actions:
        raise ValueError("empty candidate catalogue")
    return min(
        actions,
        key=lambda q: (
            asymptotic_participation_coefficient(
                q, overhead=overhead, rho=estimate
            ),
            q,
        ),
    )


def expected_plugin_coefficient_bound(
    *,
    rho: float,
    candidates: Iterable[int],
    overhead: float,
    blocks: int,
    collision_probability: float,
) -> dict[str, float | int]:
    """Hoeffding upper bound on the plug-in action's expected coefficient."""

    if blocks < 1 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid blocks or rho")
    if not 0.0 <= collision_probability < 1.0:
        raise ValueError("invalid collision probability")
    actions = sorted({int(q) for q in candidates})
    intervals = catalogue_optimal_intervals(actions, overhead=overhead)
    coefficients = {
        q: asymptotic_participation_coefficient(q, overhead=overhead, rho=rho)
        for q in actions
    }
    oracle_q = min(actions, key=lambda q: (coefficients[q], q))
    oracle = coefficients[oracle_q]
    expected = oracle
    for action in actions:
        gap = max(0.0, coefficients[action] - oracle)
        if gap == 0.0 or action not in intervals:
            continue
        lower, upper = intervals[action]
        if rho < lower:
            distance = lower - rho
        elif rho > upper:
            distance = rho - upper
        else:
            distance = 0.0
        probability = min(
            1.0,
            2.0
            * math.exp(
                -2.0
                * blocks
                * (1.0 - collision_probability) ** 2
                * distance**2
            ),
        )
        expected += gap * probability
    return {
        "oracle_q": oracle_q,
        "oracle_coefficient": oracle,
        "expected_coefficient_upper_bound": float(expected),
    }


def full_cost_plugin_ratio_bound(
    *,
    rho: float,
    candidates: Iterable[int],
    overhead: float,
    baseline_q: int,
    learning_budget: float,
    probe_blocks: int,
    probe_q: int,
    collision_probability: float,
) -> dict[str, float | int]:
    """Leading expected-risk ratio after charging all probe messages."""

    if learning_budget <= 0.0 or probe_blocks < 1 or probe_q < 2:
        raise ValueError("invalid learning budget or probe design")
    result = expected_plugin_coefficient_bound(
        rho=rho,
        candidates=candidates,
        overhead=overhead,
        blocks=probe_blocks,
        collision_probability=collision_probability,
    )
    probe_cost = probe_blocks * (overhead + probe_q)
    total_budget = learning_budget + probe_cost
    baseline = asymptotic_participation_coefficient(
        baseline_q, overhead=overhead, rho=rho
    )
    ratio = (
        float(result["expected_coefficient_upper_bound"])
        / learning_budget
        / (baseline / total_budget)
    )
    return {
        **result,
        "baseline_q": int(baseline_q),
        "probe_message_cost": float(probe_cost),
        "total_message_budget": float(total_budget),
        "expected_risk_ratio_upper_bound": float(ratio),
    }
