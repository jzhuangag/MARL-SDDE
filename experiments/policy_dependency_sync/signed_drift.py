"""Closed-form signed one-edge Lyapunov drift minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DriftChoice:
    donor: int | None
    step: float
    score: float
    candidate_gradient: float


def delayed_cache_reset_benefit(
    weight: float,
    current_parameter: float,
    cached_parameter: float,
    delivered_parameter: float,
) -> float:
    """Exact decrease of a scalar weighted cache-mismatch energy."""
    if weight < 0.0:
        raise ValueError("weight must be nonnegative")
    before = current_parameter - cached_parameter
    after = current_parameter - delivered_parameter
    return 0.5 * weight * (before * before - after * after)


def outgoing_cache_debt_drift(
    weight_by_recipient: Mapping[int, float],
    current_parameter: float,
    cache_by_recipient: Mapping[int, float],
    packet_gradient: float,
    step: float,
) -> float:
    """Exact cache-energy change when a donor parameter is updated."""
    if set(weight_by_recipient) != set(cache_by_recipient):
        raise ValueError("weight and cache recipients must match")
    if min(weight_by_recipient.values(), default=0.0) < 0.0:
        raise ValueError("weights must be nonnegative")
    if step < 0.0:
        raise ValueError("step must be nonnegative")
    displacement = step * packet_gradient
    linear_debt = sum(
        weight_by_recipient[recipient]
        * (current_parameter - cache_by_recipient[recipient])
        for recipient in weight_by_recipient
    )
    total_weight = sum(weight_by_recipient.values())
    return float(
        -displacement * linear_debt
        + 0.5 * displacement * displacement * total_weight
    )


def receipt_optimal_step(
    objective_gradient: float,
    packet_gradient: float,
    objective_smoothness: float,
    weight_by_recipient: Mapping[int, float],
    current_parameter: float,
    cache_by_recipient: Mapping[int, float],
    maximum_step: float,
    pending_linear_coefficient: float = 0.0,
    pending_curvature: float = 0.0,
) -> float:
    """Minimize objective, outgoing-cache, and pending quadratic drift."""
    if objective_smoothness <= 0.0:
        raise ValueError("objective smoothness must be positive")
    if pending_linear_coefficient < 0.0 or pending_curvature < 0.0:
        raise ValueError("pending coefficients must be nonnegative")
    if set(weight_by_recipient) != set(cache_by_recipient):
        raise ValueError("weight and cache recipients must match")
    if min(weight_by_recipient.values(), default=0.0) < 0.0:
        raise ValueError("weights must be nonnegative")
    outgoing_gradient = sum(
        weight_by_recipient[recipient]
        * (current_parameter - cache_by_recipient[recipient])
        for recipient in weight_by_recipient
    )
    certified_alignment = (
        (objective_gradient + outgoing_gradient) * packet_gradient
        - pending_linear_coefficient * abs(packet_gradient)
    )
    curvature = objective_smoothness + sum(weight_by_recipient.values())
    curvature += pending_curvature
    return robust_optimal_step(
        certified_alignment,
        abs(packet_gradient),
        curvature,
        maximum_step,
    )


def pending_history_increment_envelope(
    radii: Sequence[float],
    cross_sensitivities: Sequence[float],
    weights: Sequence[float],
    parameter_displacement: float,
) -> tuple[float, float, float]:
    """Quadratic envelope for the history-energy shift of pending packets.

    Returns ``(increment, linear_coefficient, curvature)`` where the increment
    equals ``linear * displacement + 0.5 * curvature * displacement**2``.
    """
    if not (len(radii) == len(cross_sensitivities) == len(weights)):
        raise ValueError("pending sequences must have equal lengths")
    if min(radii, default=0.0) < 0.0:
        raise ValueError("radii must be nonnegative")
    if min(cross_sensitivities, default=0.0) < 0.0:
        raise ValueError("cross sensitivities must be nonnegative")
    if min(weights, default=0.0) < 0.0:
        raise ValueError("weights must be nonnegative")
    if parameter_displacement < 0.0:
        raise ValueError("parameter displacement must be nonnegative")
    linear = sum(
        weight * radius * sensitivity
        for radius, sensitivity, weight in zip(
            radii, cross_sensitivities, weights, strict=True
        )
    )
    curvature = sum(
        weight * sensitivity * sensitivity
        for sensitivity, weight in zip(cross_sensitivities, weights, strict=True)
    )
    increment = linear * parameter_displacement
    increment += 0.5 * curvature * parameter_displacement**2
    return float(increment), float(linear), float(curvature)


def robust_alignment_lower_bound(
    true_gradient_estimate: np.ndarray | float,
    candidate_gradient_estimate: np.ndarray | float,
    true_gradient_radius: float,
    candidate_gradient_radius: float,
) -> float:
    """Lower-bound the true gradient/candidate-gradient inner product."""
    if true_gradient_radius < 0.0 or candidate_gradient_radius < 0.0:
        raise ValueError("confidence radii must be nonnegative")
    estimate_a = np.atleast_1d(np.asarray(true_gradient_estimate, dtype=float))
    estimate_g = np.atleast_1d(np.asarray(candidate_gradient_estimate, dtype=float))
    if estimate_a.shape != estimate_g.shape:
        raise ValueError("gradient estimates must have the same shape")
    return float(
        estimate_a @ estimate_g
        - true_gradient_radius * np.linalg.norm(estimate_g)
        - candidate_gradient_radius * np.linalg.norm(estimate_a)
        - true_gradient_radius * candidate_gradient_radius
    )


def robust_candidate_norm_upper_bound(
    candidate_gradient_estimate: np.ndarray | float,
    candidate_gradient_radius: float,
) -> float:
    """Upper-bound the norm of the unknown candidate gradient."""
    if candidate_gradient_radius < 0.0:
        raise ValueError("confidence radius must be nonnegative")
    estimate = np.atleast_1d(np.asarray(candidate_gradient_estimate, dtype=float))
    return float(np.linalg.norm(estimate) + candidate_gradient_radius)


def robust_optimal_step(
    alignment_lower_bound: float,
    candidate_norm_upper_bound: float,
    smoothness: float,
    maximum_step: float,
) -> float:
    """Minimize a confidence-valid smoothness drift upper bound."""
    if smoothness <= 0.0:
        raise ValueError("smoothness must be positive")
    if candidate_norm_upper_bound < 0.0:
        raise ValueError("candidate norm upper bound must be nonnegative")
    if maximum_step < 0.0:
        raise ValueError("maximum_step must be nonnegative")
    if alignment_lower_bound <= 0.0 or candidate_norm_upper_bound == 0.0:
        return 0.0
    unconstrained = alignment_lower_bound / (
        smoothness * candidate_norm_upper_bound * candidate_norm_upper_bound
    )
    return min(unconstrained, maximum_step)


def fixed_step_score(
    true_gradient: float,
    candidate_gradient: float,
    curvature: float,
    step: float,
    reset_benefit: float = 0.0,
    queue_price: float = 0.0,
) -> float:
    """Exact quadratic-potential drift plus reset and queue increments."""
    if curvature <= 0.0:
        raise ValueError("curvature must be positive")
    if step < 0.0:
        raise ValueError("step must be nonnegative")
    return (
        -step * true_gradient * candidate_gradient
        + 0.5 * step * step * curvature * candidate_gradient * candidate_gradient
        - reset_benefit
        + queue_price
    )


def optimal_step(
    true_gradient: float,
    candidate_gradient: float,
    curvature: float,
    maximum_step: float,
) -> float:
    """Minimize the smooth quadratic progress bound on a closed interval."""
    if curvature <= 0.0:
        raise ValueError("curvature must be positive")
    if maximum_step < 0.0:
        raise ValueError("maximum_step must be nonnegative")
    if candidate_gradient == 0.0:
        return 0.0
    unconstrained = (
        true_gradient * candidate_gradient
        / (curvature * candidate_gradient * candidate_gradient)
    )
    return min(max(unconstrained, 0.0), maximum_step)


def choose_edge_and_step(
    true_gradient: float,
    stale_gradient: float,
    edge_gradient_changes: Mapping[int, float],
    curvature: float,
    maximum_step: float,
    reset_benefits: Mapping[int, float] | None = None,
    queue_prices: Mapping[int, float] | None = None,
) -> DriftChoice:
    """Exactly minimize the registered one-edge composite drift score."""
    reset_benefits = {} if reset_benefits is None else reset_benefits
    queue_prices = {} if queue_prices is None else queue_prices
    candidates: list[DriftChoice] = []
    for donor in (None, *sorted(edge_gradient_changes)):
        candidate_gradient = stale_gradient
        reset = 0.0
        price = 0.0
        if donor is not None:
            candidate_gradient += edge_gradient_changes[donor]
            reset = float(reset_benefits.get(donor, 0.0))
            price = float(queue_prices.get(donor, 0.0))
        step = optimal_step(
            true_gradient,
            candidate_gradient,
            curvature,
            maximum_step,
        )
        score = fixed_step_score(
            true_gradient,
            candidate_gradient,
            curvature,
            step,
            reset,
            price,
        )
        candidates.append(DriftChoice(donor, step, score, candidate_gradient))
    return min(candidates, key=lambda item: (item.score, item.donor is not None, item.donor or -1))
