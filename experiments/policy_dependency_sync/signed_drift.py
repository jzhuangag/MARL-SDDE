"""Closed-form signed one-edge Lyapunov drift minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DriftChoice:
    donor: int | None
    step: float
    score: float
    candidate_gradient: float


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

