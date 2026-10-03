"""Deterministic identities for noisy drift-plus-penalty selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping


@dataclass(frozen=True)
class ScoreChoice:
    action: Hashable
    estimated_score: float
    cost: float


def choose_drift_plus_penalty(
    estimated_learning_scores: Mapping[Hashable, float],
    costs: Mapping[Hashable, float],
    queue: float,
) -> ScoreChoice:
    """Minimize an estimated learning score plus the current queue price."""
    if queue < 0.0:
        raise ValueError("queue must be nonnegative")
    if not estimated_learning_scores:
        raise ValueError("at least one action is required")
    if set(estimated_learning_scores) != set(costs):
        raise ValueError("score and cost actions must match")
    if min(costs.values()) < 0.0:
        raise ValueError("costs must be nonnegative")
    action = min(
        estimated_learning_scores,
        key=lambda item: (
            estimated_learning_scores[item] + queue * costs[item],
            str(item),
        ),
    )
    return ScoreChoice(
        action=action,
        estimated_score=float(
            estimated_learning_scores[action] + queue * costs[action]
        ),
        cost=float(costs[action]),
    )


def realized_selection_regret(
    true_scores: Mapping[Hashable, float],
    estimated_scores: Mapping[Hashable, float],
) -> tuple[float, float]:
    """Return realized regret and the deterministic ``2 sup-error`` bound."""
    if not true_scores or set(true_scores) != set(estimated_scores):
        raise ValueError("nonempty true and estimated action sets must match")
    selected = min(estimated_scores, key=lambda item: (estimated_scores[item], str(item)))
    regret = float(true_scores[selected] - min(true_scores.values()))
    sup_error = max(
        abs(float(estimated_scores[item] - true_scores[item])) for item in true_scores
    )
    return regret, 2.0 * sup_error


def queue_update(queue: float, cost: float, average_budget: float) -> float:
    """Update a nonnegative virtual communication queue."""
    if queue < 0.0 or cost < 0.0 or average_budget < 0.0:
        raise ValueError("queue, cost, and budget must be nonnegative")
    return max(queue + cost - average_budget, 0.0)
