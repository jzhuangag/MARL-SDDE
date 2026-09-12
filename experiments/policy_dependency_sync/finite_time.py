"""Finite-time identities for expectation-level Lyapunov scheduling.

The functions in this module implement deterministic pieces of the theorem.
They do not estimate policy gradients or assert Markov-chain concentration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping


@dataclass(frozen=True)
class DPPChoice:
    """Action selected by a drift-plus-penalty index."""

    action: Hashable
    index: float
    learning_score: float
    cost: float


def choose_scaled_drift_plus_penalty(
    estimated_learning_scores: Mapping[Hashable, float],
    costs: Mapping[Hashable, float],
    queue: float,
    learning_weight: float,
) -> DPPChoice:
    """Minimize ``V * estimated_score + Q * cost`` with stable ties."""
    if not estimated_learning_scores:
        raise ValueError("at least one action is required")
    if set(estimated_learning_scores) != set(costs):
        raise ValueError("score and cost actions must match")
    if queue < 0.0 or learning_weight <= 0.0:
        raise ValueError("queue must be nonnegative and learning_weight positive")
    if min(costs.values()) < 0.0:
        raise ValueError("costs must be nonnegative")
    action = min(
        estimated_learning_scores,
        key=lambda item: (
            learning_weight * estimated_learning_scores[item]
            + queue * costs[item],
            str(item),
        ),
    )
    score = float(estimated_learning_scores[action])
    cost = float(costs[action])
    return DPPChoice(
        action=action,
        index=learning_weight * score + queue * cost,
        learning_score=score,
        cost=cost,
    )


def selected_index_regret_bound(
    true_learning_scores: Mapping[Hashable, float],
    estimated_learning_scores: Mapping[Hashable, float],
    costs: Mapping[Hashable, float],
    queue: float,
    learning_weight: float,
) -> tuple[float, float]:
    """Return true index regret and its ``2 V sup-error`` upper bound."""
    if not true_learning_scores:
        raise ValueError("at least one action is required")
    if not (
        set(true_learning_scores)
        == set(estimated_learning_scores)
        == set(costs)
    ):
        raise ValueError("true scores, estimates, and costs must match")
    chosen = choose_scaled_drift_plus_penalty(
        estimated_learning_scores,
        costs,
        queue,
        learning_weight,
    ).action
    true_indices = {
        action: learning_weight * true_learning_scores[action]
        + queue * costs[action]
        for action in true_learning_scores
    }
    regret = float(true_indices[chosen] - min(true_indices.values()))
    sup_error = max(
        abs(estimated_learning_scores[action] - true_learning_scores[action])
        for action in true_learning_scores
    )
    return regret, 2.0 * learning_weight * float(sup_error)


def queue_squared_drift_upper(
    queue: float,
    cost: float,
    average_budget: float,
) -> float:
    """Standard upper bound for half the squared virtual-queue increment."""
    if queue < 0.0 or cost < 0.0 or average_budget < 0.0:
        raise ValueError("queue, cost, and average_budget must be nonnegative")
    difference = cost - average_budget
    return queue * difference + 0.5 * difference * difference


def finite_time_stationarity_upper(
    initial_learning_energy: float,
    initial_queue: float,
    horizon: int,
    learning_weight: float,
    descent_coefficient: float,
    queue_drift_constant: float,
    comparator_remainder_sum: float,
    score_error_sum: float,
) -> float:
    """Upper bound on average stationarity from the telescoped theorem.

    ``comparator_remainder_sum`` and ``score_error_sum`` are unscaled learning
    quantities.  The returned bound is

    ``(U0 + Q0^2/(2V))/(gamma K) + B/(gamma V)
       + (R + 2E)/(gamma K)``.
    """
    if initial_learning_energy < 0.0 or initial_queue < 0.0:
        raise ValueError("initial energies must be nonnegative")
    if horizon <= 0 or learning_weight <= 0.0 or descent_coefficient <= 0.0:
        raise ValueError("horizon, learning_weight, and descent must be positive")
    if min(
        queue_drift_constant,
        comparator_remainder_sum,
        score_error_sum,
    ) < 0.0:
        raise ValueError("remainder terms must be nonnegative")
    initial_term = (
        initial_learning_energy
        + 0.5 * initial_queue * initial_queue / learning_weight
    ) / (descent_coefficient * horizon)
    queue_term = queue_drift_constant / (
        descent_coefficient * learning_weight
    )
    remainder_term = (
        comparator_remainder_sum + 2.0 * score_error_sum
    ) / (descent_coefficient * horizon)
    return float(initial_term + queue_term + remainder_term)


def deterministic_queue_cap(
    learning_weight: float,
    estimated_score_bound: float,
    minimum_positive_cost: float,
    maximum_cost: float,
) -> float:
    """Queue cap when a zero-cost null action is always available.

    If every nonzero action costs at least ``minimum_positive_cost`` and every
    estimated learning score has magnitude at most ``estimated_score_bound``,
    the scheduler selects the null action above the threshold.  One final
    positive increment yields this inclusive cap.
    """
    if learning_weight <= 0.0:
        raise ValueError("learning_weight must be positive")
    if estimated_score_bound < 0.0:
        raise ValueError("estimated_score_bound must be nonnegative")
    if minimum_positive_cost <= 0.0 or maximum_cost < minimum_positive_cost:
        raise ValueError("invalid positive-cost range")
    threshold = (
        2.0 * learning_weight * estimated_score_bound
        / minimum_positive_cost
    )
    return float(threshold + maximum_cost)


def average_cost_upper(
    average_budget: float,
    initial_queue: float,
    terminal_queue_upper: float,
    horizon: int,
) -> float:
    """Pathwise average-cost upper bound induced by the virtual queue."""
    if min(average_budget, initial_queue, terminal_queue_upper) < 0.0:
        raise ValueError("budget and queues must be nonnegative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return float(
        average_budget
        + (terminal_queue_upper - initial_queue) / horizon
    )
