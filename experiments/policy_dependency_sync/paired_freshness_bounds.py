"""Executable constants for the paired freshness Lyapunov theorem."""

from __future__ import annotations

from math import isfinite


def selected_utility_lagrangian_regret_upper(
    *, utility_weight: float, selected_confidence_radius: float
) -> float:
    """Twice-radius regret of an optimistic launch action."""

    if (
        not isfinite(utility_weight)
        or not isfinite(selected_confidence_radius)
        or utility_weight <= 0.0
        or selected_confidence_radius < 0.0
    ):
        raise ValueError("weight must be positive and radius nonnegative")
    return float(2.0 * utility_weight * selected_confidence_radius)


def utility_queue_cap(
    *,
    utility_weight: float,
    utility_range_upper: float,
    minimum_positive_cost: float,
    queue_step: float,
    maximum_cost: float,
    average_budget: float,
) -> float:
    """Pathwise queue cap when a zero-cost null launch is always feasible."""

    values = (
        utility_weight,
        utility_range_upper,
        minimum_positive_cost,
        queue_step,
        maximum_cost,
        average_budget,
    )
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("queue-cap inputs must be finite")
    if (
        utility_weight <= 0.0
        or utility_range_upper < 0.0
        or minimum_positive_cost <= 0.0
        or queue_step <= 0.0
        or maximum_cost < minimum_positive_cost
        or average_budget < 0.0
    ):
        raise ValueError("invalid queue-cap domain")
    rejection_threshold = (
        utility_weight * utility_range_upper / minimum_positive_cost
    )
    maximum_increment = queue_step * max(maximum_cost - average_budget, 0.0)
    return float(rejection_threshold + maximum_increment)


def pathwise_average_cost_upper(
    *, average_budget: float, terminal_queue: float, queue_step: float, launches: int
) -> float:
    """Return ``bar_c + Q_N/(nu N)`` from the reflected queue recursion."""

    if launches <= 0 or queue_step <= 0.0:
        raise ValueError("launches and queue step must be positive")
    if min(average_budget, terminal_queue) < 0.0:
        raise ValueError("budget and terminal queue must be nonnegative")
    return float(average_budget + terminal_queue / (queue_step * launches))


def paired_finite_time_rhs(
    *,
    initial_objective_gap: float,
    initial_queue: float,
    queue_step: float,
    utility_weight: float,
    selected_radius_sum: float,
    queue_remainder_sum: float,
    receipt_remainder_sum: float,
    topology_motion_positive_sum: float = 0.0,
) -> float:
    """Right side of the conditional paired utility--stationarity bound.

    The returned quantity is in Lyapunov units, before division by the launch
    utility weight or by a stationarity coefficient.
    """

    values = (
        initial_objective_gap,
        initial_queue,
        queue_step,
        utility_weight,
        selected_radius_sum,
        queue_remainder_sum,
        receipt_remainder_sum,
        topology_motion_positive_sum,
    )
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("finite-time inputs must be finite")
    if min(values) < 0.0 or queue_step <= 0.0 or utility_weight <= 0.0:
        raise ValueError("finite-time inputs must be nonnegative")
    return float(
        initial_objective_gap
        + initial_queue**2 / (2.0 * queue_step)
        + 2.0 * utility_weight * selected_radius_sum
        + queue_remainder_sum
        + receipt_remainder_sum
        + topology_motion_positive_sum
    )
