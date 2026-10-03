"""Finite-horizon performance constants for cache-potential greedy control."""

from __future__ import annotations

from math import isfinite
from typing import Sequence


def drift_structured_relative_value(
    *, optimal_value: float, current_potential: float, lyapunov_scale: float
) -> float:
    """Return ``h*=V*+P/V`` for the exact drift-structured decomposition."""

    values = (optimal_value, current_potential, lyapunov_scale)
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("value-decomposition inputs must be finite")
    if current_potential < 0.0 or lyapunov_scale <= 0.0:
        raise ValueError("potential is nonnegative and scale is positive")
    return float(optimal_value + current_potential / lyapunov_scale)


def drift_structured_action_score(
    *,
    stage_reward: float,
    expected_next_potential: float,
    expected_next_relative_value: float,
    lyapunov_scale: float,
) -> float:
    """Exact Bellman score after separating Lyapunov and residual value.

    The current potential is action independent and is therefore omitted.
    With ``h_(t+1)^*=V_(t+1)^*+P_(t+1)/V``, this score equals the ordinary
    Bellman score ``r+E[V_(t+1)^*]`` exactly.
    """

    values = (
        stage_reward,
        expected_next_potential,
        expected_next_relative_value,
        lyapunov_scale,
    )
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("action-score inputs must be finite")
    if expected_next_potential < 0.0 or lyapunov_scale <= 0.0:
        raise ValueError("next potential is nonnegative and scale is positive")
    return float(
        stage_reward
        - expected_next_potential / lyapunov_scale
        + expected_next_relative_value
    )


def proxy_greedy_policy_regret_upper(
    *,
    continuation_value_errors: Sequence[float],
    action_score_errors: Sequence[float] | None = None,
) -> float:
    """Return the finite-horizon loss bound for approximate-value greediness.

    At decision ``t``, ``continuation_value_errors[t]`` uniformly bounds
    ``|V*_(t+1) - Vhat_(t+1)|``. ``action_score_errors[t]`` is a uniform
    additional error in the action score, for example from a queue-drift upper
    bound or a utility confidence interval.  The standard one-step sandwich
    contributes twice each error.
    """

    continuation = tuple(float(value) for value in continuation_value_errors)
    scores = (
        (0.0,) * len(continuation)
        if action_score_errors is None
        else tuple(float(value) for value in action_score_errors)
    )
    if len(scores) != len(continuation):
        raise ValueError("continuation and action-score error sequences must align")
    if any(not isfinite(value) or value < 0.0 for value in (*continuation, *scores)):
        raise ValueError("performance errors must be finite and nonnegative")
    return float(2.0 * sum(continuation) + 2.0 * sum(scores))


def queue_drift_score_error_upper(
    *, queue_step: float, maximum_cost_deviation: float, lyapunov_weight: float
) -> float:
    """Uniform utility-scale error from first-order queue drift pricing."""

    values = (queue_step, maximum_cost_deviation, lyapunov_weight)
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("queue approximation inputs must be finite")
    if queue_step <= 0.0 or maximum_cost_deviation < 0.0 or lyapunov_weight <= 0.0:
        raise ValueError("invalid queue approximation domain")
    return float(
        queue_step * maximum_cost_deviation**2 / (2.0 * lyapunov_weight)
    )


def residual_shielded_queue_cap(
    *,
    nonqueue_score_advantage_upper: float,
    minimum_positive_cost: float,
    queue_step: float,
    maximum_cost: float,
) -> float:
    """Pathwise queue cap when a bounded residual competes with a null action.

    ``nonqueue_score_advantage_upper`` must include every possible advantage
    of a non-null action over the zero-cost null action, including learned
    residual span and confidence bonuses.  Above its ratio to the minimum
    positive cost, the queue price forces the null action.  The second term is
    the largest conservative one-step overshoot.
    """

    values = (
        nonqueue_score_advantage_upper,
        minimum_positive_cost,
        queue_step,
        maximum_cost,
    )
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("queue-cap inputs must be finite")
    if (
        nonqueue_score_advantage_upper < 0.0
        or minimum_positive_cost <= 0.0
        or queue_step <= 0.0
        or maximum_cost < minimum_positive_cost
    ):
        raise ValueError("invalid residual queue-cap domain")
    return float(
        nonqueue_score_advantage_upper / minimum_positive_cost
        + queue_step * maximum_cost
    )
