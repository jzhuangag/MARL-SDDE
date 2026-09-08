"""Finite-horizon performance constants for cache-potential greedy control."""

from __future__ import annotations

from math import isfinite
from typing import Sequence


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
