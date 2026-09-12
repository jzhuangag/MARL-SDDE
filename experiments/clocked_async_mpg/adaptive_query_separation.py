"""Exact fresh-query versus frozen-shadow-batch separation witness."""

from __future__ import annotations

import math


def one_round_minimax_gradient(curvature_low: float, curvature_high: float) -> float:
    """Best worst-case gradient after one nonadaptive query at zero.

    The two objectives are ``f_a(x)=0.5*a*x^2-x``.  Both return gradient ``-1``
    at ``x=0``.  Any number of duplicate oracle calls at that point therefore
    gives the same information.  The best common output has worst gradient
    ``(a_high-a_low)/(a_high+a_low)``.
    """

    _validate_curvatures(curvature_low, curvature_high)
    return (curvature_high-curvature_low)/(curvature_high+curvature_low)


def fixed_safe_step_worst_gradient(
    curvature_low: float, curvature_high: float, fresh_queries: int
) -> float:
    """Worst gradient after fixed-step fresh sequential gradient queries."""

    _validate_curvatures(curvature_low, curvature_high)
    if fresh_queries < 0:
        raise ValueError("fresh_queries must be nonnegative")
    contraction = 1.0-curvature_low/curvature_high
    return contraction**fresh_queries


def expected_async_two_query_time(fast_rate: float, slow_rate: float) -> float:
    """Expected max of Gamma(2, fast_rate) and Exp(slow_rate)."""

    _validate_rates(fast_rate, slow_rate)
    rate_sum = fast_rate+slow_rate
    expected_minimum = 1.0/rate_sum+fast_rate/rate_sum**2
    return 2.0/fast_rate+1.0/slow_rate-expected_minimum


def expected_two_barrier_rounds(fast_rate: float, slow_rate: float) -> float:
    """Expected duration of two iid two-agent exponential barriers."""

    _validate_rates(fast_rate, slow_rate)
    one_round = (
        1.0/fast_rate+1.0/slow_rate-1.0/(fast_rate+slow_rate)
    )
    return 2.0*one_round


def separation_certificate(
    curvature_low: float,
    curvature_high: float,
    fast_rate: float,
    slow_rate: float,
) -> dict[str, float | bool]:
    """Return the two-query accuracy and elapsed-time separation."""

    one_round = one_round_minimax_gradient(curvature_low, curvature_high)
    asynchronous_error = fixed_safe_step_worst_gradient(
        curvature_low, curvature_high, fresh_queries=2
    )
    asynchronous_time = expected_async_two_query_time(fast_rate, slow_rate)
    synchronous_time = expected_two_barrier_rounds(fast_rate, slow_rate)
    return {
        "async_expected_time": asynchronous_time,
        "async_two_query_worst_gradient": asynchronous_error,
        "nonempty_accuracy_interval": asynchronous_error < one_round,
        "one_round_shadow_batch_lower_bound": one_round,
        "strict_time_advantage": asynchronous_time < synchronous_time,
        "two_barrier_expected_time": synchronous_time,
    }


def _validate_curvatures(curvature_low: float, curvature_high: float) -> None:
    if not (
        math.isfinite(curvature_low)
        and math.isfinite(curvature_high)
        and 0.0 < curvature_low < curvature_high
    ):
        raise ValueError("curvatures must satisfy 0 < low < high")


def _validate_rates(fast_rate: float, slow_rate: float) -> None:
    if not (
        math.isfinite(fast_rate)
        and math.isfinite(slow_rate)
        and fast_rate > 0.0
        and slow_rate > 0.0
    ):
        raise ValueError("rates must be finite and positive")
