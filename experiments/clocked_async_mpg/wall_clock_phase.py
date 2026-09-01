"""Outcome-free wall-clock coefficients for the theorem candidate."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
from numpy.typing import NDArray

from .finite_time_drift import rate_balanced_steps


Array = NDArray[np.float64]


def expected_maximum_exponential(completion_rates: Array) -> float:
    """Exact expected maximum of independent exponential service times."""

    rates = np.asarray(completion_rates, dtype=float)
    if (
        rates.ndim != 1
        or rates.size == 0
        or (rates <= 0.0).any()
        or not np.isfinite(rates).all()
    ):
        raise ValueError("completion_rates must be a finite positive vector")
    expectation = 0.0
    indices = tuple(range(rates.size))
    for size in range(1, rates.size+1):
        sign = 1.0 if size%2 else -1.0
        for subset in combinations(indices, size):
            expectation += sign/float(np.sum(rates[list(subset)]))
    return expectation


def certified_wall_clock_coefficients(
    cross_lipschitz: Array,
    completion_rates: Array,
    maximum_event_delay: int,
    synchronous_smoothness: float,
    history_inflation: float = 1.0,
) -> dict[str, float | Array]:
    """Return asynchronous and one-packet barrier descent coefficients.

    The asynchronous coefficient is ``Lambda*c_star``.  The synchronous
    coefficient uses step ``1/synchronous_smoothness`` and one independent
    exponential completion per agent in each barrier round.
    """

    rates = np.asarray(completion_rates, dtype=float)
    if (
        rates.ndim != 1
        or rates.size == 0
        or (rates <= 0.0).any()
        or not np.isfinite(rates).all()
    ):
        raise ValueError("completion_rates must be a finite positive vector")
    if synchronous_smoothness <= 0.0 or not math.isfinite(
        synchronous_smoothness
    ):
        raise ValueError("synchronous_smoothness must be finite and positive")
    total_rate = float(np.sum(rates))
    probabilities = rates/total_rate
    allocation = rate_balanced_steps(
        np.asarray(cross_lipschitz, dtype=float),
        probabilities,
        maximum_event_delay,
        history_inflation,
    )
    asynchronous = total_rate*float(allocation["descent_scale"])
    barrier_time = expected_maximum_exponential(rates)
    synchronous = 1.0/(synchronous_smoothness*barrier_time)
    return {
        "asynchronous_coefficient": asynchronous,
        "barrier_round_time": barrier_time,
        "coefficient_ratio": asynchronous/synchronous,
        "descent_scale": float(allocation["descent_scale"]),
        "mark_probabilities": probabilities,
        "step_sizes": np.asarray(allocation["step_sizes"]),
        "synchronous_coefficient": synchronous,
        "total_completion_rate": total_rate,
    }


def symmetric_interaction_phase(
    agents: int,
    diagonal_smoothness: float,
    cross_smoothness: float,
    completion_rate: float,
    maximum_event_delay: int,
    history_inflation: float = 1.0,
) -> dict[str, float | Array]:
    """Certified phase coefficients for a symmetric interaction matrix."""

    if agents <= 0:
        raise ValueError("agents must be positive")
    if diagonal_smoothness <= 0.0 or cross_smoothness < 0.0:
        raise ValueError("smoothness constants are invalid")
    matrix = np.full((agents, agents), cross_smoothness, dtype=float)
    np.fill_diagonal(matrix, diagonal_smoothness)
    global_smoothness = diagonal_smoothness+(agents-1)*cross_smoothness
    result = certified_wall_clock_coefficients(
        matrix,
        np.full(agents, completion_rate, dtype=float),
        maximum_event_delay,
        global_smoothness,
        history_inflation,
    )
    result["global_smoothness"] = global_smoothness
    return result
