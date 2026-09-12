"""Outcome-free wall-clock coefficients for the theorem candidate."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
from numpy.typing import NDArray

from .finite_time_drift import rate_balanced_steps


Array = NDArray[np.float64]


def robust_stale_direction_progress(
    signal_norm: float,
    uncertainty_radius: float,
    smoothness: float,
    maximum_step: float = math.inf,
) -> dict[str, float]:
    """Exact worst-case smooth progress along one stale direction.

    The stale packet has norm ``signal_norm`` and the current gradient is only
    known to lie in an Euclidean ball of radius ``uncertainty_radius`` around
    it.  For an update ``alpha * packet``, block smoothness gives the robust
    certificate

    ``alpha*s*(s-B) - 0.5*L*alpha**2*s**2``.

    This helper returns the maximizing nonnegative step and certificate.  It
    is an upper bound on what *any* rule restricted to the observed stale
    direction can certify from this information, rather than a new controller.
    """

    values = (signal_norm, uncertainty_radius, smoothness, maximum_step)
    if not all(math.isfinite(value) or value == math.inf for value in values):
        raise ValueError("arguments must be finite, except an infinite step cap")
    if signal_norm < 0.0 or uncertainty_radius < 0.0:
        raise ValueError("signal and uncertainty must be nonnegative")
    if smoothness <= 0.0 or maximum_step < 0.0:
        raise ValueError("smoothness must be positive and the step cap nonnegative")
    if signal_norm == 0.0 or uncertainty_radius >= signal_norm:
        return {
            "certified_progress": 0.0,
            "step": 0.0,
            "unconstrained_step": 0.0,
        }
    unconstrained = (
        signal_norm-uncertainty_radius
    )/(smoothness*signal_norm)
    step = min(unconstrained, maximum_step)
    progress = (
        step*signal_norm*(signal_norm-uncertainty_radius)
        -0.5*smoothness*step**2*signal_norm**2
    )
    return {
        "certified_progress": progress,
        "step": step,
        "unconstrained_step": unconstrained,
    }


def _validated_essential_packet_service_vectors(
    required_fresh_packets: Array,
    service_values: Array,
    service_label: str,
) -> tuple[Array, Array]:
    """Validate essential packet counts and a positive per-block service vector."""

    packets = np.asarray(required_fresh_packets, dtype=float)
    service = np.asarray(service_values, dtype=float)
    if packets.ndim != 1 or packets.size == 0 or packets.shape != service.shape:
        raise ValueError(
            f"packet counts and {service_label} must be equal nonempty vectors"
        )
    if (
        not np.isfinite(packets).all()
        or (packets < 0.0).any()
        or not np.equal(packets, np.floor(packets)).all()
    ):
        raise ValueError("required packet counts must be finite nonnegative integers")
    if not np.isfinite(service).all() or (service <= 0.0).any():
        raise ValueError(f"{service_label} must be finite and positive")
    return packets, service


def essential_agent_clock_lower_bound(
    required_fresh_packets: Array, completion_rates: Array
) -> float:
    """Clock lower bound imposed by strategically essential policy blocks.

    If block ``i`` needs at least ``m_i`` sequential fresh packets and its
    packet clock is Poisson with rate ``lambda_i``, the expected time of its
    ``m_i``-th packet is ``m_i/lambda_i``.  Any full-policy stopping time that
    requires all blocks is therefore at least the maximum of these means.
    """

    packets, rates = _validated_essential_packet_service_vectors(
        required_fresh_packets, completion_rates, "completion rates"
    )
    return float(np.max(packets/rates))


def essential_agent_periodic_service_clock_lower_bound(
    required_fresh_packets: Array, service_periods: Array
) -> float:
    """Clock lower bound for essential blocks with deterministic service periods."""

    packets, periods = _validated_essential_packet_service_vectors(
        required_fresh_packets, service_periods, "service periods"
    )
    return float(np.max(packets*periods))


def gaussian_sign_packet_lower_bound(
    signal_magnitude: float,
    noise_standard_deviation: float,
    error_probability: float,
) -> float:
    """Expected packets required to identify a Gaussian gradient sign.

    Under hypotheses ``N(+Delta, sigma^2)`` and ``N(-Delta, sigma^2)``, the
    per-packet KL divergence is ``2*Delta^2/sigma^2``.  Binary data processing
    at a sequential stopping rule with both errors at most ``delta`` gives the
    returned change-of-measure lower bound on expected packet count.
    """

    if not (
        math.isfinite(signal_magnitude)
        and math.isfinite(noise_standard_deviation)
        and signal_magnitude > 0.0
        and noise_standard_deviation > 0.0
    ):
        raise ValueError("signal and noise scale must be finite and positive")
    if not (
        math.isfinite(error_probability)
        and 0.0 < error_probability < 0.5
    ):
        raise ValueError("error_probability must lie strictly between zero and one half")
    binary_kl = (
        (1.0-error_probability)
        *math.log((1.0-error_probability)/error_probability)
        +error_probability
        *math.log(error_probability/(1.0-error_probability))
    )
    per_packet_kl = 2.0*signal_magnitude**2/noise_standard_deviation**2
    return binary_kl/per_packet_kl


def stochastic_essential_agent_clock_lower_bound(
    signal_magnitudes: Array,
    noise_standard_deviations: Array,
    completion_rates: Array,
    error_probability: float,
) -> float:
    """Wall-clock lower bound for all-block Gaussian sign identification."""

    signals = np.asarray(signal_magnitudes, dtype=float)
    noise = np.asarray(noise_standard_deviations, dtype=float)
    rates = np.asarray(completion_rates, dtype=float)
    if (
        signals.ndim != 1
        or signals.size == 0
        or signals.shape != noise.shape
        or signals.shape != rates.shape
    ):
        raise ValueError("signals, noise scales and rates must be equal vectors")
    packet_bounds = np.asarray(
        [
            gaussian_sign_packet_lower_bound(signal, scale, error_probability)
            for signal, scale in zip(signals, noise, strict=True)
        ]
    )
    if not np.isfinite(rates).all() or (rates <= 0.0).any():
        raise ValueError("completion rates must be finite and positive")
    return float(np.max(packet_bounds/rates))


def stochastic_essential_agent_periodic_service_lower_bound(
    signal_magnitudes: Array,
    noise_standard_deviations: Array,
    service_periods: Array,
    error_probability: float,
) -> float:
    """Periodic-service lower bound for all-block Gaussian sign identification."""

    signals = np.asarray(signal_magnitudes, dtype=float)
    noise = np.asarray(noise_standard_deviations, dtype=float)
    periods = np.asarray(service_periods, dtype=float)
    if (
        signals.ndim != 1
        or signals.size == 0
        or signals.shape != noise.shape
        or signals.shape != periods.shape
    ):
        raise ValueError("signals, noise scales and periods must be equal vectors")
    packet_bounds = np.asarray(
        [
            gaussian_sign_packet_lower_bound(signal, scale, error_probability)
            for signal, scale in zip(signals, noise, strict=True)
        ]
    )
    if not np.isfinite(periods).all() or (periods <= 0.0).any():
        raise ValueError("service periods must be finite and positive")
    return float(np.max(packet_bounds*periods))


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
