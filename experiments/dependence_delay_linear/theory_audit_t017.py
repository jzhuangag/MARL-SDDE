"""Executable algebraic checks for the T-017 theory audit.

This module contains deterministic identities only.  It does not launch or
simulate an experiment.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from adaptive_change_of_measure import AdaptiveAction, dense_common_covariance


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def centered_normal_scale_tv(theta0: float, theta1: float) -> float:
    """Exact TV distance between N(0, theta0) and N(0, theta1)."""

    if theta0 <= 0.0 or theta1 <= 0.0:
        raise ValueError("variances must be strictly positive")
    if theta0 == theta1:
        return 0.0
    small, large = sorted((theta0, theta1))
    crossing_squared = small * large * math.log(large / small) / (large - small)
    crossing = math.sqrt(crossing_squared)
    return 2.0 * (
        normal_cdf(crossing / math.sqrt(small))
        - normal_cdf(crossing / math.sqrt(large))
    )


def le_cam_error_floor(theta0: float, theta1: float) -> float:
    """Equal-prior Bayes error, and hence a lower bound on maximum error."""

    return 0.5 * (1.0 - centered_normal_scale_tv(theta0, theta1))


def zero_mean_gaussian_kl(source: np.ndarray, target: np.ndarray) -> float:
    """KL(N(0, source) || N(0, target))."""

    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("covariances must be square and have equal shape")
    sign_source, logdet_source = np.linalg.slogdet(source)
    sign_target, logdet_target = np.linalg.slogdet(target)
    if sign_source <= 0 or sign_target <= 0:
        raise ValueError("covariances must be positive definite")
    dimension = source.shape[0]
    return 0.5 * (
        np.trace(np.linalg.solve(target, source))
        - dimension
        + logdet_target
        - logdet_source
    )


def mixing_boundary_kl(
    actions: Sequence[AdaptiveAction], theta: float, mixing: float
) -> float:
    """KL from a finite lambda experiment to its lambda=1 boundary law."""

    covariance = dense_common_covariance(actions, theta, mixing)
    boundary = dense_common_covariance(actions, theta, 1.0)
    return float(zero_mean_gaussian_kl(covariance, boundary))


def ar1_average_variance_factor(coefficient: float, updates: int) -> float:
    """Variance of the mean of a stationary unit-variance AR(1) sequence."""

    if not -1.0 < coefficient < 1.0:
        raise ValueError("coefficient must be in (-1, 1)")
    if updates < 1:
        raise ValueError("updates must be positive")
    lags = np.arange(1, updates, dtype=float)
    weighted = np.sum((updates - lags) * coefficient**lags)
    return float((updates + 2.0 * weighted) / (updates * updates))


def terminal_mean_risk(
    theta: float, q: int, coefficient: float, updates: int
) -> float:
    """Exact mean-estimation risk for private noise plus an AR common factor."""

    if theta < 0.0 or q < 1:
        raise ValueError("theta must be nonnegative and q positive")
    return 1.0 / (q * updates) + theta * ar1_average_variance_factor(
        coefficient, updates
    )


def asymptotic_risk_coefficient(theta: float, q: int, coefficient: float) -> float:
    if theta < 0.0 or q < 1 or not -1.0 < coefficient < 1.0:
        raise ValueError("invalid risk parameters")
    return 1.0 / q + theta * (1.0 + coefficient) / (1.0 - coefficient)


def update_rate(
    message_rate: float, environment_rate: float, overhead: int, q: int, b: int
) -> float:
    if min(message_rate, environment_rate) <= 0 or min(overhead + q, b) <= 0:
        raise ValueError("rates and costs must be positive")
    return min(message_rate / (overhead + q), environment_rate / b)


def budget_risk_coefficient(
    theta: float,
    mixing: float,
    message_rate: float,
    environment_rate: float,
    overhead: int,
    q: int,
    b: int,
) -> float:
    rate = update_rate(message_rate, environment_rate, overhead, q, b)
    return asymptotic_risk_coefficient(theta, q, mixing**b) / rate


def amortization_scale_lower(
    probe_updates: int, opportunity_constant: float, oracle_gap: float
) -> float:
    """Scale needed for O(n/s^2) probing cost to fit in a gap/s gain."""

    if probe_updates < 0 or opportunity_constant < 0.0 or oracle_gap <= 0.0:
        raise ValueError("invalid threshold parameters")
    return probe_updates * opportunity_constant / oracle_gap
