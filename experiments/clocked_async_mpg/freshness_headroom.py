"""Exact equal-cost headroom calculations for freshness sensing.

This module contains no learning outcome and no tuned controller.  It asks a
prior feasibility question: when the conditional stale-gradient risk varies
over completion events, can a state-dependent refresh schedule improve over a
strong, outcome-aware periodic schedule with exactly the same refresh count?
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class EqualCostHeadroom:
    horizon: int
    refresh_count: int
    never_refresh_risk: float
    always_refresh_risk: float
    periodic_risk: float
    oracle_risk: float
    periodic_refresh_value: float
    oracle_refresh_value: float
    oracle_over_periodic_ratio: float
    relative_oracle_improvement: float


def markov_regime_path(
    *,
    horizon: int,
    high_prevalence: float,
    persistence: float,
    seed: int,
) -> NDArray[np.bool_]:
    """Sample a stationary two-state Markov path.

    ``persistence`` is the nontrivial eigenvalue of the transition matrix.
    The construction has stationary high-state probability
    ``high_prevalence`` and remains valid at persistence zero.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    prevalence = float(high_prevalence)
    correlation = float(persistence)
    if not 0.0 < prevalence < 1.0:
        raise ValueError("high_prevalence must lie strictly between zero and one")
    if not 0.0 <= correlation < 1.0:
        raise ValueError("persistence must lie in [0, 1)")
    probability_low_to_high = prevalence * (1.0 - correlation)
    probability_high_to_low = (1.0 - prevalence) * (1.0 - correlation)
    rng = np.random.default_rng(int(seed))
    path = np.empty(horizon, dtype=bool)
    path[0] = bool(rng.random() < prevalence)
    uniforms = rng.random(max(0, horizon - 1))
    for index, uniform in enumerate(uniforms, start=1):
        if path[index - 1]:
            path[index] = not (uniform < probability_high_to_low)
        else:
            path[index] = uniform < probability_low_to_high
    return path


def refresh_value(stale_risk: Array, fresh_variance: float) -> Array:
    """Return the eventwise reduction from optimal fresh/birth fusion."""

    risk = np.asarray(stale_risk, dtype=float)
    variance = float(fresh_variance)
    if risk.ndim != 1 or not np.isfinite(risk).all() or (risk < 0.0).any():
        raise ValueError("stale_risk must be a finite nonnegative vector")
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("fresh_variance must be finite and nonnegative")
    denominator = risk + variance
    values = np.zeros_like(risk)
    positive = denominator > 0.0
    values[positive] = risk[positive] ** 2 / denominator[positive]
    return values


def balanced_periodic_indicator(horizon: int, refresh_count: int) -> Array:
    """Return one exactly charged, maximally even periodic indicator."""

    if horizon <= 0 or refresh_count < 0 or refresh_count > horizon:
        raise ValueError("invalid horizon or refresh_count")
    indicator = np.zeros(horizon, dtype=float)
    if refresh_count:
        positions = np.floor(
            (np.arange(refresh_count, dtype=float) + 0.5)
            * horizon
            / refresh_count
        ).astype(int)
        if len(np.unique(positions)) != refresh_count:
            raise RuntimeError("periodic construction did not preserve the budget")
        indicator[positions] = 1.0
    return indicator


def best_periodic_refresh_value(values: Array, refresh_count: int) -> float:
    """Choose the best cyclic phase of an evenly spaced fixed schedule.

    The phase is selected using the realized path.  This makes the periodic
    comparator stronger than a deployable fixed-phase policy while preserving
    its fixed, state-independent spacing and exact refresh count.
    """

    gains = np.asarray(values, dtype=float)
    if gains.ndim != 1 or not np.isfinite(gains).all():
        raise ValueError("values must be a finite vector")
    indicator = balanced_periodic_indicator(len(gains), refresh_count)
    if refresh_count == 0:
        return 0.0
    correlations = np.fft.ifft(
        np.conjugate(np.fft.fft(indicator)) * np.fft.fft(gains)
    ).real
    return float(np.max(correlations))


def oracle_refresh_value(values: Array, refresh_count: int) -> float:
    """Return the largest attainable value with an exact cardinality budget."""

    gains = np.asarray(values, dtype=float)
    if gains.ndim != 1 or not np.isfinite(gains).all():
        raise ValueError("values must be a finite vector")
    if refresh_count < 0 or refresh_count > len(gains):
        raise ValueError("refresh_count is invalid")
    if refresh_count == 0:
        return 0.0
    selected = np.partition(gains, len(gains) - refresh_count)[-refresh_count:]
    return float(np.sum(selected))


def equal_cost_headroom(
    stale_risk: Array,
    *,
    fresh_variance: float,
    refresh_count: int,
) -> EqualCostHeadroom:
    """Compare oracle and strong periodic sensing under identical charges."""

    risk = np.asarray(stale_risk, dtype=float)
    values = refresh_value(risk, fresh_variance)
    periodic_value = best_periodic_refresh_value(values, refresh_count)
    oracle_value = oracle_refresh_value(values, refresh_count)
    never_risk = float(np.sum(risk))
    always_risk = float(np.sum(risk - values))
    periodic_risk = never_risk - periodic_value
    oracle_risk = never_risk - oracle_value
    if oracle_risk > periodic_risk + 1e-9:
        raise RuntimeError("oracle schedule cannot be worse than periodic")
    ratio = oracle_risk / periodic_risk if periodic_risk > 0.0 else 1.0
    improvement = (
        (periodic_risk - oracle_risk) / periodic_risk
        if periodic_risk > 0.0
        else 0.0
    )
    return EqualCostHeadroom(
        horizon=len(risk),
        refresh_count=refresh_count,
        never_refresh_risk=never_risk,
        always_refresh_risk=always_risk,
        periodic_risk=float(periodic_risk),
        oracle_risk=float(oracle_risk),
        periodic_refresh_value=periodic_value,
        oracle_refresh_value=oracle_value,
        oracle_over_periodic_ratio=float(ratio),
        relative_oracle_improvement=float(improvement),
    )
