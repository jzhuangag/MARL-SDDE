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


@dataclass(frozen=True)
class CausalSchedule:
    incurred_risk: float
    refresh_count: int
    refresh_fraction: float
    final_resource_debt: float
    maximum_resource_debt: float
    selected_refresh_value: float
    refresh_events: tuple[bool, ...]


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


def causal_resource_schedule(
    stale_risk: Array,
    *,
    fresh_variance: float,
    maximum_refresh_count: int,
    average_refresh_budget: float,
    risk_tradeoff: float,
) -> CausalSchedule:
    """Run the scalar resource-debt policy using only current-event risk.

    The finite-horizon cap guarantees at most ``maximum_refresh_count`` costly
    measurements.  The virtual queue prices the declared average refresh rate.
    No future risk value is inspected by the action loop.
    """

    from .freshness_sensing import (
        choose_budgeted_freshness_refresh,
        optimal_fusion_certificate,
    )

    risk = np.asarray(stale_risk, dtype=float)
    if risk.ndim != 1 or not np.isfinite(risk).all() or (risk < 0.0).any():
        raise ValueError("stale_risk must be a finite nonnegative vector")
    if maximum_refresh_count < 0 or maximum_refresh_count > len(risk):
        raise ValueError("maximum_refresh_count is invalid")
    if not 0.0 <= average_refresh_budget <= 1.0:
        raise ValueError("average_refresh_budget must lie in [0, 1]")
    debt = np.zeros(1, dtype=float)
    used = 0
    maximum_debt = 0.0
    total_risk = 0.0
    total_value = 0.0
    refresh_events: list[bool] = []
    for stale in risk:
        certificate = optimal_fusion_certificate(
            birth_variance=float(stale),
            fresh_variance=fresh_variance,
            birth_bias_upper=0.0,
        )
        decision = choose_budgeted_freshness_refresh(
            certificate,
            resource_debts=debt,
            refresh_costs=np.ones(1),
            average_budgets=np.asarray([average_refresh_budget]),
            risk_tradeoff=risk_tradeoff,
            hard_budget_feasible=used < maximum_refresh_count,
        )
        debt = np.asarray(decision.resource_debts_after)
        maximum_debt = max(maximum_debt, float(debt[0]))
        total_risk += decision.incurred_mse_upper
        refresh_events.append(decision.refresh)
        if decision.refresh:
            used += 1
            total_value += decision.refresh_value
    return CausalSchedule(
        incurred_risk=float(total_risk),
        refresh_count=used,
        refresh_fraction=used / len(risk),
        final_resource_debt=float(debt[0]),
        maximum_resource_debt=maximum_debt,
        selected_refresh_value=float(total_value),
        refresh_events=tuple(refresh_events),
    )


def causal_resource_schedule_fast(
    stale_risk: Array,
    *,
    fresh_variance: float,
    maximum_refresh_count: int,
    average_refresh_budget: float,
    risk_tradeoff: float,
) -> CausalSchedule:
    """Scalar-equivalent implementation for large research grids.

    This avoids allocating NumPy arrays and dataclasses at every event.  It is
    required to match :func:`causal_resource_schedule` field for field.
    """

    risk = np.asarray(stale_risk, dtype=float)
    variance = float(fresh_variance)
    tradeoff = float(risk_tradeoff)
    if risk.ndim != 1 or not np.isfinite(risk).all() or (risk < 0.0).any():
        raise ValueError("stale_risk must be a finite nonnegative vector")
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("fresh_variance must be finite and nonnegative")
    if not math.isfinite(tradeoff) or tradeoff < 0.0:
        raise ValueError("risk_tradeoff must be finite and nonnegative")
    if maximum_refresh_count < 0 or maximum_refresh_count > len(risk):
        raise ValueError("maximum_refresh_count is invalid")
    if not 0.0 <= average_refresh_budget <= 1.0:
        raise ValueError("average_refresh_budget must lie in [0, 1]")
    debt = 0.0
    used = 0
    maximum_debt = 0.0
    total_risk = 0.0
    total_value = 0.0
    refresh_events: list[bool] = []
    for stale in risk:
        denominator = float(stale) + variance
        if denominator == 0.0:
            refresh_risk = 0.0
        else:
            refresh_risk = float(stale) * variance / denominator
        value = float(stale) - refresh_risk
        refresh = bool(
            used < maximum_refresh_count and tradeoff * value > debt
        )
        if refresh:
            used += 1
            total_risk += refresh_risk
            total_value += value
            debt = max(0.0, debt + 1.0 - average_refresh_budget)
        else:
            total_risk += float(stale)
            debt = max(0.0, debt - average_refresh_budget)
        maximum_debt = max(maximum_debt, debt)
        refresh_events.append(refresh)
    return CausalSchedule(
        incurred_risk=float(total_risk),
        refresh_count=used,
        refresh_fraction=used / len(risk),
        final_resource_debt=float(debt),
        maximum_resource_debt=float(maximum_debt),
        selected_refresh_value=float(total_value),
        refresh_events=tuple(refresh_events),
    )
