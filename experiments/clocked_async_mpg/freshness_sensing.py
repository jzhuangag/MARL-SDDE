"""Lyapunov-scheduled freshness sensing for clocked policy-block updates.

The module separates two decisions that the failed scalar packet filter mixed
together.  A completed packet always supplies a birth-gradient estimator.  A
fully charged arrival-time refresh can additionally supply an independent
current-gradient estimator.  If refreshed, the two estimators are fused with
the exact minimizer of a conditional MSE upper bound.  A virtual queue decides
whether the reduction in estimation risk is worth the refresh cost.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def _nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class FusionCertificate:
    fresh_weight: float
    birth_weight: float
    no_refresh_mse_upper: float
    refresh_mse_upper: float
    refresh_value: float


@dataclass(frozen=True)
class FreshnessDecision:
    refresh: bool
    refresh_value: float
    priced_value: float
    risk_debt_before: float
    risk_debt_after: float
    incurred_mse_upper: float
    refresh_cost: float
    fresh_weight: float


@dataclass(frozen=True)
class BudgetedFreshnessDecision:
    refresh: bool
    refresh_value: float
    risk_weighted_value: float
    resource_price: float
    resource_debts_before: tuple[float, ...]
    resource_debts_after: tuple[float, ...]
    incurred_costs: tuple[float, ...]
    incurred_mse_upper: float
    fresh_weight: float


def cross_policy_bias_upper(
    teammate_mean_kls: Array,
    *,
    cross_gradient_lipschitz: float,
) -> float:
    """Convert teammate policy drift into a current-gradient bias bound.

    The sum of Pinsker bounds controls joint-policy total-variation drift on a
    fixed reference distribution.  ``cross_gradient_lipschitz`` converts that
    observable drift into the norm of the strategic partial-gradient change.
    """

    values = np.asarray(teammate_mean_kls, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < -1e-12).any():
        raise ValueError("teammate_mean_kls must be a finite nonnegative vector")
    coefficient = _nonnegative("cross_gradient_lipschitz", cross_gradient_lipschitz)
    total_variation_upper = float(
        np.sum(np.sqrt(0.5 * np.maximum(values, 0.0)))
    )
    return coefficient * total_variation_upper


def optimal_fusion_certificate(
    *,
    birth_variance: float,
    fresh_variance: float,
    birth_bias_upper: float,
) -> FusionCertificate:
    """Return the MSE-optimal fresh weight and its exact upper-bound value.

    The birth estimator is unbiased for the packet-birth gradient and can have
    bias at most ``birth_bias_upper`` relative to the arrival gradient.  The
    arrival-fresh estimator is conditionally unbiased.  Their centered noises
    are independent with trace-variance bounds ``birth_variance`` and
    ``fresh_variance``.
    """

    birth_variance = _nonnegative("birth_variance", birth_variance)
    fresh_variance = _nonnegative("fresh_variance", fresh_variance)
    birth_bias_upper = _nonnegative("birth_bias_upper", birth_bias_upper)
    stale_risk = birth_variance + birth_bias_upper * birth_bias_upper
    denominator = stale_risk + fresh_variance
    if denominator == 0.0:
        fresh_weight = 0.5
        refresh_risk = 0.0
    else:
        fresh_weight = stale_risk / denominator
        refresh_risk = stale_risk * fresh_variance / denominator
    refresh_value = stale_risk - refresh_risk
    return FusionCertificate(
        fresh_weight=float(fresh_weight),
        birth_weight=float(1.0 - fresh_weight),
        no_refresh_mse_upper=float(stale_risk),
        refresh_mse_upper=float(refresh_risk),
        refresh_value=float(refresh_value),
    )


def fuse_gradient_estimates(
    birth_gradient: Array,
    fresh_gradient: Array,
    certificate: FusionCertificate,
) -> Array:
    birth = np.asarray(birth_gradient, dtype=float)
    fresh = np.asarray(fresh_gradient, dtype=float)
    if birth.ndim != 1 or birth.shape != fresh.shape:
        raise ValueError("birth and fresh gradients must be matching vectors")
    if not np.isfinite(birth).all() or not np.isfinite(fresh).all():
        raise ValueError("gradient estimates must be finite")
    return certificate.birth_weight * birth + certificate.fresh_weight * fresh


def update_risk_debt(
    debt: float,
    *,
    incurred_mse_upper: float,
    mse_budget: float,
) -> float:
    debt = _nonnegative("debt", debt)
    incurred = _nonnegative("incurred_mse_upper", incurred_mse_upper)
    budget = _nonnegative("mse_budget", mse_budget)
    return float(max(0.0, debt + incurred - budget))


def update_resource_debts(
    debts: Array,
    *,
    incurred_costs: Array,
    average_budgets: Array,
) -> Array:
    """Update nonnegative virtual queues for multiple refresh resources."""

    queue = np.asarray(debts, dtype=float)
    costs = np.asarray(incurred_costs, dtype=float)
    budgets = np.asarray(average_budgets, dtype=float)
    if queue.ndim != 1 or queue.shape != costs.shape or queue.shape != budgets.shape:
        raise ValueError("resource vectors must be matching one-dimensional arrays")
    if not all(np.isfinite(value).all() for value in (queue, costs, budgets)):
        raise ValueError("resource vectors must be finite")
    if (queue < 0.0).any() or (costs < 0.0).any() or (budgets < 0.0).any():
        raise ValueError("resource vectors must be nonnegative")
    return np.maximum(0.0, queue + costs - budgets)


def choose_budgeted_freshness_refresh(
    certificate: FusionCertificate,
    *,
    resource_debts: Array,
    refresh_costs: Array,
    average_budgets: Array,
    risk_tradeoff: float,
    hard_budget_feasible: bool = True,
) -> BudgetedFreshnessDecision:
    """Minimize resource drift plus weighted conditional estimation risk.

    For resource queues ``Z_r`` and refresh costs ``c_r``, the action-dependent
    drift-plus-risk term is ``u * (sum_r Z_r c_r - V R)``.  A refresh is
    purchased iff its risk-weighted value exceeds its resource shadow price
    and a finite-horizon hard budget permits it.
    """

    debts = np.asarray(resource_debts, dtype=float)
    costs = np.asarray(refresh_costs, dtype=float)
    budgets = np.asarray(average_budgets, dtype=float)
    tradeoff = _nonnegative("risk_tradeoff", risk_tradeoff)
    if debts.ndim != 1 or debts.shape != costs.shape or debts.shape != budgets.shape:
        raise ValueError("resource vectors must be matching one-dimensional arrays")
    if not all(np.isfinite(value).all() for value in (debts, costs, budgets)):
        raise ValueError("resource vectors must be finite")
    if (debts < 0.0).any() or (costs < 0.0).any() or (budgets < 0.0).any():
        raise ValueError("resource vectors must be nonnegative")
    resource_price = float(debts @ costs)
    risk_weighted_value = tradeoff * certificate.refresh_value
    refresh = bool(hard_budget_feasible and risk_weighted_value > resource_price)
    incurred_costs = costs if refresh else np.zeros_like(costs)
    debts_after = update_resource_debts(
        debts,
        incurred_costs=incurred_costs,
        average_budgets=budgets,
    )
    incurred_risk = (
        certificate.refresh_mse_upper
        if refresh
        else certificate.no_refresh_mse_upper
    )
    return BudgetedFreshnessDecision(
        refresh=refresh,
        refresh_value=certificate.refresh_value,
        risk_weighted_value=float(risk_weighted_value),
        resource_price=resource_price,
        resource_debts_before=tuple(float(value) for value in debts),
        resource_debts_after=tuple(float(value) for value in debts_after),
        incurred_costs=tuple(float(value) for value in incurred_costs),
        incurred_mse_upper=float(incurred_risk),
        fresh_weight=certificate.fresh_weight if refresh else 0.0,
    )


def choose_freshness_refresh(
    certificate: FusionCertificate,
    *,
    risk_debt: float,
    refresh_cost: float,
    cost_tradeoff: float,
    mse_budget: float,
) -> FreshnessDecision:
    """Apply the exact drift-plus-penalty refresh threshold.

    The refresh action reduces the slot's MSE upper bound by
    ``certificate.refresh_value`` and incurs ``refresh_cost``.  Minimizing the
    one-step quadratic-drift upper bound plus priced cost refreshes iff

        risk_debt * refresh_value > cost_tradeoff * refresh_cost.

    Ties deterministically do not refresh.
    """

    debt = _nonnegative("risk_debt", risk_debt)
    cost = _nonnegative("refresh_cost", refresh_cost)
    tradeoff = _nonnegative("cost_tradeoff", cost_tradeoff)
    budget = _nonnegative("mse_budget", mse_budget)
    priced_value = debt * certificate.refresh_value - tradeoff * cost
    refresh = priced_value > 0.0
    incurred = (
        certificate.refresh_mse_upper
        if refresh
        else certificate.no_refresh_mse_upper
    )
    debt_after = update_risk_debt(
        debt,
        incurred_mse_upper=incurred,
        mse_budget=budget,
    )
    return FreshnessDecision(
        refresh=refresh,
        refresh_value=certificate.refresh_value,
        priced_value=float(priced_value),
        risk_debt_before=debt,
        risk_debt_after=debt_after,
        incurred_mse_upper=incurred,
        refresh_cost=cost if refresh else 0.0,
        fresh_weight=certificate.fresh_weight if refresh else 0.0,
    )


def smooth_potential_progress_lower_bound(
    *,
    gradient_norm: float,
    mse_upper: float,
    learning_rate: float,
    smoothness: float,
) -> float:
    """One-update expected potential-progress lower bound.

    For a smooth maximization objective and an estimator with conditional MSE
    at most ``mse_upper``, Cauchy--Schwarz and smoothness give the returned
    conservative bound for the update ``theta += learning_rate * estimator``.
    """

    gradient_norm = _nonnegative("gradient_norm", gradient_norm)
    mse_upper = _nonnegative("mse_upper", mse_upper)
    learning_rate = _nonnegative("learning_rate", learning_rate)
    smoothness = _nonnegative("smoothness", smoothness)
    error_norm = math.sqrt(mse_upper)
    return float(
        learning_rate * gradient_norm * (gradient_norm - error_norm)
        - 0.5
        * smoothness
        * learning_rate
        * learning_rate
        * (gradient_norm + error_norm) ** 2
    )
