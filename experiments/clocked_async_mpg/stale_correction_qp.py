"""Certified low-complexity correction for stale multi-agent trajectories.

For one completed packet, ``alpha[j]`` continuously tempers teammate ``j``'s
likelihood-ratio factor.  The certified upper objective is a box QP with a
rank-one-plus-diagonal Hessian.  Its KKT system reduces to one monotone scalar
root, so the controller does not scan a catalogue of agent subsets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CorrectionDecision:
    alphas: np.ndarray
    residual_bias_upper: float
    tempered_divergence: float
    variance_upper: float
    risk_upper: float
    priced_objective: float
    iterations: int


@dataclass(frozen=True)
class ReuseCorrectRefreshDecision:
    action: str
    correction: CorrectionDecision
    refresh_risk_upper: float
    refresh_priced_objective: float


def exponential_secant_slope(total_divergence: float) -> float:
    """Return c(S) such that exp(s) <= 1 + c(S)s on 0 <= s <= S."""

    if not math.isfinite(total_divergence) or total_divergence < 0.0:
        raise ValueError("total_divergence must be finite and nonnegative")
    if total_divergence < 1e-8:
        return 1.0 + 0.5 * total_divergence
    return math.expm1(total_divergence) / total_divergence


def solve_tempered_correction_qp(
    *,
    bias_sensitivities: np.ndarray,
    divergence_proxies: np.ndarray,
    integrand_second_moment: float,
    effective_batch_size: float,
    correction_prices: np.ndarray | None = None,
    tolerance: float = 1e-12,
    maximum_iterations: int = 128,
) -> CorrectionDecision:
    """Solve the packet-wise certified box QP through scalar bisection.

    The optimized upper objective is

        (d^T(1-alpha))^2
        + sigma2/n_eff * (1 + c(S) sum_j v_j alpha_j^2)
        + p^T alpha,

    with ``0 <= alpha <= 1`` and ``S=sum(v)``.  The last term is a
    Lyapunov price rather than part of the statistical risk certificate.
    """

    d = np.asarray(bias_sensitivities, dtype=float)
    v = np.asarray(divergence_proxies, dtype=float)
    if d.ndim != 1 or v.ndim != 1 or d.shape != v.shape or d.size == 0:
        raise ValueError("bias and divergence inputs must be same-size vectors")
    if (
        np.any(~np.isfinite(d))
        or np.any(~np.isfinite(v))
        or np.any(d < 0.0)
        or np.any(v < 0.0)
    ):
        raise ValueError("bias and divergence inputs must be finite and nonnegative")
    if np.any((d > 0.0) & (v <= 0.0)):
        raise ValueError("positive sensitivity requires positive divergence proxy")
    if not math.isfinite(integrand_second_moment) or integrand_second_moment <= 0.0:
        raise ValueError("integrand_second_moment must be finite and positive")
    if not math.isfinite(effective_batch_size) or effective_batch_size <= 0.0:
        raise ValueError("effective_batch_size must be finite and positive")
    if tolerance <= 0.0 or maximum_iterations <= 0:
        raise ValueError("invalid solver tolerance or iteration limit")
    if correction_prices is None:
        prices = np.zeros_like(d)
    else:
        prices = np.asarray(correction_prices, dtype=float)
        if prices.shape != d.shape or np.any(~np.isfinite(prices)) or np.any(prices < 0.0):
            raise ValueError("correction prices must match and be nonnegative")

    total_bias = float(np.sum(d))
    total_divergence = float(np.sum(v))
    secant = exponential_secant_slope(total_divergence)
    kappa = integrand_second_moment * secant / effective_batch_size

    def alphas_at(residual: float) -> np.ndarray:
        result = np.zeros_like(d)
        active = v > 0.0
        result[active] = np.clip(
            (d[active] * residual - 0.5 * prices[active])
            / (kappa * v[active]),
            0.0,
            1.0,
        )
        return result

    lower = 0.0
    upper = total_bias
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        midpoint = 0.5 * (lower + upper)
        alphas = alphas_at(midpoint)
        root_value = midpoint + float(np.dot(d, alphas)) - total_bias
        if root_value > 0.0:
            upper = midpoint
        else:
            lower = midpoint
        if upper - lower <= tolerance * max(1.0, total_bias):
            break
    residual = 0.5 * (lower + upper)
    alphas = alphas_at(residual)
    # Recompute the primal residual; this is more accurate near clipped faces.
    residual = max(0.0, total_bias - float(np.dot(d, alphas)))
    tempered_divergence = float(np.dot(v, alphas * alphas))
    variance_upper = integrand_second_moment / effective_batch_size * (
        1.0 + secant * tempered_divergence
    )
    risk_upper = residual * residual + variance_upper
    priced_objective = risk_upper + float(np.dot(prices, alphas))
    return CorrectionDecision(
        alphas=alphas,
        residual_bias_upper=residual,
        tempered_divergence=tempered_divergence,
        variance_upper=variance_upper,
        risk_upper=risk_upper,
        priced_objective=priced_objective,
        iterations=iterations,
    )


def choose_reuse_correct_or_refresh(
    *,
    bias_sensitivities: np.ndarray,
    divergence_proxies: np.ndarray,
    integrand_second_moment: float,
    effective_batch_size: float,
    correction_prices: np.ndarray | None,
    refresh_risk_upper: float,
    refresh_price: float,
    refresh_feasible: bool,
) -> ReuseCorrectRefreshDecision:
    """Compare the optimal reuse/correction QP with a fully charged refresh."""

    if (
        not math.isfinite(refresh_risk_upper)
        or refresh_risk_upper < 0.0
        or not math.isfinite(refresh_price)
        or refresh_price < 0.0
    ):
        raise ValueError("refresh risk and price must be finite and nonnegative")
    correction = solve_tempered_correction_qp(
        bias_sensitivities=bias_sensitivities,
        divergence_proxies=divergence_proxies,
        integrand_second_moment=integrand_second_moment,
        effective_batch_size=effective_batch_size,
        correction_prices=correction_prices,
    )
    refresh_objective = refresh_risk_upper + refresh_price
    action = (
        "refresh"
        if refresh_feasible and refresh_objective < correction.priced_objective
        else "correct"
    )
    return ReuseCorrectRefreshDecision(
        action=action,
        correction=correction,
        refresh_risk_upper=refresh_risk_upper,
        refresh_priced_objective=refresh_objective,
    )


def update_resource_debt(
    debt: float, *, incurred_cost: float, average_budget: float
) -> float:
    """One scalar Lyapunov virtual-queue update."""

    values = (debt, incurred_cost, average_budget)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("debt, cost, and budget must be finite and nonnegative")
    return max(0.0, debt + incurred_cost - average_budget)
