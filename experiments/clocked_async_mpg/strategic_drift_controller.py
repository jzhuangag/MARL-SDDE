"""Low-complexity scaling of stale single-flight policy-block proposals.

The controller uses a local quadratic lower bound

    improvement(x) >= gain*x - stale_penalty*x - curvature*x**2

for a proposal scale ``x`` in ``[0, cap]``.  A virtual queue prices the
quadratic-plus-linear certificate penalty.  The resulting drift-plus-penalty
subproblem is one-dimensional and has a closed-form solution; no candidate
grid, Hessian inverse, or generic QP solver is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StrategicDriftDecision:
    """One arrival-time scaling decision and its certified quantities."""

    scale: float
    predicted_gain: float
    certificate_penalty: float
    improvement_lower_bound: float
    debt_before: float
    debt_after: float
    debt_increment: float
    objective_value: float
    effective_cap: float


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def strategic_improvement_lower_bound(
    scale: float,
    directional_gain: float,
    curvature_penalty: float,
    stale_penalty: float,
) -> float:
    """Evaluate the mixed-drift lower bound at a proposal scale."""

    scale = _finite("scale", scale)
    directional_gain = _finite("directional_gain", directional_gain)
    curvature_penalty = _finite("curvature_penalty", curvature_penalty)
    stale_penalty = _finite("stale_penalty", stale_penalty)
    if scale < 0.0 or scale > 1.0:
        raise ValueError("scale must lie in [0, 1]")
    if curvature_penalty < 0.0 or stale_penalty < 0.0:
        raise ValueError("certificate penalties must be nonnegative")
    return float(
        (directional_gain-stale_penalty)*scale
        -curvature_penalty*scale*scale
    )


def no_harm_scale_cap(
    directional_gain: float,
    curvature_penalty: float,
    stale_penalty: float,
    maximum_scale: float = 1.0,
) -> float:
    """Largest scale whose deterministic lower bound is nonnegative."""

    directional_gain = _finite("directional_gain", directional_gain)
    curvature_penalty = _finite("curvature_penalty", curvature_penalty)
    stale_penalty = _finite("stale_penalty", stale_penalty)
    maximum_scale = _finite("maximum_scale", maximum_scale)
    if curvature_penalty < 0.0 or stale_penalty < 0.0:
        raise ValueError("certificate penalties must be nonnegative")
    if maximum_scale < 0.0 or maximum_scale > 1.0:
        raise ValueError("maximum_scale must lie in [0, 1]")
    margin = directional_gain-stale_penalty
    if margin <= 0.0 or maximum_scale == 0.0:
        return 0.0
    if curvature_penalty == 0.0:
        return maximum_scale
    return float(min(maximum_scale, margin/curvature_penalty))


def update_certificate_debt(
    debt: float, certificate_penalty: float, risk_budget: float
) -> float:
    """Apply the projected Lyapunov virtual-queue recursion."""

    debt = _finite("debt", debt)
    certificate_penalty = _finite("certificate_penalty", certificate_penalty)
    risk_budget = _finite("risk_budget", risk_budget)
    if debt < 0.0 or certificate_penalty < 0.0 or risk_budget < 0.0:
        raise ValueError("debt, penalty, and budget must be nonnegative")
    return float(max(0.0, debt+certificate_penalty-risk_budget))


def choose_strategic_drift_scale(
    *,
    directional_gain: float,
    curvature_penalty: float,
    stale_penalty: float,
    debt: float,
    risk_budget: float,
    tradeoff: float,
    maximum_scale: float = 1.0,
    hard_no_harm: bool = False,
) -> StrategicDriftDecision:
    """Solve the scalar drift-plus-penalty action exactly.

    The maximized objective is

    ``tradeoff*directional_gain*x - debt*(curvature*x**2 + stale*x)``.

    ``hard_no_harm`` additionally restricts ``x`` to the interval on which
    the certified improvement lower bound is nonnegative.  Without it, the
    virtual queue permits a declared cumulative certificate-risk budget.
    """

    directional_gain = _finite("directional_gain", directional_gain)
    curvature_penalty = _finite("curvature_penalty", curvature_penalty)
    stale_penalty = _finite("stale_penalty", stale_penalty)
    debt = _finite("debt", debt)
    risk_budget = _finite("risk_budget", risk_budget)
    tradeoff = _finite("tradeoff", tradeoff)
    maximum_scale = _finite("maximum_scale", maximum_scale)
    if curvature_penalty < 0.0 or stale_penalty < 0.0:
        raise ValueError("certificate penalties must be nonnegative")
    if debt < 0.0 or risk_budget < 0.0:
        raise ValueError("debt and risk_budget must be nonnegative")
    if tradeoff <= 0.0:
        raise ValueError("tradeoff must be positive")
    if maximum_scale < 0.0 or maximum_scale > 1.0:
        raise ValueError("maximum_scale must lie in [0, 1]")

    effective_cap = maximum_scale
    if hard_no_harm:
        effective_cap = no_harm_scale_cap(
            directional_gain,
            curvature_penalty,
            stale_penalty,
            maximum_scale,
        )

    linear = tradeoff*directional_gain-debt*stale_penalty
    quadratic = debt*curvature_penalty
    if effective_cap == 0.0 or linear <= 0.0:
        scale = 0.0
    elif quadratic == 0.0:
        scale = effective_cap
    else:
        scale = min(effective_cap, linear/(2.0*quadratic))

    predicted_gain = directional_gain*scale
    certificate_penalty = (
        stale_penalty*scale+curvature_penalty*scale*scale
    )
    lower_bound = predicted_gain-certificate_penalty
    debt_after = update_certificate_debt(
        debt, certificate_penalty, risk_budget
    )
    objective = tradeoff*predicted_gain-debt*certificate_penalty
    return StrategicDriftDecision(
        scale=float(scale),
        predicted_gain=float(predicted_gain),
        certificate_penalty=float(certificate_penalty),
        improvement_lower_bound=float(lower_bound),
        debt_before=float(debt),
        debt_after=float(debt_after),
        debt_increment=float(debt_after-debt),
        objective_value=float(objective),
        effective_cap=float(effective_cap),
    )
