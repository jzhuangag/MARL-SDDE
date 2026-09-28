"""Finite-catalogue Lyapunov learning-progress selector.

This module is benchmark independent.  It consumes already constructed
simultaneous upper certificates for the affine drift coefficients and exact
post-calibration resource horizons.  Estimation of those certificates is a
separate, domain-specific layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DriftCertificate:
    a_upper: float
    c_upper: float
    confidence: float
    samples: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.a_upper < 1.0:
            raise ValueError("a_upper must lie in [0,1)")
        if self.c_upper < 0.0:
            raise ValueError("c_upper must be nonnegative")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0,1)")
        if self.samples <= 0:
            raise ValueError("samples must be positive")


@dataclass(frozen=True)
class ResourceBudget:
    message_remaining: int
    environment_remaining: int

    def __post_init__(self) -> None:
        if min(self.message_remaining, self.environment_remaining) <= 0:
            raise ValueError("remaining budgets must be positive")


@dataclass(frozen=True)
class ActionCost:
    messages_per_update: int
    environment_per_update: int

    def __post_init__(self) -> None:
        if min(self.messages_per_update, self.environment_per_update) <= 0:
            raise ValueError("action costs must be positive")


@dataclass(frozen=True)
class LearningProgressDecision:
    selected_q: int
    horizons: dict[int, int]
    certified_risks: dict[int, float]

    def to_dict(self) -> dict:
        return asdict(self)


def fit_affine_drift_certificate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    confidence: float,
    noise_scale: float,
    parameter_radius: float,
    ridge: float = 1.0,
    markov_bias: float = 0.0,
) -> DriftCertificate:
    """Fit an affine drift model with a self-normalized upper rectangle.

    The caller supplies a conditionally sub-Gaussian noise scale after any
    blocking/mixing correction and an additive Markov-bias allowance.  The
    latter expands both coordinates and is deliberately explicit rather than
    being hidden inside an i.i.d. standard error.
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("x and y must contain at least three paired observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("drift observations must be finite")
    if np.any(x < 0.0) or np.any(y < 0.0):
        raise ValueError("Lyapunov observations must be nonnegative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if min(noise_scale, parameter_radius, markov_bias) < 0.0 or ridge <= 0.0:
        raise ValueError("scales and bias must be nonnegative; ridge positive")

    design = np.column_stack([x, np.ones_like(x)])
    gram = design.T @ design + ridge * np.eye(2)
    inverse = np.linalg.inv(gram)
    estimate = inverse @ design.T @ y
    delta = 1.0 - confidence
    log_det_ratio = np.linalg.slogdet(gram)[1] - 2.0 * np.log(ridge)
    beta = noise_scale * np.sqrt(
        max(log_det_ratio + 2.0 * np.log(1.0 / delta), 0.0)
    ) + np.sqrt(ridge) * parameter_radius
    coordinate_radii = beta * np.sqrt(np.diag(inverse)) + markov_bias
    upper = estimate + coordinate_radii
    a_upper = float(upper[0])
    c_upper = float(max(upper[1], 0.0))
    if not 0.0 <= a_upper < 1.0:
        raise ValueError("calibration cannot certify a stable contraction factor")
    return DriftCertificate(
        a_upper=a_upper,
        c_upper=c_upper,
        confidence=confidence,
        samples=int(x.size),
    )


def feasible_horizon(budget: ResourceBudget, cost: ActionCost) -> int:
    """Return the exact dual-budget learning horizon."""

    return min(
        budget.message_remaining // cost.messages_per_update,
        budget.environment_remaining // cost.environment_per_update,
    )


def affine_terminal_risk(
    initial_lyapunov: float, horizon: int, certificate: DriftCertificate
) -> float:
    """Evaluate the closed-form affine-drift terminal certificate."""

    if initial_lyapunov < 0.0:
        raise ValueError("initial_lyapunov must be nonnegative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    a = certificate.a_upper
    return float(
        (a**horizon) * initial_lyapunov
        + certificate.c_upper * (1.0 - a**horizon) / (1.0 - a)
    )


def select_learning_progress_action(
    *,
    initial_lyapunov: float,
    budget: ResourceBudget,
    costs: Mapping[int, ActionCost],
    certificates: Mapping[int, DriftCertificate],
) -> LearningProgressDecision:
    """Minimize certified terminal risk over the complete public catalogue."""

    catalogue = sorted(set(costs) & set(certificates))
    if not catalogue or set(costs) != set(certificates):
        raise ValueError("costs and certificates must share one nonempty catalogue")
    horizons: dict[int, int] = {}
    risks: dict[int, float] = {}
    for q in catalogue:
        if q <= 0:
            raise ValueError("participation levels must be positive")
        horizons[q] = feasible_horizon(budget, costs[q])
        if horizons[q] <= 0:
            raise ValueError(f"action q={q} has no feasible learning update")
        risks[q] = affine_terminal_risk(
            initial_lyapunov, horizons[q], certificates[q]
        )
    selected_q = min(catalogue, key=lambda q: (risks[q], q))
    return LearningProgressDecision(
        selected_q=selected_q,
        horizons=horizons,
        certified_risks=risks,
    )
