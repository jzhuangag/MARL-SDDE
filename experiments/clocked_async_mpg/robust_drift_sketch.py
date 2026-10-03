"""Robust constant-dimensional Lyapunov drift sketch.

The sensor estimates two linear and three symmetric quadratic coefficients.
Elementwise linear error and an operator-norm quadratic error are added before
solving the same two-dimensional box QP.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .coupled_actor_critic_drift import (
    box_qp_objective,
    solve_two_dimensional_box_qp,
)


@dataclass(frozen=True)
class RobustDriftSketchDecision:
    action: np.ndarray
    estimated_objective: float
    robust_upper_objective: float
    robust_linear: np.ndarray
    robust_quadratic: np.ndarray
    quadratic_inflation: float


def _vector(name: str, value: np.ndarray, *, nonnegative: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite length-two vector")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _matrix(name: str, value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (2, 2) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite two-by-two matrix")
    if not np.allclose(result, result.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return result


def choose_robust_drift_sketch(
    *,
    estimated_linear: np.ndarray,
    estimated_quadratic: np.ndarray,
    linear_absolute_error: np.ndarray,
    quadratic_operator_error: float,
    upper: np.ndarray,
) -> RobustDriftSketchDecision:
    """Minimize a coefficient-wise valid upper confidence quadratic.

    For nonnegative actions, ``|h_hat-h|<=e_h`` and
    ``||Q_hat-Q||_op<=e_Q`` imply

    ``h @ u + .5 u.T @ Q @ u``
    ``<= (h_hat+e_h) @ u + .5 u.T @ (Q_hat+e_Q I) @ u``.
    """

    linear = _vector("estimated_linear", estimated_linear)
    quadratic = _matrix("estimated_quadratic", estimated_quadratic)
    linear_error = _vector(
        "linear_absolute_error", linear_absolute_error, nonnegative=True
    )
    quadratic_error = float(quadratic_operator_error)
    if not math.isfinite(quadratic_error) or quadratic_error < 0.0:
        raise ValueError("quadratic_operator_error must be finite and nonnegative")
    cap = _vector("upper", upper, nonnegative=True)
    if np.linalg.eigvalsh(quadratic)[0] < -quadratic_error - 1e-11:
        raise ValueError("the error radius does not cover a PSD quadratic")
    robust_linear = linear + linear_error
    robust_quadratic = quadratic + quadratic_error * np.eye(2)
    decision = solve_two_dimensional_box_qp(
        linear=robust_linear,
        quadratic=robust_quadratic,
        lower=np.zeros(2),
        upper=cap,
    )
    action = decision.action
    return RobustDriftSketchDecision(
        action=action.copy(),
        estimated_objective=box_qp_objective(action, linear, quadratic),
        robust_upper_objective=float(decision.objective),
        robust_linear=robust_linear.copy(),
        robust_quadratic=robust_quadratic.copy(),
        quadratic_inflation=quadratic_error,
    )


def comparator_excess_bound(
    *,
    comparator: np.ndarray,
    linear_absolute_error: np.ndarray,
    quadratic_operator_error: float,
) -> float:
    """Maximum robust-objective excess over the true comparator drift."""

    action = _vector("comparator", comparator, nonnegative=True)
    linear_error = _vector(
        "linear_absolute_error", linear_absolute_error, nonnegative=True
    )
    quadratic_error = float(quadratic_operator_error)
    if not math.isfinite(quadratic_error) or quadratic_error < 0.0:
        raise ValueError("quadratic_operator_error must be finite and nonnegative")
    return float(2.0 * linear_error @ action + quadratic_error * action @ action)
