"""Warm-started accelerated solver for the T-074 simplex collaboration QP."""

from __future__ import annotations

import math

import numpy as np

from experiments.dependence_delay_linear.t073_continuous_qp_controller import (
    project_simplex,
)


def qp_components(
    *, model_values: np.ndarray, recipient_target: float,
    covariance_of_mean: np.ndarray, recipient: int, debt: float,
    drift_weight: float, variance_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = np.asarray(model_values, dtype=float)
    covariance = np.asarray(covariance_of_mean, dtype=float)
    agents = model.size
    if covariance.shape != (agents, agents) or not 0 <= recipient < agents:
        raise ValueError("QP inputs have incompatible shapes")
    if min(debt, drift_weight, variance_weight) < 0.0:
        raise ValueError("QP weights must be nonnegative")
    local = np.zeros(agents, dtype=float)
    local[recipient] = 1.0
    hessian_half = (np.outer(model, model) + variance_weight * covariance
                    + drift_weight * debt * np.eye(agents))
    linear = -float(recipient_target) * model - drift_weight * debt * local
    return hessian_half, linear


def qp_objective(weights: np.ndarray, hessian_half: np.ndarray, linear: np.ndarray) -> float:
    value = np.asarray(weights, dtype=float)
    return float(value @ hessian_half @ value + 2.0 * linear @ value)


def projected_gradient_residual(
    weights: np.ndarray, hessian_half: np.ndarray, linear: np.ndarray
) -> float:
    gradient = 2.0 * (hessian_half @ weights + linear)
    lipschitz = max(2.0 * float(np.linalg.eigvalsh(hessian_half)[-1]), 1e-12)
    mapped = project_simplex(weights - gradient / lipschitz)
    return float(np.linalg.norm(weights - mapped))


def solve_accelerated_qp(
    *, model_values: np.ndarray, recipient_target: float,
    covariance_of_mean: np.ndarray, recipient: int, debt: float,
    drift_weight: float, variance_weight: float,
    initial_weights: np.ndarray | None = None,
    max_iterations: int = 50, tolerance: float = 1e-7,
) -> tuple[np.ndarray, int, float]:
    """FISTA on the simplex with a projected-gradient residual certificate."""

    hessian_half, linear = qp_components(
        model_values=model_values, recipient_target=recipient_target,
        covariance_of_mean=covariance_of_mean, recipient=recipient, debt=debt,
        drift_weight=drift_weight, variance_weight=variance_weight,
    )
    agents = np.asarray(model_values).size
    local = np.eye(agents)[recipient]
    if initial_weights is None:
        weights = local
    else:
        warm = project_simplex(np.asarray(initial_weights, dtype=float))
        weights = warm if qp_objective(warm, hessian_half, linear) <= qp_objective(
            local, hessian_half, linear
        ) else local
    accelerated = weights.copy()
    momentum = 1.0
    lipschitz = max(2.0 * float(np.linalg.eigvalsh(hessian_half)[-1]), 1e-12)
    for iteration in range(1, max_iterations + 1):
        gradient = 2.0 * (hessian_half @ accelerated + linear)
        updated = project_simplex(accelerated - gradient / lipschitz)
        if qp_objective(updated, hessian_half, linear) > qp_objective(
            weights, hessian_half, linear
        ) + 1e-15:
            accelerated = weights.copy()
            momentum = 1.0
            gradient = 2.0 * (hessian_half @ accelerated + linear)
            updated = project_simplex(accelerated - gradient / lipschitz)
        residual_gradient = 2.0 * (hessian_half @ updated + linear)
        residual = float(np.linalg.norm(
            updated - project_simplex(updated - residual_gradient / lipschitz)
        ))
        if residual <= tolerance:
            return updated, iteration, residual
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum**2))
        accelerated = updated + (momentum - 1.0) / next_momentum * (updated - weights)
        weights = updated
        momentum = next_momentum
    residual_gradient = 2.0 * (hessian_half @ weights + linear)
    residual = float(np.linalg.norm(
        weights - project_simplex(weights - residual_gradient / lipschitz)
    ))
    return weights, max_iterations, residual
