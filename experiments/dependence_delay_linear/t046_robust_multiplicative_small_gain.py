"""Robust finite-horizon small-gain certificate for multiplicative Markov TD."""

from __future__ import annotations

import numpy as np


def finite_horizon_impulse_gain(companion: np.ndarray, horizon: int) -> float:
    """Return ``sum_(k=0)^(horizon-1) ||C^k||_op`` exactly numerically."""

    matrix = np.asarray(companion, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("companion must be square")
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    power = np.eye(matrix.shape[0])
    gain = 0.0
    for _ in range(horizon):
        gain += float(np.linalg.norm(power, ord=2))
        power = power @ matrix
    return gain


def robust_small_gain(
    *, step_size: float, multiplicative_deviation_bound: float, impulse_gain: float
) -> dict[str, float | bool]:
    if step_size <= 0.0 or multiplicative_deviation_bound < 0.0 or impulse_gain < 0.0:
        raise ValueError("invalid step size, deviation bound, or impulse gain")
    gain = step_size * multiplicative_deviation_bound * impulse_gain
    certified = gain < 1.0
    amplification = float(1.0 / (1.0 - gain)) if certified else float("inf")
    perturbation = float(gain / (1.0 - gain)) if certified else float("inf")
    return {
        "small_gain": float(gain),
        "certified": certified,
        "path_amplification": amplification,
        "relative_path_perturbation": perturbation,
    }


def multiplicative_risk_envelope(
    *,
    additive_terminal_risk: float,
    additive_lifted_second_moment_sum: float,
    relative_path_perturbation: float,
) -> dict[str, float]:
    """Convert the pathwise small-gain bound into a terminal-risk envelope."""

    if additive_terminal_risk < 0.0 or additive_lifted_second_moment_sum < 0.0:
        raise ValueError("additive risks must be nonnegative")
    if relative_path_perturbation < 0.0 or not np.isfinite(relative_path_perturbation):
        raise ValueError("a finite certified perturbation factor is required")
    delta_second = (
        relative_path_perturbation**2 * additive_lifted_second_moment_sum
    )
    cross = 2.0 * np.sqrt(additive_terminal_risk * delta_second)
    radius = cross + delta_second
    return {
        "lower": float(max(0.0, additive_terminal_risk - radius)),
        "upper": float(additive_terminal_risk + radius),
        "radius": float(radius),
        "perturbation_second_moment_bound": float(delta_second),
    }


def phase_order_certified(
    *, preferred_upper: float, comparator_lower: float
) -> bool:
    if preferred_upper < 0.0 or comparator_lower < 0.0:
        raise ValueError("risk bounds must be nonnegative")
    return bool(preferred_upper < comparator_lower)


def pathwise_perturbation_audit(
    *,
    companion: np.ndarray,
    additive_inputs: np.ndarray,
    multiplicative_updates: np.ndarray,
    initial_state: np.ndarray,
) -> dict[str, float]:
    """Propagate additive/full paths and report the deterministic inequality.

    The full recursion is ``x[t+1]=C x[t]+u[t]+E[t]x[t]``.  This helper is
    used only to test the theorem; callers remain responsible for mapping
    delayed TD into the lifted ``E[t]`` matrices.
    """

    matrix = np.asarray(companion, dtype=float)
    inputs = np.asarray(additive_inputs, dtype=float)
    perturbations = np.asarray(multiplicative_updates, dtype=float)
    initial = np.asarray(initial_state, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("companion must be square")
    dimension = matrix.shape[0]
    if inputs.ndim != 2 or inputs.shape[1] != dimension:
        raise ValueError("additive_inputs have incompatible shape")
    if perturbations.shape != (inputs.shape[0], dimension, dimension):
        raise ValueError("multiplicative_updates have incompatible shape")
    if initial.shape != (dimension,):
        raise ValueError("initial_state has incompatible shape")
    additive = initial.copy()
    full = initial.copy()
    additive_sup = float(np.linalg.norm(additive))
    full_sup = float(np.linalg.norm(full))
    difference_sup = 0.0
    for time in range(inputs.shape[0]):
        additive = matrix @ additive + inputs[time]
        full = matrix @ full + inputs[time] + perturbations[time] @ full
        additive_sup = max(additive_sup, float(np.linalg.norm(additive)))
        full_sup = max(full_sup, float(np.linalg.norm(full)))
        difference_sup = max(
            difference_sup, float(np.linalg.norm(full - additive))
        )
    epsilon = max(
        (float(np.linalg.norm(update, ord=2)) for update in perturbations),
        default=0.0,
    )
    impulse_gain = finite_horizon_impulse_gain(matrix, inputs.shape[0])
    gain = epsilon * impulse_gain
    certified = gain < 1.0
    bound = gain / (1.0 - gain) * additive_sup if certified else float("inf")
    return {
        "additive_path_sup": additive_sup,
        "full_path_sup": full_sup,
        "difference_path_sup": difference_sup,
        "small_gain": gain,
        "difference_bound": bound,
        "certified": certified,
    }
