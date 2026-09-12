"""Poisson decomposition for delayed multiplicative finite-state Markov TD.

The routines in this module are algebraic proof aids.  They keep the
sample--iterate product ``F(Z_t) y_t`` intact and split it into a martingale
transform, two boundary terms, and an iterate-increment remainder.
"""

from __future__ import annotations

import numpy as np

from experiments.dependence_delay_linear.t036_affine_markov_moments import (
    validate_markov_transition,
)


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Return the unique stationary distribution of an ergodic finite chain."""

    probability = validate_markov_transition(transition)
    modes = probability.shape[0]
    system = np.vstack((probability.T - np.eye(modes), np.ones(modes)))
    target = np.append(np.zeros(modes), 1.0)
    stationary, _, rank, _ = np.linalg.lstsq(system, target, rcond=None)
    if rank < modes or np.min(stationary) <= 0.0:
        raise ValueError("transition must have a unique positive stationary law")
    stationary = stationary / stationary.sum()
    if not np.allclose(stationary @ probability, stationary, atol=1e-11):
        raise ValueError("failed to compute a stationary distribution")
    return stationary


def solve_centered_poisson(
    *,
    transition: np.ndarray,
    field: np.ndarray,
    stationary: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Solve ``H - P H = F`` with the canonical constraint ``pi H = 0``.

    ``field`` may be scalar-, vector-, or matrix-valued; its first axis is the
    Markov mode.  Centering is checked rather than silently imposed.
    """

    probability = validate_markov_transition(transition)
    modes = probability.shape[0]
    values = np.asarray(field, dtype=float)
    if values.ndim < 1 or values.shape[0] != modes:
        raise ValueError("field first axis must match the Markov modes")
    invariant = (
        stationary_distribution(probability)
        if stationary is None
        else np.asarray(stationary, dtype=float)
    )
    if invariant.shape != (modes,) or np.any(invariant <= 0.0):
        raise ValueError("stationary distribution must be positive on every mode")
    invariant = invariant / invariant.sum()
    if not np.allclose(invariant @ probability, invariant, atol=1e-11):
        raise ValueError("stationary distribution is not invariant")
    centered_mean = np.tensordot(invariant, values, axes=(0, 0))
    if not np.allclose(centered_mean, 0.0, atol=1e-11):
        raise ValueError("field must be centered under the stationary law")

    fundamental = np.linalg.inv(
        np.eye(modes) - probability + np.ones((modes, 1)) @ invariant[None, :]
    )
    flat = values.reshape(modes, -1)
    solution = (fundamental @ flat).reshape(values.shape)
    predicted = np.tensordot(probability, solution, axes=(1, 0))
    residual = solution - predicted - values
    if not np.allclose(residual, 0.0, atol=1e-10):
        raise RuntimeError("Poisson solve failed its residual check")
    if not np.allclose(
        np.tensordot(invariant, solution, axes=(0, 0)), 0.0, atol=1e-10
    ):
        raise RuntimeError("Poisson solution violates the canonical centering")
    return {
        "stationary": invariant,
        "solution": solution,
        "predicted_solution": predicted,
        "fundamental_matrix": fundamental,
    }


def _apply_field(field_value: np.ndarray, vector: np.ndarray) -> np.ndarray:
    value = np.asarray(field_value, dtype=float)
    argument = np.asarray(vector, dtype=float)
    if value.ndim == 0:
        return np.asarray(value * argument, dtype=float)
    if value.ndim == 1:
        if argument.ndim != 0:
            raise ValueError("vector-valued fields require scalar multipliers")
        return value * argument
    if value.ndim == 2:
        return value @ argument
    raise ValueError("only scalar, vector, and matrix fields are supported")


def pathwise_poisson_decomposition(
    *,
    transition: np.ndarray,
    field: np.ndarray,
    poisson_solution: np.ndarray,
    mode_path: np.ndarray,
    predictable_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose ``sum_t F(Z_t)y_t`` along one path.

    ``mode_path`` has length ``T+1`` and ``predictable_vectors`` has length
    ``T``.  In delayed TD, use ``y_t=e_{t-D}``; predictability then holds with
    respect to the sigma-field available before ``Z_(t+1)`` is drawn.
    """

    probability = validate_markov_transition(transition)
    values = np.asarray(field, dtype=float)
    solution = np.asarray(poisson_solution, dtype=float)
    path = np.asarray(mode_path, dtype=int)
    vectors = np.asarray(predictable_vectors, dtype=float)
    modes = probability.shape[0]
    if values.shape != solution.shape or values.shape[0] != modes:
        raise ValueError("field and Poisson solution shapes must agree")
    if path.ndim != 1 or path.size < 2 or np.any((path < 0) | (path >= modes)):
        raise ValueError("mode_path must contain valid modes and length at least two")
    horizon = path.size - 1
    if vectors.shape[0] != horizon:
        raise ValueError("one predictable vector is required per summand")
    predicted = np.tensordot(probability, solution, axes=(1, 0))

    target = np.zeros_like(_apply_field(values[path[0]], vectors[0]))
    martingale = np.zeros_like(target)
    for time in range(horizon):
        target += _apply_field(values[path[time]], vectors[time])
        difference = solution[path[time + 1]] - predicted[path[time]]
        martingale += _apply_field(difference, vectors[time])

    boundary = _apply_field(solution[path[0]], vectors[0]) - _apply_field(
        solution[path[-1]], vectors[-1]
    )
    increment = np.zeros_like(target)
    for time in range(1, horizon):
        increment += _apply_field(
            solution[path[time]], vectors[time] - vectors[time - 1]
        )
    reconstructed = martingale + boundary + increment
    return {
        "target": target,
        "martingale": martingale,
        "boundary": boundary,
        "increment": increment,
        "reconstructed": reconstructed,
        "residual": target - reconstructed,
    }


def pathwise_weighted_poisson_decomposition(
    *,
    transition: np.ndarray,
    field: np.ndarray,
    poisson_solution: np.ndarray,
    mode_path: np.ndarray,
    predictable_vectors: np.ndarray,
    left_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exact terminal-response version of the Poisson decomposition.

    For matrix fields, this decomposes ``sum_t W_t F(Z_t)y_t``.  The two
    increment terms separately record variation of the deterministic impulse
    response ``W_t`` and variation of the delayed iterate ``y_t``.
    """

    probability = validate_markov_transition(transition)
    values = np.asarray(field, dtype=float)
    solution = np.asarray(poisson_solution, dtype=float)
    path = np.asarray(mode_path, dtype=int)
    vectors = np.asarray(predictable_vectors, dtype=float)
    weights = np.asarray(left_weights, dtype=float)
    modes = probability.shape[0]
    if values.ndim != 3 or solution.shape != values.shape:
        raise ValueError("weighted decomposition requires matching matrix fields")
    if path.ndim != 1 or path.size < 2 or np.any((path < 0) | (path >= modes)):
        raise ValueError("mode_path must contain valid modes and length at least two")
    horizon = path.size - 1
    field_output, field_input = values.shape[1:]
    if vectors.shape != (horizon, field_input):
        raise ValueError("predictable vector shape is incompatible with the field")
    if weights.ndim != 3 or weights.shape[0] != horizon or weights.shape[2] != field_output:
        raise ValueError("left_weights must have shape (T, output, field_output)")
    predicted = np.tensordot(probability, solution, axes=(1, 0))
    output_dimension = weights.shape[1]
    target = np.zeros(output_dimension)
    martingale = np.zeros(output_dimension)
    for time in range(horizon):
        target += weights[time] @ values[path[time]] @ vectors[time]
        difference = solution[path[time + 1]] - predicted[path[time]]
        martingale += weights[time] @ difference @ vectors[time]
    boundary = (
        weights[0] @ solution[path[0]] @ vectors[0]
        - weights[-1] @ solution[path[-1]] @ vectors[-1]
    )
    weight_increment = np.zeros(output_dimension)
    vector_increment = np.zeros(output_dimension)
    for time in range(1, horizon):
        weight_increment += (
            (weights[time] - weights[time - 1])
            @ solution[path[time]]
            @ vectors[time]
        )
        vector_increment += (
            weights[time - 1]
            @ solution[path[time]]
            @ (vectors[time] - vectors[time - 1])
        )
    reconstructed = martingale + boundary + weight_increment + vector_increment
    return {
        "target": target,
        "martingale": martingale,
        "boundary": boundary,
        "weight_increment": weight_increment,
        "vector_increment": vector_increment,
        "reconstructed": reconstructed,
        "residual": target - reconstructed,
    }


def poisson_moment_constants(
    *, transition: np.ndarray, poisson_solution: np.ndarray
) -> dict[str, float | np.ndarray]:
    """Return exact finite-state constants for the martingale and remainder.

    For a matrix-valued solution, ``variance_operator[z]`` equals
    ``sum_b P[z,b] Delta[z,b].T Delta[z,b]``.  Consequently

    ``E[||sum M_(t+1)y_t||^2] <= v_H sum E[||y_t||^2]``

    for every predictable vector sequence, where ``v_H`` is the maximum
    eigenvalue reported here.
    """

    probability = validate_markov_transition(transition)
    solution = np.asarray(poisson_solution, dtype=float)
    modes = probability.shape[0]
    if solution.shape[0] != modes or solution.ndim not in (1, 2, 3):
        raise ValueError("solution must be scalar-, vector-, or matrix-valued")
    predicted = np.tensordot(probability, solution, axes=(1, 0))
    def operator_norm(value: np.ndarray) -> float:
        array = np.asarray(value)
        if array.ndim == 0:
            return float(abs(array))
        if array.ndim == 1:
            return float(np.linalg.norm(array))
        return float(np.linalg.norm(array, ord=2))

    h_norm = max(operator_norm(solution[mode]) for mode in range(modes))

    if solution.ndim == 1:
        variance_operator = np.zeros(modes)
        for source in range(modes):
            for target in range(modes):
                delta = solution[target] - predicted[source]
                variance_operator[source] += probability[source, target] * delta**2
        variance_constant = float(np.max(variance_operator))
    elif solution.ndim == 2:
        dimension = solution.shape[1]
        variance_operator = np.zeros((modes, dimension, dimension))
        for source in range(modes):
            for target in range(modes):
                delta = solution[target] - predicted[source]
                variance_operator[source] += probability[source, target] * np.outer(
                    delta, delta
                )
        variance_constant = float(
            max(np.max(np.linalg.eigvalsh(matrix)) for matrix in variance_operator)
        )
    else:
        input_dimension = solution.shape[2]
        variance_operator = np.zeros((modes, input_dimension, input_dimension))
        for source in range(modes):
            for target in range(modes):
                delta = solution[target] - predicted[source]
                variance_operator[source] += (
                    probability[source, target] * delta.T @ delta
                )
        variance_constant = float(
            max(np.max(np.linalg.eigvalsh(matrix)) for matrix in variance_operator)
        )
    return {
        "h_max": h_norm,
        "variance_operator": variance_operator,
        "variance_constant": variance_constant,
    }
