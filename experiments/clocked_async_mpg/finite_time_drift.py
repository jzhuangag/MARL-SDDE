"""Exact-gradient finite-time drift tools for asynchronous block updates."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def validate_cross_lipschitz(
    cross_lipschitz: Array, activation_probabilities: Array
) -> tuple[Array, Array]:
    matrix = np.asarray(cross_lipschitz, dtype=float)
    probabilities = np.asarray(activation_probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("cross_lipschitz must be square")
    if probabilities.shape != (matrix.shape[0],):
        raise ValueError("activation probabilities have the wrong shape")
    if (matrix < 0.0).any() or not np.isfinite(matrix).all():
        raise ValueError("cross_lipschitz must be finite and nonnegative")
    if (probabilities <= 0.0).any() or not np.isfinite(probabilities).all():
        raise ValueError("activation probabilities must be finite and positive")
    if not math.isclose(float(np.sum(probabilities)), 1.0, abs_tol=1e-12):
        raise ValueError("activation probabilities must sum to one")
    return matrix, probabilities


def interaction_history_weights(
    cross_lipschitz: Array, activation_probabilities: Array
) -> Array:
    """Return the block-history weights in the expected delay penalty.

    With ``ell_i=sum_j L_ij`` and activation probability ``p_i``, the weight of
    past block ``j`` is ``w_j=sum_i p_i ell_i L_ij``.
    """

    matrix, probabilities = validate_cross_lipschitz(
        cross_lipschitz, activation_probabilities
    )
    row_sums = np.sum(matrix, axis=1)
    return np.asarray((probabilities*row_sums)@matrix, dtype=float)


def maximum_constant_step(
    cross_lipschitz: Array,
    activation_probabilities: Array,
    maximum_delay: int,
) -> float:
    """Largest constant step satisfying every block drift condition.

    The condition is ``L_ii*alpha + D^2*w_i*alpha^2 <= 1``.  A zero-curvature,
    zero-history block imposes no finite restriction.
    """

    if maximum_delay < 0:
        raise ValueError("maximum_delay must be nonnegative")
    matrix, probabilities = validate_cross_lipschitz(
        cross_lipschitz, activation_probabilities
    )
    weights = interaction_history_weights(matrix, probabilities)
    bounds: list[float] = []
    for block in range(matrix.shape[0]):
        linear = float(matrix[block, block])
        quadratic = float(maximum_delay**2*weights[block])
        if quadratic > 0.0:
            root = (
                -linear+math.sqrt(linear*linear+4.0*quadratic)
            )/(2.0*quadratic)
        elif linear > 0.0:
            root = 1.0/linear
        else:
            root = math.inf
        bounds.append(root)
    return float(min(bounds))


def weighted_history_energy(step_vectors: Array, block_weights: Array) -> float:
    """Krasovskii history energy for ``D`` past vector steps.

    Rows are ordered oldest to newest and receive temporal weights ``1,...,D``.
    """

    steps = np.asarray(step_vectors, dtype=float)
    weights = np.asarray(block_weights, dtype=float)
    if steps.ndim != 2 or steps.shape[1:] != weights.shape:
        raise ValueError("step_vectors and block_weights have incompatible shapes")
    if steps.shape[0] == 0:
        return 0.0
    if (weights < 0.0).any():
        raise ValueError("block_weights must be nonnegative")
    time_weights = np.arange(1, steps.shape[0]+1, dtype=float)
    per_step = np.sum((steps*steps)*weights[None, :], axis=1)
    return float(time_weights@per_step)


def expected_quadratic_lyapunov_step(
    state_path: Array,
    delays: NDArray[np.int_],
    curvature: Array,
    activation_probabilities: Array,
    step_size: float,
) -> dict[str, float]:
    """Enumerate one expected asynchronous step on a convex quadratic.

    ``state_path`` contains ``D+1`` states from ``x_(k-D)`` through ``x_k``.
    Candidate block ``i`` uses the exact gradient at ``x_(k-delays[i])``.
    The returned ``expected_next`` and ``certified_upper`` verify the finite-time
    Lyapunov drift lemma for ``f(x)=0.5*x.T@curvature@x``.
    """

    path = np.asarray(state_path, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    probabilities = np.asarray(activation_probabilities, dtype=float)
    delays = np.asarray(delays, dtype=int)
    if path.ndim != 2 or path.shape[0] < 1:
        raise ValueError("state_path must contain at least one state")
    dimension = path.shape[1]
    maximum_delay = path.shape[0]-1
    if curvature.shape != (dimension, dimension):
        raise ValueError("curvature shape does not match the state dimension")
    if not np.allclose(curvature, curvature.T, atol=1e-12, rtol=0.0):
        raise ValueError("curvature must be symmetric")
    if float(np.min(np.linalg.eigvalsh(curvature))) < -1e-12:
        raise ValueError("curvature must be positive semidefinite")
    if delays.shape != (dimension,) or (delays < 0).any() or (
        delays > maximum_delay
    ).any():
        raise ValueError("delays must provide one valid delay per block")
    matrix, probabilities = validate_cross_lipschitz(
        np.abs(curvature), probabilities
    )
    if step_size < 0.0:
        raise ValueError("step_size must be nonnegative")

    current = path[-1]
    past_steps = np.diff(path, axis=0)
    history_weights = interaction_history_weights(matrix, probabilities)
    history = weighted_history_energy(past_steps, history_weights)
    coefficient = 0.5*step_size*maximum_delay
    objective = 0.5*float(current@curvature@current)
    current_lyapunov = objective+coefficient*history
    current_gradient = curvature@current

    next_values: list[float] = []
    for block in range(dimension):
        stale = path[maximum_delay-int(delays[block])]
        stale_gradient = float(curvature[block]@stale)
        step = np.zeros(dimension, dtype=float)
        step[block] = -step_size*stale_gradient
        updated = current+step
        next_steps = np.vstack((past_steps[1:], step)) if maximum_delay else np.empty(
            (0, dimension), dtype=float
        )
        next_history = weighted_history_energy(next_steps, history_weights)
        next_objective = 0.5*float(updated@curvature@updated)
        next_values.append(next_objective+coefficient*next_history)
    expected_next = float(probabilities@np.asarray(next_values))
    stationarity = float(probabilities@(current_gradient*current_gradient))
    certified_upper = current_lyapunov-0.5*step_size*stationarity
    return {
        "certified_upper": certified_upper,
        "current": current_lyapunov,
        "expected_next": expected_next,
        "slack": certified_upper-expected_next,
        "weighted_stationarity": stationarity,
    }


def expected_noisy_quadratic_lyapunov_step(
    state_path: Array,
    delays: NDArray[np.int_],
    curvature: Array,
    activation_probabilities: Array,
    step_size: float,
    noise_standard_deviations: Array,
) -> dict[str, float]:
    """Enumerate the conditional-noise extension using Rademacher innovations.

    Candidate block ``i`` uses ``grad_i(x_stale) +/- sigma_i`` with equal
    probability.  This exactly realizes a centered innovation with conditional
    variance ``sigma_i**2`` and permits a deterministic check of the stochastic
    Lyapunov upper bound.
    """

    path = np.asarray(state_path, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    probabilities = np.asarray(activation_probabilities, dtype=float)
    delays = np.asarray(delays, dtype=int)
    sigma = np.asarray(noise_standard_deviations, dtype=float)
    if path.ndim != 2 or path.shape[0] < 1:
        raise ValueError("state_path must contain at least one state")
    dimension = path.shape[1]
    maximum_delay = path.shape[0]-1
    if curvature.shape != (dimension, dimension):
        raise ValueError("curvature shape does not match the state dimension")
    if not np.allclose(curvature, curvature.T, atol=1e-12, rtol=0.0):
        raise ValueError("curvature must be symmetric")
    if float(np.min(np.linalg.eigvalsh(curvature))) < -1e-12:
        raise ValueError("curvature must be positive semidefinite")
    if delays.shape != (dimension,) or (delays < 0).any() or (
        delays > maximum_delay
    ).any():
        raise ValueError("delays must provide one valid delay per block")
    if (
        sigma.shape != (dimension,)
        or (sigma < 0.0).any()
        or not np.isfinite(sigma).all()
    ):
        raise ValueError("noise standard deviations must be finite and nonnegative")
    matrix, probabilities = validate_cross_lipschitz(
        np.abs(curvature), probabilities
    )
    if step_size < 0.0 or not math.isfinite(step_size):
        raise ValueError("step_size must be finite and nonnegative")

    current = path[-1]
    past_steps = np.diff(path, axis=0)
    history_weights = interaction_history_weights(matrix, probabilities)
    history = weighted_history_energy(past_steps, history_weights)
    coefficient = 0.5*step_size*maximum_delay
    objective = 0.5*float(current@curvature@current)
    current_lyapunov = objective+coefficient*history
    current_gradient = curvature@current

    next_by_block: list[float] = []
    for block in range(dimension):
        stale = path[maximum_delay-int(delays[block])]
        stale_gradient = float(curvature[block]@stale)
        next_by_noise: list[float] = []
        for innovation in (-float(sigma[block]), float(sigma[block])):
            noisy_gradient = stale_gradient+innovation
            step = np.zeros(dimension, dtype=float)
            step[block] = -step_size*noisy_gradient
            updated = current+step
            next_steps = (
                np.vstack((past_steps[1:], step))
                if maximum_delay
                else np.empty((0, dimension), dtype=float)
            )
            next_history = weighted_history_energy(next_steps, history_weights)
            next_objective = 0.5*float(updated@curvature@updated)
            next_by_noise.append(next_objective+coefficient*next_history)
        next_by_block.append(float(np.mean(next_by_noise)))
    expected_next = float(probabilities@np.asarray(next_by_block))
    stationarity = float(probabilities@(current_gradient*current_gradient))
    variance_coefficient = (
        np.diag(matrix)*step_size**2
        +step_size**3*maximum_delay**2*history_weights
    )
    variance_penalty = 0.5*float(
        probabilities@(variance_coefficient*(sigma*sigma))
    )
    certified_upper = (
        current_lyapunov-0.5*step_size*stationarity+variance_penalty
    )
    return {
        "certified_upper": certified_upper,
        "current": current_lyapunov,
        "expected_next": expected_next,
        "slack": certified_upper-expected_next,
        "variance_penalty": variance_penalty,
        "weighted_stationarity": stationarity,
    }
