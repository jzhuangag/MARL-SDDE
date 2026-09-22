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


def rate_balanced_steps(
    cross_lipschitz: Array,
    activation_probabilities: Array,
    maximum_delay: int,
    history_inflation: float = 1.0,
) -> dict[str, Array | float]:
    """Closed-form heterogeneous steps with ``p_i*alpha_i`` equalized.

    For ``alpha_i=c/p_i``, the block stability condition is a scalar quadratic
    in ``c``.  The smallest positive block root is returned together with the
    steps and realized condition values.
    """

    if maximum_delay < 0:
        raise ValueError("maximum_delay must be nonnegative")
    if history_inflation <= 0.0 or not math.isfinite(history_inflation):
        raise ValueError("history_inflation must be finite and positive")
    matrix, probabilities = validate_cross_lipschitz(
        cross_lipschitz, activation_probabilities
    )
    row_sums = np.sum(matrix, axis=1)
    interaction_columns = row_sums@matrix
    roots: list[float] = []
    for block in range(matrix.shape[0]):
        linear = float(matrix[block, block]/probabilities[block])
        quadratic = float(
            history_inflation
            *maximum_delay**2
            *interaction_columns[block]
            /probabilities[block]
        )
        if quadratic > 0.0:
            root = (
                -linear+math.sqrt(linear*linear+4.0*quadratic)
            )/(2.0*quadratic)
        elif linear > 0.0:
            root = 1.0/linear
        else:
            root = math.inf
        roots.append(root)
    descent_scale = float(min(roots))
    steps = descent_scale/probabilities
    history_weights = np.asarray(
        (probabilities*steps*row_sums)@matrix, dtype=float
    )
    conditions = (
        np.diag(matrix)*steps
        +history_inflation
        *maximum_delay**2
        *history_weights
        *steps
    )
    return {
        "conditions": np.asarray(conditions, dtype=float),
        "descent_scale": descent_scale,
        "history_weights": history_weights,
        "step_sizes": np.asarray(steps, dtype=float),
    }


def single_flight_local_steps(
    cross_lipschitz: Array,
    activation_probabilities: Array,
    maximum_delay: int,
    history_inflation: float = 1.0,
) -> dict[str, Array | float]:
    """Closed-form local-curvature steps for one in-flight packet per block.

    A packet's own block is unchanged between birth and completion, so only
    off-diagonal teammate sensitivities enter the stale-history cost.  Steps
    have the form ``alpha_i=scale/L_ii`` with the largest common safe scale.
    """

    if maximum_delay < 0:
        raise ValueError("maximum_delay must be nonnegative")
    if history_inflation <= 0.0 or not math.isfinite(history_inflation):
        raise ValueError("history_inflation must be finite and positive")
    matrix, probabilities = validate_cross_lipschitz(
        cross_lipschitz, activation_probabilities
    )
    diagonal = np.diag(matrix)
    if (diagonal <= 0.0).any():
        raise ValueError("single-flight local scaling requires positive diagonals")
    teammate = matrix.copy()
    np.fill_diagonal(teammate, 0.0)
    teammate_row_sums = np.sum(teammate, axis=1)
    base_steps = 1.0/diagonal
    base_history_weights = np.asarray(
        (probabilities*base_steps*teammate_row_sums)@teammate,
        dtype=float,
    )
    roots: list[float] = []
    for block in range(matrix.shape[0]):
        quadratic = float(
            history_inflation
            *maximum_delay**2
            *base_history_weights[block]
            /diagonal[block]
        )
        if quadratic > 0.0:
            root = (-1.0+math.sqrt(1.0+4.0*quadratic))/(2.0*quadratic)
        else:
            root = 1.0
        roots.append(root)
    scale = float(min(roots))
    steps = scale*base_steps
    history_weights = scale*base_history_weights
    conditions = (
        diagonal*steps
        +history_inflation
        *maximum_delay**2
        *history_weights
        *steps
    )
    return {
        "conditions": np.asarray(conditions, dtype=float),
        "history_weights": np.asarray(history_weights, dtype=float),
        "scale": scale,
        "step_sizes": np.asarray(steps, dtype=float),
        "teammate_lipschitz": teammate,
    }


def single_flight_constant_step(
    cross_lipschitz: Array,
    activation_probabilities: Array,
    maximum_delay: int,
    history_inflation: float = 1.0,
) -> dict[str, Array | float]:
    """Largest certified common step for one in-flight packet per block.

    Unlike :func:`maximum_constant_step`, this certificate removes diagonal
    terms from the stale-history penalty: a block cannot change while its own
    packet is in flight.  Diagonal curvature still enters the ordinary block
    smoothness term.
    """

    if maximum_delay < 0:
        raise ValueError("maximum_delay must be nonnegative")
    if history_inflation <= 0.0 or not math.isfinite(history_inflation):
        raise ValueError("history_inflation must be finite and positive")
    matrix, probabilities = validate_cross_lipschitz(
        cross_lipschitz, activation_probabilities
    )
    teammate = matrix.copy()
    np.fill_diagonal(teammate, 0.0)
    teammate_row_sums = np.sum(teammate, axis=1)
    base_history_weights = np.asarray(
        (probabilities*teammate_row_sums)@teammate, dtype=float
    )
    roots: list[float] = []
    for block in range(matrix.shape[0]):
        linear = float(matrix[block, block])
        quadratic = float(
            history_inflation*maximum_delay**2*base_history_weights[block]
        )
        if quadratic > 0.0:
            root = (
                -linear+math.sqrt(linear*linear+4.0*quadratic)
            )/(2.0*quadratic)
        elif linear > 0.0:
            root = 1.0/linear
        else:
            root = math.inf
        roots.append(root)
    step_size = float(min(roots))
    steps = np.full(matrix.shape[0], step_size, dtype=float)
    history_weights = step_size*base_history_weights
    conditions = (
        np.diag(matrix)*step_size
        +history_inflation
        *maximum_delay**2
        *history_weights
        *step_size
    )
    return {
        "conditions": np.asarray(conditions, dtype=float),
        "history_weights": np.asarray(history_weights, dtype=float),
        "step_size": step_size,
        "step_sizes": steps,
        "teammate_lipschitz": teammate,
    }


def single_flight_pathwise_constant_step(
    cross_lipschitz: Array,
    maximum_delay: int,
    history_inflation: float = 1.0,
) -> dict[str, Array | float]:
    """Largest common step certified for every bounded activation sequence.

    The history weight of block ``j`` is the largest mismatch coefficient
    ``ell_i_off * L_ij_off`` over a possible arriving block ``i``.  This avoids
    any iid event-mark assumption and is therefore the certificate matching
    independent bounded-renewal worker clocks.
    """

    if maximum_delay < 0:
        raise ValueError("maximum_delay must be nonnegative")
    if history_inflation <= 0.0 or not math.isfinite(history_inflation):
        raise ValueError("history_inflation must be finite and positive")
    matrix = np.asarray(cross_lipschitz, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("cross_lipschitz must be square")
    if (matrix < 0.0).any() or not np.isfinite(matrix).all():
        raise ValueError("cross_lipschitz must be finite and nonnegative")
    teammate = matrix.copy()
    np.fill_diagonal(teammate, 0.0)
    teammate_row_sums = np.sum(teammate, axis=1)
    base_history_weights = np.max(
        teammate_row_sums[:, None]*teammate, axis=0
    )
    roots: list[float] = []
    for block in range(matrix.shape[0]):
        linear = float(matrix[block, block])
        quadratic = float(
            history_inflation*maximum_delay**2*base_history_weights[block]
        )
        if quadratic > 0.0:
            root = (
                -linear+math.sqrt(linear*linear+4.0*quadratic)
            )/(2.0*quadratic)
        elif linear > 0.0:
            root = 1.0/linear
        else:
            root = math.inf
        roots.append(root)
    step_size = float(min(roots))
    history_weights = step_size*base_history_weights
    conditions = (
        np.diag(matrix)*step_size
        +history_inflation
        *maximum_delay**2
        *history_weights
        *step_size
    )
    return {
        "conditions": np.asarray(conditions, dtype=float),
        "history_weights": np.asarray(history_weights, dtype=float),
        "step_size": step_size,
        "step_sizes": np.full(matrix.shape[0], step_size, dtype=float),
        "teammate_lipschitz": teammate,
    }


def maximum_constant_step(
    cross_lipschitz: Array,
    activation_probabilities: Array,
    maximum_delay: int,
    history_inflation: float = 1.0,
) -> float:
    """Largest constant step satisfying every block drift condition.

    The condition is
    ``L_ii*alpha + history_inflation*D^2*w_i*alpha^2 <= 1``.  A
    zero-curvature, zero-history block imposes no finite restriction.
    """

    if maximum_delay < 0:
        raise ValueError("maximum_delay must be nonnegative")
    if history_inflation <= 0.0 or not math.isfinite(history_inflation):
        raise ValueError("history_inflation must be finite and positive")
    matrix, probabilities = validate_cross_lipschitz(
        cross_lipschitz, activation_probabilities
    )
    weights = interaction_history_weights(matrix, probabilities)
    bounds: list[float] = []
    for block in range(matrix.shape[0]):
        linear = float(matrix[block, block])
        quadratic = float(
            history_inflation*maximum_delay**2*weights[block]
        )
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


def expected_biased_noisy_quadratic_lyapunov_step(
    state_path: Array,
    delays: NDArray[np.int_],
    curvature: Array,
    activation_probabilities: Array,
    step_size: float,
    conditional_biases: Array,
    noise_standard_deviations: Array,
    young_parameter: float,
) -> dict[str, float]:
    """Enumerate the bias-aware drift bound using two-point innovations.

    The stale scalar-block estimator is ``grad_i(x_stale)+bias_i+/-sigma_i``.
    ``young_parameter`` is the positive ``delta`` used to separate the stale
    gradient mismatch from the conditional bias.
    """

    path = np.asarray(state_path, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    probabilities = np.asarray(activation_probabilities, dtype=float)
    delays = np.asarray(delays, dtype=int)
    bias = np.asarray(conditional_biases, dtype=float)
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
    if bias.shape != (dimension,) or not np.isfinite(bias).all():
        raise ValueError("conditional biases must be finite")
    if (
        sigma.shape != (dimension,)
        or (sigma < 0.0).any()
        or not np.isfinite(sigma).all()
    ):
        raise ValueError("noise standard deviations must be finite and nonnegative")
    if young_parameter <= 0.0 or not math.isfinite(young_parameter):
        raise ValueError("young_parameter must be finite and positive")
    if step_size < 0.0 or not math.isfinite(step_size):
        raise ValueError("step_size must be finite and nonnegative")
    matrix, probabilities = validate_cross_lipschitz(
        np.abs(curvature), probabilities
    )

    inflation = 1.0+young_parameter
    current = path[-1]
    past_steps = np.diff(path, axis=0)
    history_weights = interaction_history_weights(matrix, probabilities)
    history = weighted_history_energy(past_steps, history_weights)
    coefficient = 0.5*step_size*maximum_delay*inflation
    objective = 0.5*float(current@curvature@current)
    current_lyapunov = objective+coefficient*history
    current_gradient = curvature@current

    next_by_block: list[float] = []
    for block in range(dimension):
        stale = path[maximum_delay-int(delays[block])]
        stale_gradient = float(curvature[block]@stale)
        next_by_noise: list[float] = []
        for innovation in (-float(sigma[block]), float(sigma[block])):
            estimator = stale_gradient+float(bias[block])+innovation
            step = np.zeros(dimension, dtype=float)
            step[block] = -step_size*estimator
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
    bias_penalty = 0.5*step_size*(1.0+1.0/young_parameter)*float(
        probabilities@(bias*bias)
    )
    variance_coefficient = (
        np.diag(matrix)*step_size**2
        +inflation*step_size**3*maximum_delay**2*history_weights
    )
    variance_penalty = 0.5*float(
        probabilities@(variance_coefficient*(sigma*sigma))
    )
    certified_upper = (
        current_lyapunov
        -0.5*step_size*stationarity
        +bias_penalty
        +variance_penalty
    )
    return {
        "bias_penalty": bias_penalty,
        "certified_upper": certified_upper,
        "current": current_lyapunov,
        "expected_next": expected_next,
        "slack": certified_upper-expected_next,
        "variance_penalty": variance_penalty,
        "weighted_stationarity": stationarity,
    }


def expected_rate_balanced_quadratic_lyapunov_step(
    state_path: Array,
    delays: NDArray[np.int_],
    curvature: Array,
    activation_probabilities: Array,
    step_sizes: Array,
    history_inflation: float = 1.0,
) -> dict[str, float]:
    """Enumerate one exact-gradient step with heterogeneous block steps."""

    path = np.asarray(state_path, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    probabilities = np.asarray(activation_probabilities, dtype=float)
    delays = np.asarray(delays, dtype=int)
    steps = np.asarray(step_sizes, dtype=float)
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
        steps.shape != (dimension,)
        or (steps < 0.0).any()
        or not np.isfinite(steps).all()
    ):
        raise ValueError("step_sizes must be finite and nonnegative")
    if history_inflation <= 0.0 or not math.isfinite(history_inflation):
        raise ValueError("history_inflation must be finite and positive")
    matrix, probabilities = validate_cross_lipschitz(
        np.abs(curvature), probabilities
    )

    row_sums = np.sum(matrix, axis=1)
    history_weights = np.asarray(
        (probabilities*steps*row_sums)@matrix, dtype=float
    )
    current = path[-1]
    past_steps = np.diff(path, axis=0)
    history = weighted_history_energy(past_steps, history_weights)
    coefficient = 0.5*maximum_delay*history_inflation
    objective = 0.5*float(current@curvature@current)
    current_lyapunov = objective+coefficient*history
    current_gradient = curvature@current

    next_values: list[float] = []
    for block in range(dimension):
        stale = path[maximum_delay-int(delays[block])]
        stale_gradient = float(curvature[block]@stale)
        step = np.zeros(dimension, dtype=float)
        step[block] = -float(steps[block])*stale_gradient
        updated = current+step
        next_steps = (
            np.vstack((past_steps[1:], step))
            if maximum_delay
            else np.empty((0, dimension), dtype=float)
        )
        next_history = weighted_history_energy(next_steps, history_weights)
        next_objective = 0.5*float(updated@curvature@updated)
        next_values.append(next_objective+coefficient*next_history)
    expected_next = float(probabilities@np.asarray(next_values))
    weighted_stationarity = float(
        (probabilities*steps)@(current_gradient*current_gradient)
    )
    certified_upper = current_lyapunov-0.5*weighted_stationarity
    return {
        "certified_upper": certified_upper,
        "current": current_lyapunov,
        "expected_next": expected_next,
        "slack": certified_upper-expected_next,
        "weighted_stationarity": weighted_stationarity,
    }


def expected_single_flight_quadratic_lyapunov_step(
    state_path: Array,
    delays: NDArray[np.int_],
    curvature: Array,
    activation_probabilities: Array,
    step_sizes: Array,
    history_inflation: float = 1.0,
) -> dict[str, float]:
    """Enumerate the exact drift when each packet's own block is fresh."""

    path = np.asarray(state_path, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    probabilities = np.asarray(activation_probabilities, dtype=float)
    delays = np.asarray(delays, dtype=int)
    steps = np.asarray(step_sizes, dtype=float)
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
        steps.shape != (dimension,)
        or (steps < 0.0).any()
        or not np.isfinite(steps).all()
    ):
        raise ValueError("step_sizes must be finite and nonnegative")
    if history_inflation <= 0.0 or not math.isfinite(history_inflation):
        raise ValueError("history_inflation must be finite and positive")
    matrix, probabilities = validate_cross_lipschitz(
        np.abs(curvature), probabilities
    )
    teammate = matrix.copy()
    np.fill_diagonal(teammate, 0.0)
    teammate_row_sums = np.sum(teammate, axis=1)
    history_weights = np.asarray(
        (probabilities*steps*teammate_row_sums)@teammate, dtype=float
    )

    current = path[-1]
    for block in range(dimension):
        stale = path[maximum_delay-int(delays[block])]
        if not math.isclose(
            float(stale[block]), float(current[block]), abs_tol=1e-12
        ):
            raise ValueError("single-flight packet has a stale own block")
    past_steps = np.diff(path, axis=0)
    history = weighted_history_energy(past_steps, history_weights)
    coefficient = 0.5*maximum_delay*history_inflation
    objective = 0.5*float(current@curvature@current)
    current_lyapunov = objective+coefficient*history
    current_gradient = curvature@current

    next_values: list[float] = []
    for block in range(dimension):
        stale = path[maximum_delay-int(delays[block])]
        stale_gradient = float(curvature[block]@stale)
        step = np.zeros(dimension, dtype=float)
        step[block] = -float(steps[block])*stale_gradient
        updated = current+step
        next_steps = (
            np.vstack((past_steps[1:], step))
            if maximum_delay
            else np.empty((0, dimension), dtype=float)
        )
        next_history = weighted_history_energy(next_steps, history_weights)
        next_objective = 0.5*float(updated@curvature@updated)
        next_values.append(next_objective+coefficient*next_history)
    expected_next = float(probabilities@np.asarray(next_values))
    weighted_stationarity = float(
        (probabilities*steps)@(current_gradient*current_gradient)
    )
    certified_upper = current_lyapunov-0.5*weighted_stationarity
    return {
        "certified_upper": certified_upper,
        "current": current_lyapunov,
        "expected_next": expected_next,
        "slack": certified_upper-expected_next,
        "weighted_stationarity": weighted_stationarity,
    }


def expected_single_flight_biased_noisy_quadratic_lyapunov_step(
    state_path: Array,
    delays: NDArray[np.int_],
    curvature: Array,
    activation_probabilities: Array,
    step_sizes: Array,
    conditional_biases: Array,
    noise_standard_deviations: Array,
    young_parameter: float,
) -> dict[str, float]:
    """Bias/noise drift enumeration for self-fresh single-flight packets.

    The estimator for the arriving block is its stale exact gradient plus a
    deterministic conditional bias and a centered two-point innovation.  This
    is the scalar-block algebra used by the fixed-horizon Markov packet
    corollary; diagonal curvature is excluded only from the history term.
    """

    path = np.asarray(state_path, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    probabilities = np.asarray(activation_probabilities, dtype=float)
    delays = np.asarray(delays, dtype=int)
    steps = np.asarray(step_sizes, dtype=float)
    bias = np.asarray(conditional_biases, dtype=float)
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
        steps.shape != (dimension,)
        or (steps < 0.0).any()
        or not np.isfinite(steps).all()
    ):
        raise ValueError("step_sizes must be finite and nonnegative")
    if bias.shape != (dimension,) or not np.isfinite(bias).all():
        raise ValueError("conditional biases must be finite")
    if (
        sigma.shape != (dimension,)
        or (sigma < 0.0).any()
        or not np.isfinite(sigma).all()
    ):
        raise ValueError("noise standard deviations must be finite and nonnegative")
    if young_parameter <= 0.0 or not math.isfinite(young_parameter):
        raise ValueError("young_parameter must be finite and positive")
    matrix, probabilities = validate_cross_lipschitz(
        np.abs(curvature), probabilities
    )
    teammate = matrix.copy()
    np.fill_diagonal(teammate, 0.0)
    teammate_row_sums = np.sum(teammate, axis=1)
    history_weights = np.asarray(
        (probabilities*steps*teammate_row_sums)@teammate, dtype=float
    )
    inflation = 1.0+young_parameter

    current = path[-1]
    for block in range(dimension):
        stale = path[maximum_delay-int(delays[block])]
        if not math.isclose(
            float(stale[block]), float(current[block]), abs_tol=1e-12
        ):
            raise ValueError("single-flight packet has a stale own block")
    past_steps = np.diff(path, axis=0)
    history = weighted_history_energy(past_steps, history_weights)
    coefficient = 0.5*maximum_delay*inflation
    objective = 0.5*float(current@curvature@current)
    current_lyapunov = objective+coefficient*history
    current_gradient = curvature@current

    next_by_block: list[float] = []
    for block in range(dimension):
        stale = path[maximum_delay-int(delays[block])]
        stale_gradient = float(curvature[block]@stale)
        next_by_noise: list[float] = []
        for innovation in (-float(sigma[block]), float(sigma[block])):
            estimator = stale_gradient+float(bias[block])+innovation
            step = np.zeros(dimension, dtype=float)
            step[block] = -float(steps[block])*estimator
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
    weighted_stationarity = float(
        (probabilities*steps)@(current_gradient*current_gradient)
    )
    bias_penalty = 0.5*(1.0+1.0/young_parameter)*float(
        (probabilities*steps)@(bias*bias)
    )
    variance_coefficient = (
        np.diag(matrix)*(steps*steps)
        +inflation*maximum_delay**2*history_weights*(steps*steps)
    )
    variance_penalty = 0.5*float(
        probabilities@(variance_coefficient*(sigma*sigma))
    )
    certified_upper = (
        current_lyapunov
        -0.5*weighted_stationarity
        +bias_penalty
        +variance_penalty
    )
    return {
        "bias_penalty": bias_penalty,
        "certified_upper": certified_upper,
        "current": current_lyapunov,
        "expected_next": expected_next,
        "slack": certified_upper-expected_next,
        "variance_penalty": variance_penalty,
        "weighted_stationarity": weighted_stationarity,
    }


def pathwise_single_flight_biased_noisy_quadratic_lyapunov_step(
    state_path: Array,
    delay: int,
    activated_block: int,
    curvature: Array,
    step_size: float,
    conditional_bias: float,
    noise_standard_deviation: float,
    young_parameter: float,
) -> dict[str, float]:
    """One pathwise block drift under self-fresh bounded-renewal activation."""

    path = np.asarray(state_path, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
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
    if activated_block < 0 or activated_block >= dimension:
        raise ValueError("activated_block is invalid")
    if delay < 0 or delay > maximum_delay:
        raise ValueError("delay is invalid")
    if step_size < 0.0 or not math.isfinite(step_size):
        raise ValueError("step_size must be finite and nonnegative")
    if not math.isfinite(conditional_bias):
        raise ValueError("conditional_bias must be finite")
    if noise_standard_deviation < 0.0 or not math.isfinite(
        noise_standard_deviation
    ):
        raise ValueError("noise_standard_deviation must be finite and nonnegative")
    if young_parameter <= 0.0 or not math.isfinite(young_parameter):
        raise ValueError("young_parameter must be finite and positive")

    matrix = np.abs(curvature)
    teammate = matrix.copy()
    np.fill_diagonal(teammate, 0.0)
    teammate_row_sums = np.sum(teammate, axis=1)
    base_history_weights = np.max(
        teammate_row_sums[:, None]*teammate, axis=0
    )
    history_weights = step_size*base_history_weights
    inflation = 1.0+young_parameter
    current = path[-1]
    stale = path[maximum_delay-delay]
    if not math.isclose(
        float(stale[activated_block]),
        float(current[activated_block]),
        abs_tol=1e-12,
    ):
        raise ValueError("single-flight packet has a stale own block")
    past_steps = np.diff(path, axis=0)
    history = weighted_history_energy(past_steps, history_weights)
    coefficient = 0.5*maximum_delay*inflation
    objective = 0.5*float(current@curvature@current)
    current_lyapunov = objective+coefficient*history
    current_gradient = curvature@current
    stale_gradient = float(curvature[activated_block]@stale)

    next_values: list[float] = []
    for innovation in (
        -float(noise_standard_deviation),
        float(noise_standard_deviation),
    ):
        estimator = stale_gradient+conditional_bias+innovation
        step = np.zeros(dimension, dtype=float)
        step[activated_block] = -step_size*estimator
        updated = current+step
        next_steps = (
            np.vstack((past_steps[1:], step))
            if maximum_delay
            else np.empty((0, dimension), dtype=float)
        )
        next_history = weighted_history_energy(next_steps, history_weights)
        next_objective = 0.5*float(updated@curvature@updated)
        next_values.append(next_objective+coefficient*next_history)
    expected_next = float(np.mean(next_values))
    stationarity = step_size*float(current_gradient[activated_block]**2)
    bias_penalty = (
        0.5*(1.0+1.0/young_parameter)*step_size*conditional_bias**2
    )
    variance_coefficient = (
        matrix[activated_block, activated_block]*step_size**2
        +inflation
        *maximum_delay**2
        *history_weights[activated_block]
        *step_size**2
    )
    variance_penalty = (
        0.5*variance_coefficient*noise_standard_deviation**2
    )
    certified_upper = (
        current_lyapunov
        -0.5*stationarity
        +bias_penalty
        +variance_penalty
    )
    return {
        "bias_penalty": bias_penalty,
        "certified_upper": certified_upper,
        "current": current_lyapunov,
        "expected_next": expected_next,
        "slack": certified_upper-expected_next,
        "variance_penalty": variance_penalty,
        "weighted_stationarity": stationarity,
    }
