"""Exact finite-horizon moments for affine finite-state Markov jump systems."""

from __future__ import annotations

import numpy as np


def validate_markov_transition(transition: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transition, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition must be square")
    if np.any(matrix < 0.0) or not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("transition must be row stochastic")
    return matrix


def propagate_affine_markov_moments(
    *,
    transition: np.ndarray,
    augmented_updates: np.ndarray,
    initial_mode_probability: np.ndarray,
    initial_state: np.ndarray,
    steps: int,
) -> dict[str, np.ndarray]:
    """Propagate unnormalized mode-conditioned first and second moments.

    ``augmented_updates[a,b]`` maps ``[x_t;1]`` to ``[x_(t+1);1]``
    conditional on the exogenous transition ``Z_t=a, Z_(t+1)=b``.
    """

    probability = validate_markov_transition(transition)
    updates = np.asarray(augmented_updates, dtype=float)
    initial_probability = np.asarray(initial_mode_probability, dtype=float)
    state = np.asarray(initial_state, dtype=float)
    modes = probability.shape[0]
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if initial_probability.shape != (modes,) or np.any(initial_probability < 0.0):
        raise ValueError("initial mode probabilities must match modes")
    if not np.isclose(initial_probability.sum(), 1.0):
        raise ValueError("initial mode probabilities must sum to one")
    if updates.ndim != 4 or updates.shape[:2] != (modes, modes):
        raise ValueError("updates must have one matrix for every mode transition")
    augmented_dimension = updates.shape[2]
    if updates.shape[3] != augmented_dimension or state.shape != (
        augmented_dimension - 1,
    ):
        raise ValueError("update and state dimensions are incompatible")
    augmented_state = np.append(state, 1.0)
    first = initial_probability[:, None] * augmented_state[None, :]
    second = (
        initial_probability[:, None, None]
        * augmented_state[None, :, None]
        * augmented_state[None, None, :]
    )
    for _ in range(steps):
        next_first = np.zeros_like(first)
        next_second = np.zeros_like(second)
        for source in range(modes):
            for target in range(modes):
                weight = probability[source, target]
                update = updates[source, target]
                next_first[target] += weight * update @ first[source]
                next_second[target] += (
                    weight * update @ second[source] @ update.T
                )
        first, second = next_first, next_second
    unconditional_first = first.sum(axis=0)
    unconditional_second = second.sum(axis=0)
    return {
        "mode_first": first,
        "mode_second": second,
        "mean": unconditional_first[:-1],
        "second_moment": unconditional_second[:-1, :-1],
        "covariance": unconditional_second[:-1, :-1]
        - np.outer(unconditional_first[:-1], unconditional_first[:-1]),
        "mode_probability": first[:, -1],
    }


def delayed_scalar_mode_updates(
    *,
    innovations: np.ndarray,
    mu: float,
    step_size: float,
    delay: int,
) -> np.ndarray:
    innovation = np.asarray(innovations, dtype=float)
    if innovation.ndim != 1 or mu <= 0.0 or step_size <= 0.0 or delay < 0:
        raise ValueError("invalid innovations, drift, step size, or delay")
    modes = innovation.size
    dimension = delay + 2
    updates = np.zeros((modes, modes, dimension, dimension))
    for source in range(modes):
        update = np.zeros((dimension, dimension))
        update[0, 0] = 1.0
        update[0, delay] -= step_size * mu
        update[0, -1] = step_size * innovation[source]
        if delay:
            update[1 : delay + 1, :delay] = np.eye(delay)
        update[-1, -1] = 1.0
        updates[source, :, :, :] = update
    return updates


def symmetric_ar_sign_chain(markov_lambda: float) -> np.ndarray:
    if not 0.0 <= markov_lambda < 1.0:
        raise ValueError("markov_lambda must lie in [0,1)")
    stay = (1.0 + markov_lambda) / 2.0
    switch = 1.0 - stay
    return np.array([[stay, switch], [switch, stay]], dtype=float)
