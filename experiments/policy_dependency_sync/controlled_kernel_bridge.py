"""Finite-horizon controlled-kernel perturbation utilities.

These routines audit the deterministic inequalities used to transfer the
tabular common-kernel argument to cache actions that change a Markov-game
behavior policy.  They do not estimate a neural critic confidence radius.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def categorical_total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """Return total variation between two categorical distributions."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("categorical distributions must be equal-length vectors")
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise ValueError("categorical distributions must be finite")
    if np.any(left_array < 0.0) or np.any(right_array < 0.0):
        raise ValueError("categorical probabilities must be nonnegative")
    if not np.isclose(np.sum(left_array), 1.0) or not np.isclose(
        np.sum(right_array), 1.0
    ):
        raise ValueError("categorical probabilities must sum to one")
    return float(0.5 * np.sum(np.abs(left_array - right_array)))


def induced_state_kernel(
    policy: np.ndarray, controlled_transition: np.ndarray
) -> np.ndarray:
    """Marginalize a finite controlled transition kernel over a policy.

    ``policy`` has shape ``(states, actions)`` and
    ``controlled_transition`` has shape ``(states, actions, next_states)``.
    """

    policy_array = np.asarray(policy, dtype=float)
    transition_array = np.asarray(controlled_transition, dtype=float)
    if policy_array.ndim != 2 or transition_array.ndim != 3:
        raise ValueError("invalid policy or transition rank")
    states, actions = policy_array.shape
    if transition_array.shape != (states, actions, states):
        raise ValueError("controlled transition must have shape (S, A, S)")
    if np.any(policy_array < 0.0) or not np.allclose(
        np.sum(policy_array, axis=1), 1.0
    ):
        raise ValueError("each policy row must be a probability distribution")
    if np.any(transition_array < 0.0) or not np.allclose(
        np.sum(transition_array, axis=2), 1.0
    ):
        raise ValueError("each controlled-transition row must sum to one")
    return np.einsum("sa,san->sn", policy_array, transition_array)


def maximum_row_total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """Return the largest row-wise TV distance between two kernels."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape or left_array.ndim != 2:
        raise ValueError("kernels must be equal-shape matrices")
    if np.any(left_array < 0.0) or np.any(right_array < 0.0):
        raise ValueError("kernel entries must be nonnegative")
    if not np.allclose(np.sum(left_array, axis=1), 1.0) or not np.allclose(
        np.sum(right_array, axis=1), 1.0
    ):
        raise ValueError("each kernel row must sum to one")
    return float(0.5 * np.max(np.sum(np.abs(left_array - right_array), axis=1)))


def trajectory_marginal_tv_bound(one_step_tv: float, horizon: int) -> float:
    """Coupling bound on state-law TV after ``horizon`` transitions."""

    if not np.isfinite(one_step_tv) or not 0.0 <= one_step_tv <= 1.0:
        raise ValueError("one_step_tv must lie in [0, 1]")
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    return float(1.0 - (1.0 - one_step_tv) ** horizon)


def finite_horizon_oscillation_charge(
    oscillations: Sequence[float],
    one_step_tv: float,
    *,
    score_uses_action: bool = False,
) -> float:
    """Bound the expectation shift of a finite-horizon score sum."""

    values = np.asarray(tuple(oscillations), dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("oscillations must be finite and nonnegative")
    offset = 1 if score_uses_action else 0
    return float(
        sum(
            oscillation
            * trajectory_marginal_tv_bound(one_step_tv, step + offset)
            for step, oscillation in enumerate(values)
        )
    )
