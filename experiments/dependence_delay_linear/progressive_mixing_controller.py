"""Exact progressive anytime controller tools for EXP-009D."""

from typing import Dict

import numpy as np

from markov_jump_ms import (
    covariance_operator_coefficients,
    homogeneous_delays,
    polynomial_matrix,
    regime_transition,
    registered_expanding_td_model,
    spectral_radius_with_residual,
    thinned_persistence,
)
from predictable_mixing_controller import (
    ADDITIVE_SCALE,
    SERVER_OVERHEAD,
)


INITIAL_PILOT = 128
BLOCK_BUDGET = 2_000
ANYTIME_ALPHA = 0.01


def initial_covariance_state(delay: int) -> np.ndarray:
    lifted = delay + 1
    matrix = np.ones((lifted, lifted), dtype=float)
    vector = matrix.reshape(-1, order="F")
    return np.concatenate((0.5 * vector, 0.5 * vector))


def advance_observations(
    state: np.ndarray,
    persistence: float,
    transitions: int,
    delay: int,
) -> np.ndarray:
    if transitions <= 0:
        return state.copy()
    used = thinned_persistence(persistence, transitions)
    transition = regime_transition(used)
    block = (delay + 1) ** 2
    operator = np.kron(transition.T, np.eye(block))
    return operator.dot(state)


def advance_action(
    state: np.ndarray,
    action: Dict[str, float],
    persistence: float,
    rho: float,
    delay: int,
    updates: int,
) -> Dict[str, object]:
    if updates <= 0 or action["eta"] <= 0.0:
        return {
            "state": state.copy(),
            "radius": 1.0,
            "residual": 0.0,
        }
    model = registered_expanding_td_model()
    num_agents = int(action["num_agents"])
    delays = homogeneous_delays(num_agents, delay)
    used = thinned_persistence(persistence, int(action["gap"]))
    coefficients = covariance_operator_coefficients(
        model, delays, rho, used
    )["markov"]
    eta = float(action["eta"])
    radius, residual = spectral_radius_with_residual(coefficients, eta)
    operator = polynomial_matrix(coefficients, eta)
    block = (delay + 1) ** 2
    variance = ADDITIVE_SCALE ** 2 * (
        rho + (1.0 - rho) / float(num_agents)
    )
    forcing = np.zeros(2 * block, dtype=float)
    forcing[0] = 0.5 * eta * eta * variance
    forcing[block] = 0.5 * eta * eta * variance
    augmented = np.zeros(
        (len(state) + 1, len(state) + 1), dtype=float
    )
    augmented[:-1, :-1] = operator
    augmented[:-1, -1] = forcing
    augmented[-1, -1] = 1.0
    combined = np.concatenate((state, np.asarray((1.0,))))
    final = np.linalg.matrix_power(augmented, int(updates)).dot(combined)
    return {
        "state": final[:-1],
        "radius": radius,
        "residual": residual,
    }


def final_expected_error(state: np.ndarray, delay: int) -> float:
    block = (delay + 1) ** 2
    return float(state[0] + state[block])


def block_execution_counts(
    action: Dict[str, float], block_budget: int
) -> Dict[str, int]:
    if action["eta"] <= 0.0:
        return {
            "updates": 0,
            "observation_transitions": int(block_budget),
            "leftover_observations": int(block_budget),
        }
    cost = (
        int(action["gap"])
        + SERVER_OVERHEAD
        + int(action["num_agents"])
    )
    updates = int(block_budget) // cost
    leftover = int(block_budget) - updates * cost
    observations = updates * int(action["gap"]) + leftover
    return {
        "updates": updates,
        "observation_transitions": observations,
        "leftover_observations": leftover,
    }
