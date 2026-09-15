"""Outcome-free scalar surrogate and safety shield proposed by T-020."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SERVER_OVERHEAD_BYTES = 65_536
FLOAT_BYTES = 4


@dataclass(frozen=True)
class ObservableState:
    signal_sq_lcb: float
    noise_variance_ucb: float
    rho_upper: float
    mixing_time_upper: float
    delay_bias_ucb: float
    smoothness_upper: float
    learning_rate: float
    message_price: float
    environment_price: float


def variance_factor(q: np.ndarray, rho_upper: float) -> np.ndarray:
    q_array = np.asarray(q, dtype=float)
    return rho_upper + (1.0 - rho_upper) / q_array


def continuous_message_optimum(
    rho: float, server_overhead: float, per_agent_payload: float
) -> float:
    """Interior optimum of variance_factor times affine message cost."""

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0,1]")
    if server_overhead <= 0.0 or per_agent_payload <= 0.0:
        raise ValueError("costs must be positive")
    if rho == 0.0:
        return float("inf")
    return float(
        np.sqrt((1.0 - rho) * server_overhead / (rho * per_agent_payload))
    )


def usable_horizon(
    message_budget: int,
    environment_budget: int,
    q: int,
    b: int,
    parameters: int,
    delay_p90: float,
) -> int:
    """Static cost/delay horizon; depends only on public design quantities."""

    message_cost = SERVER_OVERHEAD_BYTES + q * parameters * FLOAT_BYTES
    raw = min(message_budget // message_cost, environment_budget // b)
    delay_loss = int(np.ceil(delay_p90 / b))
    return max(0, int(raw) - delay_loss)


def learning_value_lcb(
    q: np.ndarray,
    b: np.ndarray,
    message_cost: np.ndarray,
    state: ObservableState,
    delay_p90: float,
    confidence_radius: np.ndarray,
) -> np.ndarray:
    """Curvature-free scalar lower bound on one-block descent minus cost."""

    q_array = np.asarray(q, dtype=float)
    b_array = np.asarray(b, dtype=float)
    eta = state.learning_rate
    second_moment_ucb = state.signal_sq_lcb + (
        state.mixing_time_upper
        * state.noise_variance_ucb
        * variance_factor(q_array, state.rho_upper)
    )
    descent = eta * state.signal_sq_lcb
    smoothness_penalty = 0.5 * state.smoothness_upper * eta**2 * second_moment_ucb
    delay_penalty = eta * state.delay_bias_ucb * delay_p90 / b_array
    resource_penalty = (
        state.message_price * np.asarray(message_cost, dtype=float)
        + state.environment_price * b_array
    )
    return (
        descent
        - smoothness_penalty
        - delay_penalty
        - resource_penalty
        - np.asarray(confidence_radius, dtype=float)
    )


def shielded_choice(
    value_lcb: np.ndarray,
    fallback_index: int,
    safety_wealth: float,
) -> tuple[int, float]:
    """Permit adaptation only when cumulative certified surplus stays nonnegative."""

    values = np.asarray(value_lcb, dtype=float)
    candidate = int(np.argmax(values))
    delta = float(values[candidate] - values[fallback_index])
    if candidate == fallback_index or safety_wealth + delta < 0.0:
        return fallback_index, safety_wealth
    return candidate, safety_wealth + delta
