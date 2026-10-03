"""Exact finite-horizon risk for delayed affine stochastic approximation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from experiments.dependence_delay_linear.t065_discrete_joint_certificate import (
    delayed_companion_matrix,
)


@dataclass(frozen=True)
class ScheduledAction:
    participation: int
    gain: float
    updates: int


@dataclass(frozen=True)
class RiskState:
    mean: np.ndarray
    covariance: np.ndarray
    updates: int


def correlation_factor(participation: int, rho: float) -> float:
    if int(participation) != participation or participation < 1:
        raise ValueError("participation must be a positive integer")
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    return float(rho + (1.0 - rho) / participation)


def exact_learning_horizon(
    *,
    participation: int,
    message_overhead: int,
    message_budget: int,
    environment_budget: int,
    sensor_message_cost: int = 0,
    sensor_environment_cost: int = 0,
    reserved_delay_updates: int = 0,
) -> int:
    """Return the exact dual-budget horizon after fully charged sensing.

    One learning update consumes ``overhead + q`` messages and ``q`` actor
    transitions.  This explicitly fixes the historical missing division by q
    in an environment-horizon expression.
    """

    integer_values = (
        participation,
        message_overhead,
        message_budget,
        environment_budget,
        sensor_message_cost,
        sensor_environment_cost,
        reserved_delay_updates,
    )
    if any(int(value) != value or value < 0 for value in integer_values):
        raise ValueError("costs and budgets must be nonnegative integers")
    if participation < 1:
        raise ValueError("participation must be positive")
    remaining_messages = message_budget - sensor_message_cost
    remaining_environment = environment_budget - sensor_environment_cost
    if remaining_messages < 0 or remaining_environment < 0:
        return 0
    message_horizon = remaining_messages // (message_overhead + participation)
    environment_horizon = remaining_environment // participation
    return max(0, min(message_horizon, environment_horizon) - reserved_delay_updates)


def initial_risk_state(initial_error: np.ndarray, delay: int) -> RiskState:
    error = np.asarray(initial_error, dtype=float)
    if error.ndim != 1 or not np.all(np.isfinite(error)):
        raise ValueError("initial_error must be a finite vector")
    if int(delay) != delay or delay < 0:
        raise ValueError("delay must be a nonnegative integer")
    lifted = np.tile(error, delay + 1)
    return RiskState(
        mean=lifted,
        covariance=np.zeros((lifted.size, lifted.size), dtype=float),
        updates=0,
    )


def propagate_action(
    state: RiskState,
    *,
    drift: np.ndarray,
    base_noise_covariance: np.ndarray,
    delay: int,
    rho: float,
    action: ScheduledAction,
) -> RiskState:
    """Propagate exact first and second moments for one constant-action block."""

    drift_matrix = np.asarray(drift, dtype=float)
    noise = np.asarray(base_noise_covariance, dtype=float)
    dimension = drift_matrix.shape[0]
    if drift_matrix.shape != (dimension, dimension) or noise.shape != (
        dimension,
        dimension,
    ):
        raise ValueError("drift and noise covariance must be same-size square matrices")
    if int(action.updates) != action.updates or action.updates < 0:
        raise ValueError("updates must be a nonnegative integer")
    companion = delayed_companion_matrix(drift_matrix, action.gain, delay)
    if state.mean.shape != (companion.shape[0],) or state.covariance.shape != companion.shape:
        raise ValueError("risk state has incompatible lifted dimension")
    injection = np.zeros_like(companion)
    injection[:dimension, :dimension] = (
        action.gain**2 * correlation_factor(action.participation, rho) * noise
    )
    mean = state.mean.copy()
    covariance = state.covariance.copy()
    for _ in range(action.updates):
        mean = companion @ mean
        covariance = companion @ covariance @ companion.T + injection
        covariance = 0.5 * (covariance + covariance.T)
    return RiskState(
        mean=mean,
        covariance=covariance,
        updates=state.updates + action.updates,
    )


def propagate_schedule(
    *,
    drift: np.ndarray,
    base_noise_covariance: np.ndarray,
    initial_error: np.ndarray,
    delay: int,
    rho: float,
    schedule: Sequence[ScheduledAction],
) -> RiskState:
    state = initial_risk_state(initial_error, delay)
    for action in schedule:
        state = propagate_action(
            state,
            drift=drift,
            base_noise_covariance=base_noise_covariance,
            delay=delay,
            rho=rho,
            action=action,
        )
    return state


def terminal_mean_square_risk(state: RiskState, dimension: int) -> float:
    if int(dimension) != dimension or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    mean = state.mean[:dimension]
    covariance = state.covariance[:dimension, :dimension]
    return float(mean @ mean + np.trace(covariance))
