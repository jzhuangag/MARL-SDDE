"""Kalman-innovation identities for adaptive common-factor experiments.

The action at observation t is (q_t, b_t).  The first gap is immaterial
because the latent state is stationary; for t > 0, b_t is the elapsed
environment time since the preceding observation.  Eta may be carried by a
controller, but it does not enter this observation likelihood.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AdaptiveAction:
    q: int
    b: int
    eta: float = 0.0

    def __post_init__(self) -> None:
        if self.q < 1 or self.b < 1:
            raise ValueError("q and b must be positive")


@dataclass
class GaussianFilter:
    theta: float
    mixing: float
    mean: float = 0.0
    variance: float | None = None

    def __post_init__(self) -> None:
        if self.theta < 0.0:
            raise ValueError("theta must be nonnegative")
        if not 0.0 <= self.mixing <= 1.0:
            raise ValueError("mixing must be in [0,1]")
        if self.variance is None:
            self.variance = self.theta

    def propagate(self, gap: int) -> None:
        coefficient = self.mixing**gap
        self.mean = coefficient * self.mean
        self.variance = (
            coefficient * coefficient * float(self.variance)
            + (1.0 - coefficient * coefficient) * self.theta
        )

    def innovation(self, observation: float, q: int) -> tuple[float, float]:
        root_q = math.sqrt(q)
        residual = observation - root_q * self.mean
        innovation_variance = 1.0 + q * float(self.variance)
        return residual, innovation_variance

    def update(self, observation: float, q: int) -> tuple[float, float]:
        residual, innovation_variance = self.innovation(observation, q)
        root_q = math.sqrt(q)
        gain = float(self.variance) * root_q / innovation_variance
        self.mean += gain * residual
        self.variance = max(
            0.0, float(self.variance) - gain * root_q * float(self.variance)
        )
        return residual, innovation_variance


def scalar_normal_log_density(residual: float, variance: float) -> float:
    return -0.5 * (
        math.log(2.0 * math.pi * variance) + residual * residual / variance
    )


def conditional_gaussian_kl(
    source_mean: float,
    source_variance: float,
    target_mean: float,
    target_variance: float,
    q: int,
) -> float:
    """KL between the next common-direction innovations."""

    if q < 1:
        raise ValueError("q must be positive")
    source_innovation = 1.0 + q * source_variance
    target_innovation = 1.0 + q * target_variance
    mean_difference = math.sqrt(q) * (source_mean - target_mean)
    return 0.5 * (
        math.log(target_innovation / source_innovation)
        + (source_innovation + mean_difference * mean_difference)
        / target_innovation
        - 1.0
    )


def adaptive_log_likelihood(
    observations: Sequence[float],
    actions: Sequence[AdaptiveAction],
    theta: float,
    mixing: float,
) -> float:
    if len(observations) != len(actions):
        raise ValueError("one action is required for every observation")
    state = GaussianFilter(theta, mixing)
    value = 0.0
    for index, (observation, action) in enumerate(zip(observations, actions)):
        if index:
            state.propagate(action.b)
        residual, variance = state.update(float(observation), action.q)
        value += scalar_normal_log_density(residual, variance)
    return float(value)


def adaptive_log_likelihood_ratio(
    observations: Sequence[float],
    actions: Sequence[AdaptiveAction],
    theta_numerator: float,
    theta_denominator: float,
    mixing: float,
) -> float:
    return adaptive_log_likelihood(
        observations, actions, theta_numerator, mixing
    ) - adaptive_log_likelihood(
        observations, actions, theta_denominator, mixing
    )


def innovation_information_trace(
    observations: Sequence[float],
    actions: Sequence[AdaptiveAction],
    theta_source: float,
    theta_target: float,
    mixing: float,
) -> list[float]:
    """Realized predictable conditional-KL terms along one history."""

    if len(observations) != len(actions):
        raise ValueError("one action is required for every observation")
    source = GaussianFilter(theta_source, mixing)
    target = GaussianFilter(theta_target, mixing)
    values: list[float] = []
    for index, (observation, action) in enumerate(zip(observations, actions)):
        if index:
            source.propagate(action.b)
            target.propagate(action.b)
        values.append(
            conditional_gaussian_kl(
                source.mean,
                float(source.variance),
                target.mean,
                float(target.variance),
                action.q,
            )
        )
        source.update(float(observation), action.q)
        target.update(float(observation), action.q)
    return values


def observation_times(actions: Sequence[AdaptiveAction]) -> np.ndarray:
    times = np.zeros(len(actions), dtype=int)
    for index in range(1, len(actions)):
        times[index] = times[index - 1] + actions[index].b
    return times


def dense_common_covariance(
    actions: Sequence[AdaptiveAction], theta: float, mixing: float
) -> np.ndarray:
    times = observation_times(actions)
    q_roots = np.sqrt(np.asarray([action.q for action in actions], dtype=float))
    distances = np.abs(times[:, None] - times[None, :])
    return np.eye(len(actions)) + theta * (
        q_roots[:, None] * q_roots[None, :] * mixing**distances
    )


def dense_log_likelihood(
    observations: Sequence[float],
    actions: Sequence[AdaptiveAction],
    theta: float,
    mixing: float,
) -> float:
    vector = np.asarray(observations, dtype=float)
    covariance = dense_common_covariance(actions, theta, mixing)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")
    return float(
        -0.5
        * (
            len(vector) * math.log(2.0 * math.pi)
            + logdet
            + vector @ np.linalg.solve(covariance, vector)
        )
    )


def simulate_adaptive_path(
    rng: np.random.RandomState,
    theta: float,
    mixing: float,
    action_rule,
    observations: int,
) -> tuple[np.ndarray, list[AdaptiveAction]]:
    """Simulate a predictable action rule for numerical audits."""

    latent = rng.normal(scale=math.sqrt(theta))
    values: list[float] = []
    actions: list[AdaptiveAction] = []
    for index in range(observations):
        action = action_rule(tuple(values), tuple(actions))
        if index:
            coefficient = mixing**action.b
            latent = coefficient * latent + rng.normal(
                scale=math.sqrt(theta * (1.0 - coefficient * coefficient))
            )
        values.append(
            math.sqrt(action.q) * latent + float(rng.normal())
        )
        actions.append(action)
    return np.asarray(values), actions


def dual_budget_feasible(
    actions: Iterable[AdaptiveAction],
    overhead: int,
    message_budget: int,
    environment_budget: int,
    delay: int,
) -> bool:
    chosen = list(actions)
    messages = sum(overhead + action.q for action in chosen)
    environment = sum(action.b for action in chosen) + (delay if chosen else 0)
    return messages <= message_budget and environment <= environment_budget


def usable_commit_updates(scheduled_updates: int, delay: int) -> int:
    return max(0, scheduled_updates - delay)


def theorem_derived_fallback(
    probe_opportunity_cost_upper: float,
    wrong_commit_penalty_upper: float,
    delay_penalty_upper: float,
    guaranteed_commit_gain_lower: float,
) -> tuple[bool, float]:
    """Return (explore, certified low-instance safety slack)."""

    if min(
        probe_opportunity_cost_upper,
        wrong_commit_penalty_upper,
        delay_penalty_upper,
        guaranteed_commit_gain_lower,
    ) < 0.0:
        raise ValueError("certificate inputs must be nonnegative")
    total_cost = (
        probe_opportunity_cost_upper
        + wrong_commit_penalty_upper
        + delay_penalty_upper
    )
    return total_cost < guaranteed_commit_gain_lower, total_cost
