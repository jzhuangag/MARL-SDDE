"""Exact finite-horizon tabular actor--critic geometry.

The actor statistic uses a time-indexed value critic in a discounted TD
residual.  The critic is an on-policy Monte-Carlo value-regression operator.
These exact helpers instantiate constants for the theorem interface; they are
not a sampled efficacy benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .trajectory_interface import _joint_policy, _validate_game


@dataclass(frozen=True)
class TabularCriticGeometry:
    target: np.ndarray
    occupancy: np.ndarray
    diagonal: np.ndarray
    strong_convexity: float
    smoothness: float


def finite_horizon_value_and_occupancy(
    transition: np.ndarray,
    reward: np.ndarray,
    start_distribution: np.ndarray,
    logits: np.ndarray,
    discount: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    transition, reward, start, logits, profiles = _validate_game(
        transition, reward, start_distribution, logits, discount
    )
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    _, joint = _joint_policy(logits, profiles)
    states = transition.shape[0]
    policy_transition = np.einsum("sa,san->sn", joint, transition)
    policy_reward = np.sum(joint * reward, axis=1)
    values = np.zeros((horizon + 1, states), dtype=float)
    for time in range(horizon - 1, -1, -1):
        values[time] = policy_reward + discount * policy_transition @ values[time + 1]
    occupancy = np.zeros((horizon, states), dtype=float)
    occupancy[0] = start
    for time in range(1, horizon):
        occupancy[time] = occupancy[time - 1] @ policy_transition
    return values, occupancy


def tabular_critic_geometry(
    transition: np.ndarray,
    reward: np.ndarray,
    start_distribution: np.ndarray,
    logits: np.ndarray,
    discount: float,
    horizon: int,
) -> TabularCriticGeometry:
    values, occupancy = finite_horizon_value_and_occupancy(
        transition, reward, start_distribution, logits, discount, horizon
    )
    diagonal = occupancy / float(horizon)
    return TabularCriticGeometry(
        target=values[:-1].copy(),
        occupancy=occupancy.copy(),
        diagonal=diagonal.copy(),
        strong_convexity=float(np.min(diagonal)),
        smoothness=float(np.max(diagonal)),
    )


def tabular_critic_population_operator(
    critic: np.ndarray, geometry: TabularCriticGeometry
) -> np.ndarray:
    critic = np.asarray(critic, dtype=float)
    if critic.shape != geometry.target.shape or np.any(~np.isfinite(critic)):
        raise ValueError("critic must be finite and match the target shape")
    return geometry.diagonal * (critic - geometry.target)


def td_actor_population_mean(
    transition: np.ndarray,
    reward: np.ndarray,
    start_distribution: np.ndarray,
    logits: np.ndarray,
    discount: float,
    horizon: int,
    critic: np.ndarray,
) -> np.ndarray:
    """Exact mean of the discounted time-indexed TD actor statistic."""

    transition, reward, start, logits, profiles = _validate_game(
        transition, reward, start_distribution, logits, discount
    )
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    policies, joint = _joint_policy(logits, profiles)
    agents, states, _ = policies.shape
    critic = np.asarray(critic, dtype=float)
    if critic.shape != (horizon, states) or np.any(~np.isfinite(critic)):
        raise ValueError("critic must have shape (horizon, states)")
    terminal = np.vstack([critic, np.zeros((1, states), dtype=float)])
    _, occupancy = finite_horizon_value_and_occupancy(
        transition, reward, start, logits, discount, horizon
    )
    gradient = np.zeros_like(logits)
    for time in range(horizon):
        for state in range(states):
            for profile_index, profile in enumerate(profiles):
                for next_state in range(states):
                    mass = (
                        occupancy[time, state]
                        * joint[state, profile_index]
                        * transition[state, profile_index, next_state]
                    )
                    residual = (
                        reward[state, profile_index]
                        + discount * terminal[time + 1, next_state]
                        - terminal[time, state]
                    )
                    for agent in range(agents):
                        score = -policies[agent, state].copy()
                        score[profile[agent]] += 1.0
                        gradient[agent, state] += (
                            discount**time * mass * residual * score
                        )
    return gradient


def finite_horizon_return_bound(
    *, horizon: int, discount: float, reward_bound: float
) -> float:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < discount < 1.0:
        raise ValueError("discount must lie strictly between zero and one")
    if not math.isfinite(reward_bound) or reward_bound < 0.0:
        raise ValueError("reward_bound must be finite and nonnegative")
    return float(reward_bound * (1.0 - discount**horizon) / (1.0 - discount))


def td_actor_critic_sensitivity_bound(
    *, horizon: int, discount: float, score_norm_bound: float
) -> float:
    """Coefficient multiplying Euclidean value-critic error."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < discount < 1.0:
        raise ValueError("discount must lie strictly between zero and one")
    if not math.isfinite(score_norm_bound) or score_norm_bound < 0.0:
        raise ValueError("score_norm_bound must be finite and nonnegative")
    if horizon == 1:
        return 0.0
    return float(
        score_norm_bound
        * discount
        * (1.0 - discount ** (horizon - 1))
        / (1.0 - discount)
    )


def tabular_packet_coordinate_bounds(
    *,
    horizon: int,
    discount: float,
    reward_bound: float,
    critic_abs_bound: float,
) -> tuple[float, float]:
    """Per-coordinate actor and critic trajectory bounds.

    A categorical-softmax score coordinate has absolute value at most one.
    The critic stochastic gradient averages the H squared-loss terms.
    """

    if not math.isfinite(critic_abs_bound) or critic_abs_bound < 0.0:
        raise ValueError("critic_abs_bound must be finite and nonnegative")
    value_bound = finite_horizon_return_bound(
        horizon=horizon, discount=discount, reward_bound=reward_bound
    )
    residual_bound = reward_bound + (1.0 + discount) * critic_abs_bound
    actor = residual_bound * (1.0 - discount**horizon) / (1.0 - discount)
    critic = (critic_abs_bound + value_bound) / float(horizon)
    return float(actor), float(critic)


def tabular_value_target_lipschitz_bound(
    *,
    agents_changed: int,
    states: int,
    actions: int,
    horizon: int,
    discount: float,
    reward_bound: float,
) -> float:
    """Euclidean target sensitivity to a tabular-logit displacement."""

    if agents_changed <= 0 or states <= 0 or actions <= 1 or horizon <= 0:
        raise ValueError("dimensions and agents_changed must be positive")
    joint_tv = math.sqrt(agents_changed * actions) / 4.0
    squared = 0.0
    for time in range(horizon):
        remaining = horizon - time
        return_bound = finite_horizon_return_bound(
            horizon=remaining, discount=discount, reward_bound=reward_bound
        )
        coordinate = 2.0 * return_bound * remaining * joint_tv
        squared += states * coordinate**2
    return float(math.sqrt(squared))


def tabular_critic_operator_policy_lipschitz_bound(
    *,
    agents_changed: int,
    states: int,
    actions: int,
    horizon: int,
    discount: float,
    reward_bound: float,
    critic_abs_bound: float,
) -> float:
    """Policy Lipschitz envelope for the tabular regression operator."""

    if not math.isfinite(critic_abs_bound) or critic_abs_bound < 0.0:
        raise ValueError("critic_abs_bound must be finite and nonnegative")
    value_bound = finite_horizon_return_bound(
        horizon=horizon, discount=discount, reward_bound=reward_bound
    )
    joint_tv = math.sqrt(agents_changed * actions) / 4.0
    diagonal_lipschitz = (horizon - 1) * joint_tv / float(horizon)
    error_envelope = (critic_abs_bound + value_bound) * math.sqrt(horizon * states)
    target_lipschitz = tabular_value_target_lipschitz_bound(
        agents_changed=agents_changed,
        states=states,
        actions=actions,
        horizon=horizon,
        discount=discount,
        reward_bound=reward_bound,
    )
    return float(diagonal_lipschitz * error_envelope + target_lipschitz / horizon)


def teammate_actor_policy_lipschitz_bound(
    *,
    teammates_changed: int,
    teammate_actions: int,
    horizon: int,
    actor_trajectory_norm_bound: float,
) -> float:
    """Owner-gradient Lipschitz envelope for teammate-only motion."""

    if teammates_changed <= 0 or teammate_actions <= 1 or horizon <= 0:
        raise ValueError("dimensions and teammates_changed must be positive")
    statistic = float(actor_trajectory_norm_bound)
    if not math.isfinite(statistic) or statistic < 0.0:
        raise ValueError("actor_trajectory_norm_bound must be nonnegative")
    return float(
        statistic
        * horizon
        * math.sqrt(teammates_changed * teammate_actions)
        / 2.0
    )
