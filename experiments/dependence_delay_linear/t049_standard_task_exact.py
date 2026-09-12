"""Exact vector TD objects from public finite-state Gymnasium kernels.

The construction keeps the complete lag-covariance matrices of the TD
innovation. It does not replace a standard task by a scalar drift/noise/SLEM
proxy. The fixed evaluation policy is generated deterministically by exact
value iteration followed by a public epsilon-soft mixture.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import gymnasium as gym
import numpy as np

from experiments.dependence_delay_linear.t042_poisson_td_remainder import (
    stationary_distribution,
)
from experiments.dependence_delay_linear.t043a_standard_task_static import (
    _fourier_features,
    _reachable_states,
)


def _array_hash(arrays: list[np.ndarray]) -> str:
    digest = sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def exact_epsilon_soft_policy(
    transition_dictionary: dict[int, dict[int, list[tuple[Any, ...]]]],
    *,
    states: int,
    actions: int,
    discount: float,
    epsilon: float,
    tolerance: float = 1e-13,
    maximum_iterations: int = 100_000,
) -> np.ndarray:
    """Return a deterministic-tie value-iteration policy with soft support."""

    if not 0.0 <= discount < 1.0 or not 0.0 < epsilon < 1.0:
        raise ValueError("discount and epsilon must lie in their open ranges")
    values = np.zeros(states)
    q_values = np.zeros((states, actions))
    for _ in range(maximum_iterations):
        for state in range(states):
            for action in range(actions):
                q_values[state, action] = sum(
                    float(probability)
                    * (
                        float(reward)
                        + discount
                        * (0.0 if bool(terminated) else values[int(next_state)])
                    )
                    for probability, next_state, reward, terminated, *_ in (
                        tuple(outcome)
                        for outcome in transition_dictionary[state][action]
                    )
                )
        updated = np.max(q_values, axis=1)
        if np.max(np.abs(updated - values)) <= tolerance:
            values = updated
            break
        values = updated
    else:
        raise RuntimeError("value iteration did not converge")

    for state in range(states):
        for action in range(actions):
            q_values[state, action] = sum(
                float(probability)
                * (
                    float(reward)
                    + discount
                    * (0.0 if bool(terminated) else values[int(next_state)])
                )
                for probability, next_state, reward, terminated, *_ in (
                    tuple(outcome)
                    for outcome in transition_dictionary[state][action]
                )
            )
    greedy = np.argmax(q_values, axis=1)
    policy = np.full((states, actions), epsilon / actions)
    policy[np.arange(states), greedy] += 1.0 - epsilon
    return policy


def build_exact_projected_task(
    *,
    environment_id: str,
    kwargs: dict[str, Any],
    feature_dimension: int,
    discount: float,
    policy_epsilon: float,
) -> dict[str, Any]:
    """Build exact drift and edge-gradient moments for a fixed-policy MRP."""

    environment = gym.make(environment_id, **kwargs)
    try:
        unwrapped = environment.unwrapped
        transition_dictionary = unwrapped.P
        total_states = int(environment.observation_space.n)
        actions = int(environment.action_space.n)
        initial = np.asarray(unwrapped.initial_state_distrib, dtype=float)
        initial /= initial.sum()
        policy = exact_epsilon_soft_policy(
            transition_dictionary,
            states=total_states,
            actions=actions,
            discount=discount,
            epsilon=policy_epsilon,
        )

        continuing = np.zeros((total_states, total_states))
        bootstrap = np.zeros((total_states, total_states))
        mean_reward = np.zeros(total_states)
        raw_outcomes: list[list[tuple[float, int, float, bool]]] = [
            [] for _ in range(total_states)
        ]
        for state in range(total_states):
            for action in range(actions):
                for outcome in transition_dictionary[state][action]:
                    probability, next_state, reward, terminated, *_ = tuple(outcome)
                    weight = policy[state, action] * float(probability)
                    next_int = int(next_state)
                    done = bool(terminated)
                    raw_outcomes[state].append(
                        (weight, next_int, float(reward), done)
                    )
                    mean_reward[state] += weight * float(reward)
                    if done:
                        continuing[state] += weight * initial
                    else:
                        continuing[state, next_int] += weight
                        bootstrap[state, next_int] += weight
        if not np.allclose(continuing.sum(axis=1), 1.0, atol=1e-12):
            raise RuntimeError("regenerative transition is not row stochastic")

        active = _reachable_states(continuing, initial)
        continuing_active = continuing[np.ix_(active, active)]
        bootstrap_active = bootstrap[np.ix_(active, active)]
        initial_active = initial[active]
        initial_active /= initial_active.sum()
        stationary = stationary_distribution(continuing_active)
        dimension = min(int(feature_dimension), active.size)
        features = _fourier_features(
            active, total_states, stationary, dimension
        )
        reward_active = mean_reward[active]
        drift = features.T @ (
            stationary[:, None]
            * (features - discount * bootstrap_active @ features)
        )
        reward_vector = features.T @ (stationary * reward_active)
        theta_star = np.linalg.solve(drift, reward_vector)
        symmetric_drift = (drift + drift.T) / 2.0
        drift_minimum = float(np.min(np.linalg.eigvalsh(symmetric_drift)))
        drift_norm = float(np.linalg.norm(drift, ord=2))
        if drift_minimum <= 0.0 or drift_norm <= 0.0:
            raise RuntimeError("projected TD drift is not strongly monotone")

        active_lookup = {int(state): index for index, state in enumerate(active)}
        state_count = active.size
        edge_gradient_sum = np.zeros((state_count, state_count, dimension))
        second_moment = np.zeros((dimension, dimension))
        mean_gradient = np.zeros(dimension)
        for active_index, state in enumerate(active):
            phi = features[active_index]
            for weight, next_state, reward, terminated in raw_outcomes[int(state)]:
                if terminated:
                    gradient = phi * (reward - float(phi @ theta_star))
                    successor_weights = initial_active
                else:
                    successor_index = active_lookup[next_state]
                    gradient = phi * (
                        reward
                        + discount * float(features[successor_index] @ theta_star)
                        - float(phi @ theta_star)
                    )
                    successor_weights = np.zeros(state_count)
                    successor_weights[successor_index] = 1.0
                stationary_weight = stationary[active_index] * weight
                mean_gradient += stationary_weight * gradient
                second_moment += stationary_weight * np.outer(gradient, gradient)
                edge_gradient_sum[active_index] += (
                    weight * successor_weights[:, None] * gradient
                )
        if not np.allclose(mean_gradient, 0.0, atol=1e-8):
            raise RuntimeError("projected TD fixed point has nonzero mean gradient")
        conditional_gradient = np.sum(edge_gradient_sum, axis=1)

        eigenvalues = np.linalg.eigvals(continuing_active)
        stationary_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        remaining = np.delete(np.abs(eigenvalues), stationary_index)
        mixing_slem = float(np.max(remaining, initial=0.0))
        if not 0.0 <= mixing_slem < 1.0 - 1e-10:
            raise RuntimeError("regenerative task lacks a strict mixing gap")

        return {
            "environment_id": environment_id,
            "kwargs": kwargs,
            "gymnasium_version": gym.__version__,
            "total_states": total_states,
            "active_states": active,
            "actions": actions,
            "initial": initial_active,
            "policy": policy,
            "policy_epsilon": policy_epsilon,
            "continuing_transition": continuing_active,
            "bootstrap_transition": bootstrap_active,
            "stationary": stationary,
            "features": features,
            "drift": drift,
            "theta_star": theta_star,
            "edge_gradient_sum": edge_gradient_sum,
            "conditional_gradient": conditional_gradient,
            "gradient_second_moment": second_moment,
            "drift_minimum": drift_minimum,
            "drift_norm": drift_norm,
            "mixing_slem": mixing_slem,
            "kernel_sha256": _array_hash(
                [
                    continuing_active,
                    bootstrap_active,
                    stationary,
                    policy,
                    features,
                    drift,
                    edge_gradient_sum,
                    second_moment,
                ]
            ),
        }
    finally:
        environment.close()


def exact_gradient_lag_covariances(
    task: dict[str, Any], *, horizon: int
) -> np.ndarray:
    """Return the exact finite lag-covariance array."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    transition = np.asarray(task["continuing_transition"], dtype=float)
    stationary = np.asarray(task["stationary"], dtype=float)
    edge = np.asarray(task["edge_gradient_sum"], dtype=float)
    conditional = np.asarray(task["conditional_gradient"], dtype=float)
    second = np.asarray(task["gradient_second_moment"], dtype=float)
    dimension = second.shape[0]
    positive = [second]
    future_gradient = conditional.copy()
    for _lag in range(1, horizon):
        past_future = np.einsum(
            "s,sud,ue->de",
            stationary,
            edge,
            future_gradient,
            optimize=True,
        )
        positive.append(past_future.T)
        future_gradient = transition @ future_gradient
    lags = np.zeros((2 * horizon - 1, dimension, dimension))
    center = horizon - 1
    lags[center] = positive[0]
    for lag in range(1, horizon):
        lags[center + lag] = positive[lag]
        lags[center - lag] = positive[lag].T
    return lags


def trajectory_switch_overlap_factor(
    q_left: int, q_right: int, rho: float
) -> float:
    """Return the multiplier for marginal-preserving trajectory switches.

    Each actor uses a common trajectory with probability sqrt(rho) and an
    independent private trajectory otherwise. The switch is fixed over the
    evaluated learning trajectory and independently redrawn across replicates.
    """

    if q_left < 1 or q_right < 1 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid participation count or rho")
    overlap = min(q_left, q_right) / float(q_left * q_right)
    return float(rho + (1.0 - rho) * overlap)


def public_exact_task_summary(task: dict[str, Any]) -> dict[str, Any]:
    """Return stable, JSON-compatible public task constants."""

    return {
        "environment_id": task["environment_id"],
        "kwargs": task["kwargs"],
        "gymnasium_version": task["gymnasium_version"],
        "total_states": task["total_states"],
        "active_state_count": int(task["active_states"].size),
        "actions": task["actions"],
        "feature_dimension": int(task["features"].shape[1]),
        "policy": "exact_value_iteration_epsilon_soft",
        "policy_epsilon": task["policy_epsilon"],
        "drift_minimum": task["drift_minimum"],
        "drift_norm": task["drift_norm"],
        "theta_star_norm": float(np.linalg.norm(task["theta_star"])),
        "gradient_noise_trace": float(np.trace(task["gradient_second_moment"])),
        "mixing_slem": task["mixing_slem"],
        "kernel_sha256": task["kernel_sha256"],
    }


def stable_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
