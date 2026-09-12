"""Exact task constants for the outcome-free T-043A feasibility scan."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import gymnasium as gym
import numpy as np

from experiments.dependence_delay_linear.t042_poisson_td_remainder import (
    stationary_distribution,
)


def _reachable_states(transition: np.ndarray, initial: np.ndarray) -> np.ndarray:
    reachable = set(np.flatnonzero(initial > 0.0).tolist())
    frontier = list(reachable)
    while frontier:
        source = frontier.pop()
        for target in np.flatnonzero(transition[source] > 0.0):
            target_int = int(target)
            if target_int not in reachable:
                reachable.add(target_int)
                frontier.append(target_int)
    return np.asarray(sorted(reachable), dtype=int)


def _fourier_features(
    original_states: np.ndarray,
    total_states: int,
    stationary: np.ndarray,
    dimension: int,
) -> np.ndarray:
    angle = 2.0 * np.pi * original_states.astype(float) / float(total_states)
    columns = [np.ones_like(angle)]
    frequency = 1
    while len(columns) < dimension:
        columns.append(np.cos(frequency * angle))
        if len(columns) < dimension:
            columns.append(np.sin(frequency * angle))
        frequency += 1
    raw = np.column_stack(columns[:dimension])
    weighted = np.sqrt(stationary)[:, None] * raw
    q_matrix, _ = np.linalg.qr(weighted)
    features = q_matrix[:, :dimension] / np.sqrt(stationary)[:, None]
    gram = features.T @ (stationary[:, None] * features)
    if not np.allclose(gram, np.eye(dimension), atol=1e-10):
        raise RuntimeError("weighted feature orthonormalization failed")
    return features


def _array_hash(arrays: list[np.ndarray]) -> str:
    digest = sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_projected_task(
    *,
    environment_id: str,
    kwargs: dict[str, Any],
    feature_dimension: int,
    discount: float,
) -> dict[str, Any]:
    """Build the exact fixed-policy regenerative MRP from an upstream task."""

    environment = gym.make(environment_id, **kwargs)
    try:
        unwrapped = environment.unwrapped
        transition_dictionary = unwrapped.P
        total_states = int(environment.observation_space.n)
        actions = int(environment.action_space.n)
        initial = np.asarray(unwrapped.initial_state_distrib, dtype=float)
        initial = initial / initial.sum()
        policy = np.full((total_states, actions), 1.0 / actions)
        continuing = np.zeros((total_states, total_states))
        bootstrap = np.zeros((total_states, total_states))
        mean_reward = np.zeros(total_states)
        outcomes: list[list[tuple[float, int, float, bool]]] = [
            [] for _ in range(total_states)
        ]
        for state in range(total_states):
            for action in range(actions):
                action_weight = policy[state, action]
                for probability, next_state, reward, terminated in transition_dictionary[
                    state
                ][action]:
                    weight = action_weight * float(probability)
                    next_int = int(next_state)
                    done = bool(terminated)
                    outcomes[state].append((weight, next_int, float(reward), done))
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
        initial_active = initial_active / initial_active.sum()
        stationary = stationary_distribution(continuing_active)
        dimension = min(int(feature_dimension), active.size)
        features = _fourier_features(active, total_states, stationary, dimension)
        reward_active = mean_reward[active]
        drift = features.T @ (
            stationary[:, None]
            * (features - discount * bootstrap_active @ features)
        )
        reward_vector = features.T @ (stationary * reward_active)
        theta_star = np.linalg.solve(drift, reward_vector)
        expected_gradient = np.zeros(dimension)
        noise_second = 0.0
        active_lookup = {int(state): index for index, state in enumerate(active)}
        for active_index, state in enumerate(active):
            phi = features[active_index]
            for weight, next_state, reward, terminated in outcomes[int(state)]:
                if terminated:
                    next_value = 0.0
                else:
                    next_value = float(features[active_lookup[next_state]] @ theta_star)
                delta = reward + discount * next_value - float(phi @ theta_star)
                gradient = phi * delta
                stationary_weight = stationary[active_index] * weight
                expected_gradient += stationary_weight * gradient
                noise_second += stationary_weight * float(gradient @ gradient)
        if not np.allclose(expected_gradient, 0.0, atol=1e-9):
            raise RuntimeError("projected TD fixed point has nonzero mean gradient")
        symmetric_drift = (drift + drift.T) / 2.0
        drift_minimum = float(np.min(np.linalg.eigvalsh(symmetric_drift)))
        drift_norm = float(np.linalg.norm(drift, ord=2))
        if drift_minimum <= 0.0 or drift_norm <= 0.0:
            raise RuntimeError("projected TD drift is not strongly monotone")
        eigenvalues = np.linalg.eigvals(continuing_active)
        distances = np.abs(eigenvalues - 1.0)
        stationary_index = int(np.argmin(distances))
        remaining = np.delete(np.abs(eigenvalues), stationary_index)
        mixing_slem = float(np.max(remaining, initial=0.0))
        if not 0.0 <= mixing_slem < 1.0 - 1e-10:
            raise RuntimeError("regenerative task lacks a strict finite-state mixing gap")
        return {
            "environment_id": environment_id,
            "kwargs": kwargs,
            "gymnasium_version": gym.__version__,
            "total_states": total_states,
            "active_states": active,
            "actions": actions,
            "initial": initial_active,
            "continuing_transition": continuing_active,
            "bootstrap_transition": bootstrap_active,
            "stationary": stationary,
            "features": features,
            "drift": drift,
            "theta_star": theta_star,
            "initial_error": float(np.linalg.norm(theta_star)),
            "single_agent_noise_second": float(noise_second),
            "drift_minimum": drift_minimum,
            "drift_norm": drift_norm,
            "mixing_slem": mixing_slem,
            "kernel_sha256": _array_hash(
                [continuing_active, bootstrap_active, stationary, features, drift]
            ),
        }
    finally:
        environment.close()


def public_task_summary(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment_id": task["environment_id"],
        "kwargs": task["kwargs"],
        "gymnasium_version": task["gymnasium_version"],
        "total_states": task["total_states"],
        "active_states": task["active_states"].tolist(),
        "actions": task["actions"],
        "feature_dimension": int(task["features"].shape[1]),
        "drift_minimum": task["drift_minimum"],
        "drift_norm": task["drift_norm"],
        "initial_error": task["initial_error"],
        "single_agent_noise_second": task["single_agent_noise_second"],
        "mixing_slem": task["mixing_slem"],
        "kernel_sha256": task["kernel_sha256"],
    }


def stable_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
