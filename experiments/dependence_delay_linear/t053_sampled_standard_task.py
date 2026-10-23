"""Sample exact standard-task TD innovations for the T-053 CPU pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from numba import njit
import numpy as np


@dataclass(frozen=True)
class TaskSamplingTable:
    outcome_cdf: np.ndarray
    outcome_count: np.ndarray
    next_state: np.ndarray
    terminal: np.ndarray
    gradient: np.ndarray
    initial_cdf: np.ndarray
    stationary_cdf: np.ndarray


def build_task_sampling_table(
    task: dict[str, Any], *, discount: float = 0.95
) -> TaskSamplingTable:
    """Reconstruct the exact policy-weighted Gymnasium outcome catalogue."""

    environment = gym.make(task["environment_id"], **task["kwargs"])
    try:
        transitions = environment.unwrapped.P
        policy = np.asarray(task["policy"], dtype=float)
        active = np.asarray(task["active_states"], dtype=int)
        lookup = {int(state): index for index, state in enumerate(active)}
        features = np.asarray(task["features"], dtype=float)
        theta = np.asarray(task["theta_star"], dtype=float)
        initial = np.asarray(task["initial"], dtype=float)
        stationary = np.asarray(task["stationary"], dtype=float)
        if not 0.0 <= discount < 1.0:
            raise ValueError("discount must lie in [0,1)")
        catalogues: list[list[tuple[float, int, bool, np.ndarray]]] = []
        for active_index, state in enumerate(active):
            phi = features[active_index]
            rows: list[tuple[float, int, bool, np.ndarray]] = []
            for action in range(policy.shape[1]):
                for outcome in transitions[int(state)][action]:
                    probability, successor, reward, done, *_ = tuple(outcome)
                    weight = float(policy[int(state), action]) * float(probability)
                    terminal = bool(done)
                    if terminal:
                        next_active = -1
                        gradient = phi * (float(reward) - float(phi @ theta))
                    else:
                        next_active = lookup[int(successor)]
                        gradient = phi * (
                            float(reward)
                            + discount * float(features[next_active] @ theta)
                            - float(phi @ theta)
                        )
                    rows.append((weight, next_active, terminal, gradient))
            if not np.isclose(sum(row[0] for row in rows), 1.0):
                raise RuntimeError("policy-weighted outcome probabilities do not sum to one")
            catalogues.append(rows)
        maximum = max(len(rows) for rows in catalogues)
        states = active.size
        dimension = features.shape[1]
        cdf = np.ones((states, maximum))
        count = np.zeros(states, dtype=np.int64)
        next_state = np.full((states, maximum), -1, dtype=np.int64)
        terminal = np.ones((states, maximum), dtype=np.bool_)
        gradient = np.zeros((states, maximum, dimension))
        for state, rows in enumerate(catalogues):
            count[state] = len(rows)
            cumulative = 0.0
            for index, (probability, successor, done, innovation) in enumerate(rows):
                cumulative += probability
                cdf[state, index] = cumulative
                next_state[state, index] = successor
                terminal[state, index] = done
                gradient[state, index] = innovation
            cdf[state, len(rows) - 1] = 1.0
        return TaskSamplingTable(
            outcome_cdf=cdf,
            outcome_count=count,
            next_state=next_state,
            terminal=terminal,
            gradient=gradient,
            initial_cdf=np.cumsum(initial),
            stationary_cdf=np.cumsum(stationary),
        )
    finally:
        environment.close()


@njit(cache=True)
def _draw_cdf(cdf: np.ndarray, count: int, uniform: float) -> int:
    for index in range(count):
        if uniform <= cdf[index]:
            return index
    return count - 1


@njit(cache=True)
def _sample_gradient_paths_jit(
    outcome_cdf: np.ndarray,
    outcome_count: np.ndarray,
    next_state: np.ndarray,
    terminal: np.ndarray,
    gradient: np.ndarray,
    initial_cdf: np.ndarray,
    stationary_cdf: np.ndarray,
    paths: int,
    horizon: int,
    seed: int,
) -> np.ndarray:
    np.random.seed(seed)
    dimension = gradient.shape[2]
    innovations = np.empty((paths, horizon, dimension))
    states = np.empty(paths, dtype=np.int64)
    for path in range(paths):
        states[path] = _draw_cdf(
            stationary_cdf, stationary_cdf.size, np.random.random()
        )
    for time in range(horizon):
        for path in range(paths):
            state = states[path]
            outcome = _draw_cdf(
                outcome_cdf[state], outcome_count[state], np.random.random()
            )
            innovations[path, time] = gradient[state, outcome]
            if terminal[state, outcome]:
                states[path] = _draw_cdf(
                    initial_cdf, initial_cdf.size, np.random.random()
                )
            else:
                states[path] = next_state[state, outcome]
    return innovations


def sample_gradient_paths(
    table: TaskSamplingTable, *, paths: int, horizon: int, seed: int
) -> np.ndarray:
    """Sample independent stationary trajectories of exact TD innovations."""

    if paths < 1 or horizon < 1 or seed < 0:
        raise ValueError("invalid paths, horizon, or seed")
    return _sample_gradient_paths_jit(
        table.outcome_cdf,
        table.outcome_count,
        table.next_state,
        table.terminal,
        table.gradient,
        table.initial_cdf,
        table.stationary_cdf,
        paths,
        horizon,
        int(seed % (2**31 - 1)),
    )


def verify_sampling_table(
    task: dict[str, Any], table: TaskSamplingTable
) -> dict[str, float]:
    """Algebraically verify kernel, innovation mean, and second moment."""

    stationary = np.asarray(task["stationary"], dtype=float)
    initial = np.asarray(task["initial"], dtype=float)
    states = stationary.size
    dimension = table.gradient.shape[2]
    transition = np.zeros((states, states))
    mean = np.zeros(dimension)
    second = np.zeros((dimension, dimension))
    for state in range(states):
        previous = 0.0
        for outcome in range(table.outcome_count[state]):
            probability = table.outcome_cdf[state, outcome] - previous
            previous = table.outcome_cdf[state, outcome]
            innovation = table.gradient[state, outcome]
            mean += stationary[state] * probability * innovation
            second += (
                stationary[state]
                * probability
                * np.outer(innovation, innovation)
            )
            if table.terminal[state, outcome]:
                transition[state] += probability * initial
            else:
                transition[state, table.next_state[state, outcome]] += probability
    return {
        "transition_max_error": float(
            np.max(np.abs(transition - task["continuing_transition"]))
        ),
        "mean_norm": float(np.linalg.norm(mean)),
        "second_moment_max_error": float(
            np.max(np.abs(second - task["gradient_second_moment"]))
        ),
    }


def sample_fingerprint_matches(
    *,
    transition: np.ndarray,
    stationary: np.ndarray,
    transitions: int,
    blocks: int,
    rho: float,
    seed: int,
) -> np.ndarray:
    """Sample observable two-agent state-path matches from independent blocks."""

    probability = np.asarray(transition, dtype=float)
    invariant = np.asarray(stationary, dtype=float)
    if transitions < 0 or blocks < 1 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid fingerprint design")
    rng = np.random.default_rng(seed)
    matches = np.zeros(blocks, dtype=np.int8)
    states = probability.shape[0]
    for block in range(blocks):
        paths = np.empty((3, transitions + 1), dtype=np.int64)
        paths[:, 0] = rng.choice(states, size=3, p=invariant)
        for time in range(transitions):
            for path in range(3):
                paths[path, time + 1] = rng.choice(
                    states, p=probability[paths[path, time]]
                )
        switches = rng.random(2) < math_sqrt(rho)
        first = paths[0] if switches[0] else paths[1]
        second = paths[0] if switches[1] else paths[2]
        matches[block] = int(np.array_equal(first, second))
    return matches


def math_sqrt(value: float) -> float:
    """Small wrapper kept outside the sampling loop for testability."""

    return float(np.sqrt(value))


def prefix_aggregate_innovations(
    path_bank: np.ndarray,
    *,
    rho: float,
    candidates: tuple[int, ...],
    seed: int,
) -> dict[int, np.ndarray]:
    """Build common-random-number prefix averages for fixed-q policies."""

    bank = np.asarray(path_bank, dtype=float)
    actions = tuple(sorted({int(q) for q in candidates}))
    if bank.ndim != 3 or not actions or actions[0] < 1:
        raise ValueError("invalid path bank or candidates")
    if bank.shape[0] < max(actions) + 1 or not 0.0 <= rho <= 1.0:
        raise ValueError("path bank is too small or rho is invalid")
    rng = np.random.default_rng(seed)
    switches = rng.random(max(actions)) < np.sqrt(rho)
    cumulative = np.zeros_like(bank[0])
    aggregates: dict[int, np.ndarray] = {}
    for actor in range(max(actions)):
        cumulative += bank[0] if switches[actor] else bank[actor + 1]
        q = actor + 1
        if q in actions:
            aggregates[q] = cumulative.copy() / q
    return aggregates


@njit(cache=True)
def _delayed_pr_risk_jit(
    innovations: np.ndarray,
    drift: np.ndarray,
    step_size: float,
    delay: int,
    initial_error: np.ndarray,
) -> float:
    updates, dimension = innovations.shape
    errors = np.empty((updates + delay + 1, dimension))
    for index in range(delay + 1):
        errors[index] = initial_error
    for time in range(updates):
        current = errors[delay + time]
        delayed = errors[time]
        errors[delay + time + 1] = (
            current - step_size * (drift @ delayed) + step_size * innovations[time]
        )
    burn_in = updates // 2
    averaged = np.zeros(dimension)
    count = updates - burn_in
    for time in range(burn_in + 1, updates + 1):
        averaged += errors[delay + time]
    averaged /= count
    return float(averaged @ averaged)


def delayed_pr_risk(
    innovations: np.ndarray,
    *,
    drift: np.ndarray,
    step_size: float,
    delay: int,
    initial_error: np.ndarray,
) -> float:
    """Sampled squared norm of the half-tail delayed PR average."""

    values = np.asarray(innovations, dtype=float)
    matrix = np.asarray(drift, dtype=float)
    initial = np.asarray(initial_error, dtype=float)
    if values.ndim != 2 or matrix.shape != (values.shape[1], values.shape[1]):
        raise ValueError("innovation or drift shape mismatch")
    if initial.shape != (values.shape[1],) or step_size <= 0.0 or delay < 0:
        raise ValueError("invalid initial error, step size, or delay")
    return _delayed_pr_risk_jit(values, matrix, step_size, delay, initial)
