"""Exact vector finite-horizon risk for delayed linear SA with Markov noise.

The implementation mirrors the theorem in ``docs/t037_vector_markov_phase.md``.
It uses only lag covariances and a lifted delay companion matrix; it never
enumerates the state space of the process that generated the innovations.
"""

from __future__ import annotations

import math

import numpy as np


def delayed_vector_companion(
    drift: np.ndarray, step_size: float, delay: int
) -> np.ndarray:
    """Return the lifted matrix for e[t+1] = e[t] - eta A e[t-D]."""

    matrix = np.asarray(drift, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("drift must be square")
    if step_size <= 0.0 or delay < 0:
        raise ValueError("step size must be positive and delay nonnegative")
    dimension = matrix.shape[0]
    lifted = np.zeros((dimension * (delay + 1), dimension * (delay + 1)))
    lifted[:dimension, :dimension] = np.eye(dimension)
    delayed = slice(delay * dimension, (delay + 1) * dimension)
    lifted[:dimension, delayed] -= step_size * matrix
    if delay:
        lifted[dimension:, :-dimension] = np.eye(dimension * delay)
    return lifted


def dual_budget_updates(
    *,
    message_budget: float,
    environment_budget: float,
    message_cost: float,
    stride: int,
    delay: int,
) -> int:
    """Largest update count under message and environment-plus-delay costs."""

    if message_budget < 0.0 or environment_budget < 0.0:
        raise ValueError("budgets must be nonnegative")
    if message_cost <= 0.0 or stride < 1 or delay < 0:
        raise ValueError("cost and stride must be positive; delay nonnegative")
    message_updates = math.floor(message_budget / message_cost)
    environment_updates = math.floor(max(environment_budget - delay, 0.0) / stride)
    return int(min(message_updates, environment_updates))


def equicorrelated_ar_lag_covariances(
    *,
    horizon: int,
    single_agent_covariance: np.ndarray,
    q: int,
    rho: float,
    markov_lambda: float,
) -> np.ndarray:
    """Lag covariances K[k] for a uniform q-agent aggregate.

    The returned array has shape ``(2*horizon-1, d, d)`` and index
    ``horizon-1+k`` stores ``Cov(xi[t+k], xi[t])``.
    """

    covariance = np.asarray(single_agent_covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("single-agent covariance must be square")
    if horizon < 1 or q < 1:
        raise ValueError("horizon and q must be positive")
    if not 0.0 <= rho <= 1.0 or not 0.0 <= markov_lambda < 1.0:
        raise ValueError("rho and markov_lambda must lie in [0,1]")
    factor = rho + (1.0 - rho) / q
    lags = np.arange(-(horizon - 1), horizon)
    return np.asarray(
        [factor * markov_lambda ** abs(int(lag)) * covariance for lag in lags]
    )


def exact_vector_risk(
    *,
    initial_history: np.ndarray,
    drift: np.ndarray,
    step_size: float,
    delay: int,
    updates: int,
    lag_covariances: np.ndarray | None,
    risk_matrix: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute exact terminal first and second moments from stationary lags."""

    matrix = np.asarray(drift, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("drift must be square")
    dimension = matrix.shape[0]
    history = np.asarray(initial_history, dtype=float)
    if history.shape != (delay + 1, dimension):
        raise ValueError("initial_history must have shape (delay+1, dimension)")
    if updates < 0:
        raise ValueError("updates must be nonnegative")
    weight = np.eye(dimension) if risk_matrix is None else np.asarray(risk_matrix, dtype=float)
    if weight.shape != (dimension, dimension) or not np.allclose(weight, weight.T):
        raise ValueError("risk_matrix must be symmetric with drift dimension")

    companion = delayed_vector_companion(matrix, step_size, delay)
    selector = np.zeros((dimension, dimension * (delay + 1)))
    selector[:, :dimension] = np.eye(dimension)
    injector = selector.T
    lifted_initial = history.reshape(-1)
    terminal_mean = selector @ np.linalg.matrix_power(companion, updates) @ lifted_initial

    terminal_covariance = np.zeros((dimension, dimension))
    if updates:
        if lag_covariances is None:
            raise ValueError("lag_covariances are required when updates are positive")
        lags = np.asarray(lag_covariances, dtype=float)
        expected_shape = (2 * updates - 1, dimension, dimension)
        if lags.shape != expected_shape:
            raise ValueError(f"lag_covariances must have shape {expected_shape}")
        impulses = [
            selector @ np.linalg.matrix_power(companion, updates - 1 - time) @ injector
            for time in range(updates)
        ]
        center = updates - 1
        for left in range(updates):
            for right in range(updates):
                covariance = lags[center + left - right]
                terminal_covariance += (
                    step_size**2
                    * impulses[left]
                    @ covariance
                    @ impulses[right].T
                )
        terminal_covariance = (terminal_covariance + terminal_covariance.T) / 2.0

    bias_risk = float(terminal_mean @ weight @ terminal_mean)
    noise_risk = float(np.trace(weight @ terminal_covariance))
    return {
        "mean": terminal_mean,
        "covariance": terminal_covariance,
        "second_moment": terminal_covariance + np.outer(terminal_mean, terminal_mean),
        "bias_risk": bias_risk,
        "noise_risk": noise_risk,
        "risk": bias_risk + noise_risk,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(companion)))),
    }


def covariance_impulse_upper_bound(
    *,
    drift: np.ndarray,
    step_size: float,
    delay: int,
    updates: int,
    lag_covariances: np.ndarray,
    risk_matrix: np.ndarray | None = None,
) -> float:
    """Dimension-free trace-norm upper bound for the terminal noise risk.

    This is the direct finite-horizon inequality

        eta^2 sum_(s,r) ||Q^(1/2) H_s||op ||K_(s-r)||* ||Q^(1/2) H_r||op.

    It is intentionally computed from the finite impulse response instead of
    fitting an unspecified mixing constant.
    """

    matrix = np.asarray(drift, dtype=float)
    dimension = matrix.shape[0]
    weight = np.eye(dimension) if risk_matrix is None else np.asarray(risk_matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(weight)
    if np.min(eigenvalues) < -1e-12:
        raise ValueError("risk_matrix must be positive semidefinite")
    square_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    companion = delayed_vector_companion(matrix, step_size, delay)
    selector = np.zeros((dimension, dimension * (delay + 1)))
    selector[:, :dimension] = np.eye(dimension)
    injector = selector.T
    lags = np.asarray(lag_covariances, dtype=float)
    expected_shape = (max(2 * updates - 1, 0), dimension, dimension)
    if lags.shape != expected_shape:
        raise ValueError(f"lag_covariances must have shape {expected_shape}")
    norms = [
        np.linalg.norm(
            square_root
            @ selector
            @ np.linalg.matrix_power(companion, updates - 1 - time)
            @ injector,
            ord=2,
        )
        for time in range(updates)
    ]
    center = updates - 1
    bound = 0.0
    for left in range(updates):
        for right in range(updates):
            nuclear = np.linalg.norm(lags[center + left - right], ord="nuc")
            bound += norms[left] * nuclear * norms[right]
    return float(step_size**2 * bound)
