"""Exact scalar finite-horizon phase law for delayed multi-agent Markov SA."""

from __future__ import annotations

import math

import numpy as np


def effective_agents(q: int, rho: float) -> float:
    if q < 1:
        raise ValueError("q must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0,1]")
    return float(q / (1.0 + (q - 1) * rho))


def aggregate_variance(single_variance: float, q: int, rho: float) -> float:
    if single_variance < 0.0:
        raise ValueError("variance must be nonnegative")
    return float(single_variance / effective_agents(q, rho))


def update_count(budget: float, per_update_cost: float) -> int:
    if budget < 0.0 or per_update_cost <= 0.0:
        raise ValueError("budget must be nonnegative and cost positive")
    return int(math.floor(budget / per_update_cost))


def delayed_companion(mu: float, step_size: float, delay: int) -> np.ndarray:
    if mu <= 0.0 or step_size <= 0.0 or delay < 0:
        raise ValueError("mu and step size must be positive; delay nonnegative")
    matrix = np.zeros((delay + 1, delay + 1))
    matrix[0, 0] = 1.0
    matrix[0, delay] -= step_size * mu
    if delay:
        matrix[1:, :-1] = np.eye(delay)
    return matrix


def exact_scalar_risk(
    *,
    initial_error: float,
    mu: float,
    step_size: float,
    delay: int,
    updates: int,
    single_variance: float,
    q: int,
    rho: float,
    markov_lambda: float,
) -> dict[str, float]:
    """Exact terminal MSE for stationary Gaussian AR(1) aggregate noise.

    The recursion is e[t+1] = e[t] - eta*mu*e[t-delay] + eta*xi[t].
    The initial history is constant and equal to ``initial_error``.
    """

    if updates < 0:
        raise ValueError("updates must be nonnegative")
    if not 0.0 <= markov_lambda < 1.0:
        raise ValueError("markov_lambda must lie in [0,1)")
    transition = delayed_companion(mu, step_size, delay)
    initial_history = np.full(delay + 1, float(initial_error))
    terminal_mean = float((np.linalg.matrix_power(transition, updates) @ initial_history)[0])
    variance = aggregate_variance(single_variance, q, rho)
    if updates == 0 or variance == 0.0:
        terminal_variance = 0.0
    else:
        impulse = np.array(
            [
                np.linalg.matrix_power(transition, updates - 1 - time)[0, 0]
                for time in range(updates)
            ],
            dtype=float,
        )
        indices = np.arange(updates)
        autocorrelation = markov_lambda ** np.abs(indices[:, None] - indices[None, :])
        terminal_variance = float(
            step_size**2 * variance * impulse @ autocorrelation @ impulse
        )
    return {
        "mean_squared": terminal_mean**2,
        "noise_variance": terminal_variance,
        "risk": terminal_mean**2 + terminal_variance,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(transition)))),
    }


def noise_dominated_proxy(q: float, rho: float, overhead: float, per_agent: float) -> float:
    if q <= 0.0 or overhead < 0.0 or per_agent <= 0.0:
        raise ValueError("invalid q or cost")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0,1]")
    return float((rho + (1.0 - rho) / q) * (overhead + per_agent * q))


def continuous_proxy_optimum(rho: float, overhead: float, per_agent: float) -> float:
    """Continuous minimizer of the noise-dominated cost-risk product."""

    if overhead < 0.0 or per_agent <= 0.0:
        raise ValueError("invalid cost")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0,1]")
    if rho == 0.0:
        return math.inf
    if rho == 1.0 or overhead == 0.0:
        return 1.0
    return float(math.sqrt((1.0 - rho) * overhead / (rho * per_agent)))
