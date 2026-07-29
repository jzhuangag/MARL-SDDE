"""Exact and Monte Carlo tools for a delayed linear Markov-noise model.

The server recursion is

    x[k+1] = x[k] - eta * a * sum_d w[d] x[k-d]
             + eta * sqrt(rho) * sum_d w[d] c[k-d]
             + eta * sqrt(1-rho) * e[k],

where c is a unit-variance AR(1) common factor and e is the average of q
independent, unit-variance AR(1) idiosyncratic factors.  Consequently,
Var(e[k]) = 1/q.  The delay histogram w is induced by the selected agents.

For deterministic delays this recursion is a linear time-invariant system.
Its finite-horizon and stationary mean-square errors can therefore be computed
without simulation by solving a discrete Lyapunov equation.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, toeplitz


@dataclass(frozen=True)
class ModelConfig:
    """Parameters shared by exact and Monte Carlo calculations."""

    curvature: float = 1.0
    common_ar: float = 0.95
    idiosyncratic_ar: float = 0.20
    initial_error: float = 1.0
    horizon: int = 500
    stability_tolerance: float = 1e-9
    common_noise_alignment: str = "sample_time"


def make_agent_delays(
    max_agents: int,
    max_delay: int,
    exponent: float = 1.25,
) -> np.ndarray:
    """Create a reproducible fast-to-slow deterministic delay profile."""

    if max_agents < 1:
        raise ValueError("max_agents must be positive")
    if max_delay < 0:
        raise ValueError("max_delay must be nonnegative")
    if max_agents == 1:
        return np.zeros(1, dtype=int)

    ranks = np.arange(max_agents, dtype=float) / float(max_agents - 1)
    delays = np.floor(max_delay * np.power(ranks, exponent) + 1e-12)
    return delays.astype(int)


def delay_histogram(delays: Iterable[int]) -> np.ndarray:
    """Convert an agent delay list into aggregation weights by delay."""

    values = np.asarray(list(delays), dtype=int)
    if values.size == 0:
        raise ValueError("at least one delay is required")
    if np.any(values < 0):
        raise ValueError("delays must be nonnegative")

    counts = np.bincount(values, minlength=int(values.max()) + 1)
    return counts.astype(float) / float(values.size)


def build_augmented_system(
    eta: float,
    rho: float,
    num_agents: int,
    delays: Iterable[int],
    config: ModelConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Build the augmented transition matrix and innovation covariance."""

    if eta <= 0:
        raise ValueError("eta must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if num_agents < 1:
        raise ValueError("num_agents must be positive")

    weights = delay_histogram(delays)
    max_delay = weights.size - 1
    history_size = max_delay + 1
    x_slice = slice(0, history_size)
    common_slice = slice(history_size, 2 * history_size)
    idiosyncratic_index = 2 * history_size
    dimension = idiosyncratic_index + 1

    transition = np.zeros((dimension, dimension), dtype=float)

    # x[k+1] update.
    transition[0, 0] = 1.0
    transition[0, x_slice] -= eta * config.curvature * weights
    if config.common_noise_alignment == "sample_time":
        transition[0, common_slice] += eta * np.sqrt(rho) * weights
    elif config.common_noise_alignment == "server_time":
        transition[0, history_size] += eta * np.sqrt(rho)
    else:
        raise ValueError(
            "common_noise_alignment must be 'sample_time' or 'server_time'"
        )
    transition[0, idiosyncratic_index] = eta * np.sqrt(1.0 - rho)

    # Shift x and common-factor histories.
    if history_size > 1:
        transition[1:history_size, 0 : history_size - 1] = np.eye(
            history_size - 1
        )
        transition[
            history_size + 1 : 2 * history_size,
            history_size : 2 * history_size - 1,
        ] = np.eye(history_size - 1)

    # AR(1) common and aggregate-idiosyncratic dynamics.
    transition[history_size, history_size] = config.common_ar
    transition[idiosyncratic_index, idiosyncratic_index] = (
        config.idiosyncratic_ar
    )

    innovation_covariance = np.zeros_like(transition)
    innovation_covariance[history_size, history_size] = 1.0 - config.common_ar**2
    innovation_covariance[idiosyncratic_index, idiosyncratic_index] = (
        1.0 - config.idiosyncratic_ar**2
    ) / float(num_agents)

    metadata = {
        "weights": weights,
        "max_delay": max_delay,
        "history_size": history_size,
        "x_slice": x_slice,
        "common_slice": common_slice,
        "idiosyncratic_index": idiosyncratic_index,
    }
    return transition, innovation_covariance, metadata


def initial_moments(
    num_agents: int,
    metadata: Dict[str, object],
    config: ModelConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize deterministic x histories and stationary Markov factors."""

    history_size = int(metadata["history_size"])
    dimension = 2 * history_size + 1
    common_slice = metadata["common_slice"]
    idiosyncratic_index = int(metadata["idiosyncratic_index"])

    mean = np.zeros(dimension, dtype=float)
    mean[0:history_size] = config.initial_error

    covariance = np.zeros((dimension, dimension), dtype=float)
    common_autocovariance = np.power(
        config.common_ar, np.arange(history_size, dtype=float)
    )
    covariance[common_slice, common_slice] = toeplitz(common_autocovariance)
    covariance[idiosyncratic_index, idiosyncratic_index] = 1.0 / float(
        num_agents
    )
    return mean, covariance


def exact_risk(
    eta: float,
    rho: float,
    num_agents: int,
    delays: Iterable[int],
    config: ModelConfig,
) -> Dict[str, float]:
    """Return stability, finite-horizon MSE, and stationary MSE."""

    transition, innovation_covariance, metadata = build_augmented_system(
        eta=eta,
        rho=rho,
        num_agents=num_agents,
        delays=delays,
        config=config,
    )
    eigenvalues = np.linalg.eigvals(transition)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    stable = spectral_radius < 1.0 - config.stability_tolerance

    result = {
        "stable": bool(stable),
        "spectral_radius": spectral_radius,
        "finite_mse": float("inf"),
        "stationary_mse": float("inf"),
        "squared_bias": float("inf"),
        "finite_variance": float("inf"),
    }
    if not stable:
        return result

    initial_mean, initial_covariance = initial_moments(
        num_agents=num_agents,
        metadata=metadata,
        config=config,
    )
    stationary_covariance = solve_discrete_lyapunov(
        transition, innovation_covariance
    )
    transition_power = np.linalg.matrix_power(transition, config.horizon)
    final_mean = transition_power.dot(initial_mean)
    final_covariance = stationary_covariance + transition_power.dot(
        initial_covariance - stationary_covariance
    ).dot(transition_power.T)

    squared_bias = float(final_mean[0] ** 2)
    finite_variance = float(max(final_covariance[0, 0], 0.0))
    stationary_mse = float(max(stationary_covariance[0, 0], 0.0))
    result.update(
        {
            "finite_mse": squared_bias + finite_variance,
            "stationary_mse": stationary_mse,
            "squared_bias": squared_bias,
            "finite_variance": finite_variance,
        }
    )
    return result


def monte_carlo_risk(
    eta: float,
    rho: float,
    num_agents: int,
    delays: Iterable[int],
    config: ModelConfig,
    num_replications: int,
    seed: int,
) -> Dict[str, float]:
    """Estimate the finite-horizon MSE for an exact-system cross-check."""

    transition, innovation_covariance, metadata = build_augmented_system(
        eta=eta,
        rho=rho,
        num_agents=num_agents,
        delays=delays,
        config=config,
    )
    initial_mean, initial_covariance = initial_moments(
        num_agents=num_agents,
        metadata=metadata,
        config=config,
    )

    rng = np.random.RandomState(seed)
    dimension = transition.shape[0]
    state = np.repeat(initial_mean[:, None], num_replications, axis=1)

    # Only the common-history block is correlated at initialization.
    common_slice = metadata["common_slice"]
    common_covariance = initial_covariance[common_slice, common_slice]
    common_samples = rng.multivariate_normal(
        mean=np.zeros(common_covariance.shape[0]),
        cov=common_covariance,
        size=num_replications,
    ).T
    state[common_slice, :] = common_samples

    idiosyncratic_index = int(metadata["idiosyncratic_index"])
    state[idiosyncratic_index, :] = rng.normal(
        loc=0.0,
        scale=np.sqrt(1.0 / float(num_agents)),
        size=num_replications,
    )

    common_index = int(metadata["history_size"])
    common_innovation_std = np.sqrt(
        innovation_covariance[common_index, common_index]
    )
    idiosyncratic_innovation_std = np.sqrt(
        innovation_covariance[idiosyncratic_index, idiosyncratic_index]
    )

    for _ in range(config.horizon):
        state = transition.dot(state)
        state[common_index, :] += rng.normal(
            loc=0.0,
            scale=common_innovation_std,
            size=num_replications,
        )
        state[idiosyncratic_index, :] += rng.normal(
            loc=0.0,
            scale=idiosyncratic_innovation_std,
            size=num_replications,
        )

    squared_errors = np.square(state[0, :])
    estimate = float(np.mean(squared_errors))
    standard_error = float(
        np.std(squared_errors, ddof=1) / np.sqrt(num_replications)
    )
    return {
        "mc_mse": estimate,
        "mc_standard_error": standard_error,
        "num_replications": int(num_replications),
        "seed": int(seed),
        "dimension": int(dimension),
    }
