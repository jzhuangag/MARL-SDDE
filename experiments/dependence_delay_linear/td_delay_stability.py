"""Exact and Monte Carlo delayed-TD stability tools for EXP-007B."""

from typing import Dict, Tuple

import numpy as np

from linear_model import make_agent_delays
from linear_td_correlation import LinearTDConfig

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


AGENT_COUNTS_STABILITY: Tuple[int, ...] = (8, 16, 32)
MAX_DELAYS_STABILITY: Tuple[int, ...] = (0, 8, 32)
CORRELATIONS_STABILITY: Tuple[float, ...] = (0.0, 0.9)
MULTIPLIERS: Tuple[float, ...] = (0.50, 0.80, 0.95, 1.05, 1.20)
HORIZON = 4000
DIVERGENCE_THRESHOLD = 1e12


def build_mean_delay_transition(
    a_matrix: np.ndarray,
    delays: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Build the exact block companion matrix for delayed mean TD."""

    dimension = a_matrix.shape[0]
    maximum_delay = int(np.max(delays))
    transition = np.zeros(
        (
            dimension * (maximum_delay + 1),
            dimension * (maximum_delay + 1),
        ),
        dtype=float,
    )
    transition[:dimension, :dimension] = np.eye(dimension)
    for delay in delays:
        start = int(delay) * dimension
        transition[
            :dimension, start : start + dimension
        ] -= float(eta) * a_matrix / len(delays)
    for lag in range(1, maximum_delay + 1):
        row = lag * dimension
        previous = (lag - 1) * dimension
        transition[
            row : row + dimension,
            previous : previous + dimension,
        ] = np.eye(dimension)
    return transition


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def critical_step_size(
    a_matrix: np.ndarray,
    delays: np.ndarray,
    tolerance: float = 1e-10,
) -> float:
    """Find the first positive stability boundary by deterministic bisection."""

    lower = 0.0
    upper = 0.01
    while spectral_radius(
        build_mean_delay_transition(a_matrix, delays, upper)
    ) < 1.0:
        lower = upper
        upper *= 2.0
        if upper > 100.0:
            raise RuntimeError("failed to bracket delayed-TD boundary")
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        radius = spectral_radius(
            build_mean_delay_transition(a_matrix, delays, midpoint)
        )
        if radius < 1.0:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance * max(1.0, upper):
            break
    return float(0.5 * (lower + upper))


def build_boundary_table(
    a_matrix: np.ndarray,
    config: LinearTDConfig,
) -> Tuple[Dict[str, float], ...]:
    rows = []
    for q in AGENT_COUNTS_STABILITY:
        for max_delay in MAX_DELAYS_STABILITY:
            full_delays = make_agent_delays(
                max_agents=config.num_agents,
                max_delay=max_delay,
                exponent=config.delay_exponent,
            )
            delays = full_delays[:q]
            critical = critical_step_size(a_matrix, delays)
            rows.append(
                {
                    "num_agents": int(q),
                    "max_delay": int(max_delay),
                    "selected_max_delay": int(np.max(delays)),
                    "critical_eta": critical,
                }
            )
    return tuple(rows)


if njit is not None:

    @njit(cache=True, nogil=True)
    def _simulate_stability_kernel(
        current_states: np.ndarray,
        next_states: np.ndarray,
        features: np.ndarray,
        reward: np.ndarray,
        theta_star: np.ndarray,
        delays: np.ndarray,
        num_agents: int,
        eta: float,
        gamma: float,
        horizon: int,
        divergence_threshold: float,
    ) -> Tuple[float, int, bool]:
        dimension = features.shape[1]
        maximum_delay = int(np.max(delays[:num_agents]))
        history = np.empty(
            (maximum_delay + horizon + 1, dimension), dtype=np.float64
        )
        offset = 1.0 / np.sqrt(float(dimension))
        for index in range(maximum_delay + 1):
            for coordinate in range(dimension):
                history[index, coordinate] = (
                    theta_star[coordinate] + offset
                )
        gradient = np.zeros(dimension, dtype=np.float64)
        final_error = 1.0
        for update in range(horizon):
            for coordinate in range(dimension):
                gradient[coordinate] = 0.0
            for agent in range(num_agents):
                state = current_states[agent, update]
                next_state = next_states[agent, update]
                stale_index = maximum_delay + update - delays[agent]
                current_value = 0.0
                next_value = 0.0
                for coordinate in range(dimension):
                    current_value += (
                        features[state, coordinate]
                        * history[stale_index, coordinate]
                    )
                    next_value += (
                        features[next_state, coordinate]
                        * history[stale_index, coordinate]
                    )
                td_error = (
                    reward[state]
                    + gamma * next_value
                    - current_value
                )
                for coordinate in range(dimension):
                    gradient[coordinate] += (
                        features[state, coordinate] * td_error
                    )
            error = 0.0
            for coordinate in range(dimension):
                new_value = (
                    history[maximum_delay + update, coordinate]
                    + eta * gradient[coordinate] / num_agents
                )
                history[
                    maximum_delay + update + 1, coordinate
                ] = new_value
                difference = new_value - theta_star[coordinate]
                error += difference * difference
            final_error = error
            if (
                not np.isfinite(error)
                or error > divergence_threshold
            ):
                return divergence_threshold, update + 1, True
        return final_error, -1, False

else:  # pragma: no cover
    _simulate_stability_kernel = None


def simulate_stability_run(
    current_states: np.ndarray,
    next_states: np.ndarray,
    mrp: Dict[str, np.ndarray],
    max_delay: int,
    num_agents: int,
    eta: float,
    config: LinearTDConfig,
) -> Dict[str, object]:
    if _simulate_stability_kernel is None:
        raise RuntimeError("EXP-007B requires numba")
    full_delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=max_delay,
        exponent=config.delay_exponent,
    )
    final_error, crossing_time, crossed = _simulate_stability_kernel(
        current_states,
        next_states,
        mrp["features"],
        mrp["reward"],
        mrp["theta_star"],
        full_delays,
        int(num_agents),
        float(eta),
        config.gamma,
        HORIZON,
        DIVERGENCE_THRESHOLD,
    )
    return {
        "final_error": float(final_error),
        "crossing_time": int(crossing_time),
        "crossed_threshold": bool(crossed),
        "finite": bool(np.isfinite(final_error)),
    }
