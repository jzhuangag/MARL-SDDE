"""Joint correlation--delay mean-square step tools for EXP-007C."""

from typing import Dict, Tuple

import numpy as np

from linear_model import make_agent_delays
from linear_td_correlation import LinearTDConfig
from td_delay_stability import (
    DIVERGENCE_THRESHOLD,
    build_mean_delay_transition,
    critical_step_size,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


AGENT_COUNTS_JOINT: Tuple[int, ...] = (16, 32)
MAX_DELAYS_JOINT: Tuple[int, ...] = (8, 32)
CORRELATIONS_JOINT: Tuple[float, ...] = (0.0, 0.9)
CHECKPOINTS = np.asarray((250, 500, 1000, 2000, 4000), dtype=np.int64)
HORIZON_JOINT = int(CHECKPOINTS[-1])
GRID_STEPS = np.geomspace(0.005, 0.8, 13)
POLICIES: Tuple[str, ...] = (
    "joint_aware",
    "correlation_blind",
    "delay_blind",
    "mean_only",
    "worstcase_correlation",
)


def single_jacobian_second_moment(
    mrp: Dict[str, np.ndarray],
    config: LinearTDConfig,
) -> np.ndarray:
    """Return B = E[H' H] exactly under the stationary finite MRP."""

    features = mrp["features"]
    transition = mrp["transition"]
    stationary = mrp["stationary"]
    result = np.zeros_like(mrp["a_matrix"])
    for state in range(config.num_states):
        phi = features[state]
        for following in range(config.num_states):
            difference = phi - config.gamma * features[following]
            jacobian = np.outer(phi, difference)
            weight = stationary[state] * transition[state, following]
            result += weight * jacobian.T.dot(jacobian)
    return result


def strong_monotonicity(a_matrix: np.ndarray) -> float:
    symmetric = 0.5 * (a_matrix + a_matrix.T)
    return float(np.min(np.linalg.eigvalsh(symmetric)))


def sharing_factor(num_agents: int, rho: float) -> float:
    return float(rho + (1.0 - rho) / float(num_agents))


def aggregate_second_moment(
    a_matrix: np.ndarray,
    single_second_moment: np.ndarray,
    num_agents: int,
    rho: float,
) -> np.ndarray:
    """Exact E[Hbar' Hbar] under registered exchangeable pair sharing."""

    alpha = sharing_factor(num_agents, rho)
    return (
        alpha * single_second_moment
        + (1.0 - alpha) * a_matrix.T.dot(a_matrix)
    )


def multiplicative_curvature(
    a_matrix: np.ndarray,
    single_second_moment: np.ndarray,
    num_agents: int,
    rho: float,
) -> float:
    matrix = aggregate_second_moment(
        a_matrix,
        single_second_moment,
        num_agents,
        rho,
    )
    return float(np.max(np.linalg.eigvalsh(matrix)))


def build_mean_boundaries(
    a_matrix: np.ndarray,
    config: LinearTDConfig,
) -> Dict[Tuple[int, int], float]:
    result: Dict[Tuple[int, int], float] = {}
    for num_agents in AGENT_COUNTS_JOINT:
        for max_delay in (0,) + MAX_DELAYS_JOINT:
            full_delays = make_agent_delays(
                max_agents=config.num_agents,
                max_delay=max_delay,
                exponent=config.delay_exponent,
            )
            result[(num_agents, max_delay)] = critical_step_size(
                a_matrix,
                full_delays[:num_agents],
            )
    return result


def joint_step_size(
    mean_boundary: float,
    curvature: float,
    monotonicity: float,
) -> float:
    if mean_boundary <= 0.0 or curvature <= 0.0 or monotonicity <= 0.0:
        raise ValueError("all joint step-size inputs must be positive")
    return float(
        1.0 / (1.0 / mean_boundary + curvature / (2.0 * monotonicity))
    )


def registered_policy_steps(
    a_matrix: np.ndarray,
    single_second_moment: np.ndarray,
    boundaries: Dict[Tuple[int, int], float],
    num_agents: int,
    max_delay: int,
    rho: float,
) -> Dict[str, float]:
    mu = strong_monotonicity(a_matrix)
    curvature = multiplicative_curvature(
        a_matrix, single_second_moment, num_agents, rho
    )
    independent_curvature = multiplicative_curvature(
        a_matrix, single_second_moment, num_agents, 0.0
    )
    worst_curvature = multiplicative_curvature(
        a_matrix, single_second_moment, num_agents, 1.0
    )
    local_boundary = boundaries[(num_agents, max_delay)]
    no_delay_boundary = boundaries[(num_agents, 0)]
    return {
        "joint_aware": joint_step_size(
            local_boundary, curvature, mu
        ),
        "correlation_blind": joint_step_size(
            local_boundary, independent_curvature, mu
        ),
        "delay_blind": joint_step_size(
            no_delay_boundary, curvature, mu
        ),
        "mean_only": float(local_boundary),
        "worstcase_correlation": joint_step_size(
            local_boundary, worst_curvature, mu
        ),
    }


if njit is not None:

    @njit(cache=True, nogil=True)
    def _simulate_checkpoint_kernel(
        current_states: np.ndarray,
        next_states: np.ndarray,
        features: np.ndarray,
        reward: np.ndarray,
        theta_star: np.ndarray,
        delays: np.ndarray,
        num_agents: int,
        eta: float,
        gamma: float,
        checkpoints: np.ndarray,
        divergence_threshold: float,
    ) -> Tuple[np.ndarray, int, int, bool]:
        dimension = features.shape[1]
        maximum_delay = int(np.max(delays[:num_agents]))
        horizon = int(checkpoints[-1])
        history = np.empty(
            (maximum_delay + horizon + 1, dimension),
            dtype=np.float64,
        )
        offset = 1.0 / np.sqrt(float(dimension))
        for index in range(maximum_delay + 1):
            for coordinate in range(dimension):
                history[index, coordinate] = (
                    theta_star[coordinate] + offset
                )
        gradient = np.zeros(dimension, dtype=np.float64)
        checkpoint_errors = np.empty(len(checkpoints), dtype=np.float64)
        checkpoint_index = 0
        crossing_time = -1
        half_error_time = -1
        for update in range(horizon):
            for coordinate in range(dimension):
                gradient[coordinate] = 0.0
            for agent in range(num_agents):
                state = current_states[agent, update]
                following = next_states[agent, update]
                stale_index = maximum_delay + update - delays[agent]
                current_value = 0.0
                next_value = 0.0
                for coordinate in range(dimension):
                    current_value += (
                        features[state, coordinate]
                        * history[stale_index, coordinate]
                    )
                    next_value += (
                        features[following, coordinate]
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
            completed = update + 1
            if half_error_time < 0 and error <= 0.5:
                half_error_time = completed
            if not np.isfinite(error) or error > divergence_threshold:
                crossing_time = completed
                while checkpoint_index < len(checkpoints):
                    checkpoint_errors[checkpoint_index] = (
                        divergence_threshold
                    )
                    checkpoint_index += 1
                return (
                    checkpoint_errors,
                    crossing_time,
                    half_error_time,
                    True,
                )
            if (
                checkpoint_index < len(checkpoints)
                and completed == checkpoints[checkpoint_index]
            ):
                checkpoint_errors[checkpoint_index] = error
                checkpoint_index += 1
        return (
            checkpoint_errors,
            crossing_time,
            half_error_time,
            False,
        )

else:  # pragma: no cover
    _simulate_checkpoint_kernel = None


def simulate_checkpoint_run(
    current_states: np.ndarray,
    next_states: np.ndarray,
    mrp: Dict[str, np.ndarray],
    max_delay: int,
    num_agents: int,
    eta: float,
    config: LinearTDConfig,
) -> Dict[str, object]:
    if _simulate_checkpoint_kernel is None:
        raise RuntimeError("EXP-007C requires numba")
    full_delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=max_delay,
        exponent=config.delay_exponent,
    )
    errors, crossing_time, half_error_time, crossed = (
        _simulate_checkpoint_kernel(
            current_states,
            next_states,
            mrp["features"],
            mrp["reward"],
            mrp["theta_star"],
            full_delays,
            int(num_agents),
            float(eta),
            config.gamma,
            CHECKPOINTS,
            DIVERGENCE_THRESHOLD,
        )
    )
    result: Dict[str, object] = {
        "crossed_threshold": bool(crossed),
        "crossing_time": int(crossing_time),
        "half_error_time": int(half_error_time),
        "final_error": float(errors[-1]),
        "finite": bool(np.all(np.isfinite(errors))),
    }
    for checkpoint, error in zip(CHECKPOINTS, errors):
        result["error_{0}".format(int(checkpoint))] = float(error)
    return result

