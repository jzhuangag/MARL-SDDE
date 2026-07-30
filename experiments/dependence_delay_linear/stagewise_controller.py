"""Predictable stagewise control for the delayed linear Markov model.

The controller observes scalar aggregate-gradient streams for a logarithmic
set of candidate participation levels.  At the end of each stage it estimates
one long-run variance per candidate via batch means.  Only those preceding-
stage statistics may determine the next stage's step size and participation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from linear_model import delay_histogram, make_agent_delays


POLICIES = [
    "adaptive_joint",
    "delay_only",
    "all_agents_adaptive_eta",
    "all_agents_fixed",
    "proxy_oracle",
]


@dataclass(frozen=True)
class StagewiseConfig:
    num_agents: int = 32
    agent_counts: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    num_stages: int = 12
    stage_length: int = 200
    curvature: float = 1.0
    common_ar: float = 0.95
    idiosyncratic_ar: float = 0.20
    initial_error: float = 1.0
    delay_exponent: float = 1.25
    batch_size: int = 20
    eta_min: float = 0.0025
    eta_max: float = 0.04
    eta_count: int = 13
    default_eta: float = 0.02
    stability_tolerance: float = 1e-8

    @property
    def total_steps(self) -> int:
        return self.num_stages * self.stage_length

    @property
    def eta_grid(self) -> np.ndarray:
        return np.geomspace(self.eta_min, self.eta_max, self.eta_count)

    @property
    def rho_schedule(self) -> Tuple[float, ...]:
        return (0.0,) * 4 + (0.9,) * 4 + (0.3,) * 4

    @property
    def max_delay_schedule(self) -> Tuple[int, ...]:
        return (4,) * 4 + (16,) * 4 + (8,) * 4

    def delay_profiles(self) -> List[np.ndarray]:
        return [
            make_agent_delays(
                max_agents=self.num_agents,
                max_delay=max_delay,
                exponent=self.delay_exponent,
            )
            for max_delay in self.max_delay_schedule
        ]


def build_delay_transition(
    eta: float,
    curvature: float,
    delays: Sequence[int],
) -> np.ndarray:
    """Build the deterministic companion matrix for one aggregate update."""

    weights = delay_histogram(delays)
    dimension = len(weights)
    transition = np.zeros((dimension, dimension), dtype=float)
    transition[0, 0] = 1.0
    transition[0, :] -= eta * curvature * weights
    if dimension > 1:
        transition[1:, :-1] = np.eye(dimension - 1)
    return transition


class DelayProxyCache:
    """Cache stable transitions, stage powers, and white-noise gains."""

    def __init__(self, config: StagewiseConfig):
        self.config = config
        self._cache: Dict[Tuple[Tuple[int, ...], float], Dict[str, object]] = {}

    def get(
        self,
        delays: Sequence[int],
        eta: float,
    ) -> Dict[str, object]:
        key = (tuple(int(value) for value in delays), float(eta))
        if key in self._cache:
            return self._cache[key]

        transition = build_delay_transition(
            eta=eta,
            curvature=self.config.curvature,
            delays=delays,
        )
        spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(transition)))
        )
        stable = spectral_radius < 1.0 - self.config.stability_tolerance
        result: Dict[str, object] = {
            "stable": stable,
            "spectral_radius": spectral_radius,
            "stage_power": None,
            "unit_lrv_gain": float("inf"),
        }
        if stable:
            stage_power = np.linalg.matrix_power(
                transition, self.config.stage_length
            )
            innovation = np.zeros_like(transition)
            innovation[0, 0] = eta**2
            covariance = solve_discrete_lyapunov(transition, innovation)
            result.update(
                {
                    "stage_power": stage_power,
                    "unit_lrv_gain": float(max(covariance[0, 0], 0.0)),
                }
            )
        self._cache[key] = result
        return result


def batch_means_lrv(series: np.ndarray, batch_size: int) -> float:
    """Estimate scalar long-run variance using non-overlapping batch means."""

    values = np.asarray(series, dtype=float)
    num_batches = len(values) // batch_size
    if num_batches < 3:
        raise ValueError("at least three complete batches are required")
    trimmed = values[: num_batches * batch_size]
    means = trimmed.reshape(num_batches, batch_size).mean(axis=1)
    estimate = batch_size * float(np.var(means, ddof=1))
    return float(np.clip(estimate, 1e-8, 1e4))


def estimate_stage_statistics(
    aggregate_gradients: np.ndarray,
    aggregate_regressors: np.ndarray,
    config: StagewiseConfig,
) -> Dict[int, Dict[str, float]]:
    """Estimate one drift slope and one residual LRV for every candidate q."""

    estimates: Dict[int, Dict[str, float]] = {}
    for index, num_agents in enumerate(config.agent_counts):
        gradient = aggregate_gradients[index]
        regressor = aggregate_regressors[index]
        centered_gradient = gradient - np.mean(gradient)
        centered_regressor = regressor - np.mean(regressor)
        denominator = float(np.dot(centered_regressor, centered_regressor))
        if denominator > 1e-10:
            slope = float(
                np.dot(centered_regressor, centered_gradient) / denominator
            )
            slope = float(np.clip(slope, 0.25, 2.0))
        else:
            slope = config.curvature
        intercept = float(np.mean(gradient) - slope * np.mean(regressor))
        residual = gradient - (intercept + slope * regressor)
        lrv = batch_means_lrv(residual, config.batch_size)
        estimates[int(num_agents)] = {
            "lrv": lrv,
            "slope": slope,
            "residual_variance": float(np.var(residual, ddof=1)),
        }
    return estimates


def true_long_run_variance(
    rho: float,
    num_agents: int,
    config: StagewiseConfig,
) -> float:
    """Return the zero-frequency variance for common plus idiosyncratic AR(1)."""

    common_multiplier = (1.0 + config.common_ar) / (1.0 - config.common_ar)
    idiosyncratic_multiplier = (
        1.0 + config.idiosyncratic_ar
    ) / (1.0 - config.idiosyncratic_ar)
    return float(
        rho * common_multiplier
        + (1.0 - rho) * idiosyncratic_multiplier / float(num_agents)
    )


def _current_history(
    x_buffer: np.ndarray,
    current_index: int,
    max_delay: int,
) -> np.ndarray:
    return np.asarray(
        [x_buffer[current_index - delay] for delay in range(max_delay + 1)],
        dtype=float,
    )


def choose_action(
    policy: str,
    stage: int,
    x_buffer: np.ndarray,
    current_index: int,
    previous_statistics: Optional[Dict[int, Dict[str, float]]],
    previous_delays: Optional[np.ndarray],
    current_delays: np.ndarray,
    current_rho: float,
    cache: DelayProxyCache,
    config: StagewiseConfig,
) -> Dict[str, float]:
    """Choose a stage action while enforcing the policy information pattern."""

    if policy not in POLICIES:
        raise ValueError("unknown policy: {0}".format(policy))
    if policy == "all_agents_fixed":
        return {
            "num_agents": float(config.num_agents),
            "eta": float(config.default_eta),
            "proxy_risk": float("nan"),
            "lrv_used": float("nan"),
        }

    if policy == "proxy_oracle":
        action_delays = current_delays
        lrv_by_q = {
            int(q): true_long_run_variance(current_rho, int(q), config)
            for q in config.agent_counts
        }
        candidate_counts = config.agent_counts
    elif previous_statistics is None or previous_delays is None:
        return {
            "num_agents": float(config.num_agents),
            "eta": float(config.default_eta),
            "proxy_risk": float("nan"),
            "lrv_used": float("nan"),
        }
    else:
        action_delays = previous_delays
        if policy == "delay_only":
            single_lrv = previous_statistics[1]["lrv"]
            lrv_by_q = {
                int(q): single_lrv / float(q) for q in config.agent_counts
            }
            candidate_counts = config.agent_counts
        elif policy == "all_agents_adaptive_eta":
            lrv_by_q = {
                config.num_agents: previous_statistics[config.num_agents]["lrv"]
            }
            candidate_counts = (config.num_agents,)
        else:
            lrv_by_q = {
                int(q): previous_statistics[int(q)]["lrv"]
                for q in config.agent_counts
            }
            candidate_counts = config.agent_counts

    best_action: Optional[Dict[str, float]] = None
    best_key: Optional[Tuple[float, int, float]] = None
    for num_agents in candidate_counts:
        selected_delays = action_delays[: int(num_agents)]
        max_delay = int(np.max(selected_delays))
        history = _current_history(x_buffer, current_index, max_delay)
        lrv = float(lrv_by_q[int(num_agents)])
        for eta in config.eta_grid:
            proxy = cache.get(selected_delays, float(eta))
            if not bool(proxy["stable"]):
                continue
            final_mean = np.asarray(proxy["stage_power"]).dot(history)
            squared_bias = float(final_mean[0] ** 2)
            risk = squared_bias + lrv * float(proxy["unit_lrv_gain"])
            key = (risk, -int(num_agents), float(eta))
            if best_key is None or key < best_key:
                best_key = key
                best_action = {
                    "num_agents": float(num_agents),
                    "eta": float(eta),
                    "proxy_risk": risk,
                    "lrv_used": lrv,
                }
    if best_action is None:
        raise RuntimeError("no stable controller action was found")
    return best_action


def generate_markov_paths(
    seed: int,
    config: StagewiseConfig,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Generate stationary common and idiosyncratic AR(1) paths."""

    rng = np.random.RandomState(seed)
    maximum_delay = int(max(config.max_delay_schedule))
    length = config.total_steps + maximum_delay + 1
    common = np.empty(length, dtype=float)
    idiosyncratic = np.empty((config.num_agents, length), dtype=float)
    common[0] = rng.normal()
    idiosyncratic[:, 0] = rng.normal(size=config.num_agents)
    common_scale = np.sqrt(1.0 - config.common_ar**2)
    idiosyncratic_scale = np.sqrt(1.0 - config.idiosyncratic_ar**2)
    for time_index in range(1, length):
        common[time_index] = (
            config.common_ar * common[time_index - 1]
            + common_scale * rng.normal()
        )
        idiosyncratic[:, time_index] = (
            config.idiosyncratic_ar * idiosyncratic[:, time_index - 1]
            + idiosyncratic_scale * rng.normal(size=config.num_agents)
        )
    return common, idiosyncratic, maximum_delay


def simulate_policy(
    policy: str,
    common: np.ndarray,
    idiosyncratic: np.ndarray,
    maximum_delay: int,
    alignment: str,
    config: StagewiseConfig,
) -> Dict[str, object]:
    """Simulate one policy on one paired Markov-noise path."""

    if alignment not in ("sample_time", "server_time"):
        raise ValueError("invalid alignment")

    x_buffer = np.full(
        config.total_steps + maximum_delay + 1,
        config.initial_error,
        dtype=float,
    )
    squared_errors = np.empty(config.total_steps + 1, dtype=float)
    squared_errors[0] = config.initial_error**2
    actions: List[Dict[str, float]] = []
    previous_statistics: Optional[Dict[int, Dict[str, float]]] = None
    previous_delays: Optional[np.ndarray] = None
    delay_profiles = config.delay_profiles()
    cache = DelayProxyCache(config)
    agent_indices = np.arange(config.num_agents)

    for stage in range(config.num_stages):
        rho = float(config.rho_schedule[stage])
        delays = delay_profiles[stage]
        current_time = stage * config.stage_length
        current_index = maximum_delay + current_time
        action = choose_action(
            policy=policy,
            stage=stage,
            x_buffer=x_buffer,
            current_index=current_index,
            previous_statistics=previous_statistics,
            previous_delays=previous_delays,
            current_delays=delays,
            current_rho=rho,
            cache=cache,
            config=config,
        )
        selected_q = int(action["num_agents"])
        eta = float(action["eta"])
        action_record = {
            "stage": float(stage),
            "rho": rho,
            "max_delay": float(np.max(delays)),
            **action,
        }
        actions.append(action_record)

        aggregate_gradients = np.empty(
            (len(config.agent_counts), config.stage_length), dtype=float
        )
        aggregate_regressors = np.empty_like(aggregate_gradients)

        for local_step in range(config.stage_length):
            global_step = current_time + local_step
            time_index = maximum_delay + global_step
            delayed_indices = time_index - delays
            delayed_x = x_buffer[delayed_indices]
            if alignment == "sample_time":
                common_component = common[delayed_indices]
            else:
                common_component = np.full(
                    config.num_agents, common[time_index], dtype=float
                )
            noise = (
                np.sqrt(rho) * common_component
                + np.sqrt(1.0 - rho)
                * idiosyncratic[agent_indices, delayed_indices]
            )
            gradients = config.curvature * delayed_x - noise
            gradient_prefix = np.cumsum(gradients)
            regressor_prefix = np.cumsum(delayed_x)
            for q_index, q in enumerate(config.agent_counts):
                aggregate_gradients[q_index, local_step] = (
                    gradient_prefix[q - 1] / float(q)
                )
                aggregate_regressors[q_index, local_step] = (
                    regressor_prefix[q - 1] / float(q)
                )

            selected_index = config.agent_counts.index(selected_q)
            x_buffer[time_index + 1] = (
                x_buffer[time_index]
                - eta * aggregate_gradients[selected_index, local_step]
            )
            squared_errors[global_step + 1] = x_buffer[time_index + 1] ** 2

        previous_statistics = estimate_stage_statistics(
            aggregate_gradients=aggregate_gradients,
            aggregate_regressors=aggregate_regressors,
            config=config,
        )
        previous_delays = delays.copy()

    return {
        "squared_errors": squared_errors,
        "actions": actions,
        "finite": bool(np.all(np.isfinite(squared_errors))),
    }
