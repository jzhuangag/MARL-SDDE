"""Online, probe-charging participation control under clustered Markov noise."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from budget_participation import AGENT_COUNTS
from linear_model import make_agent_delays
from stagewise_controller import batch_means_lrv, build_delay_transition


POLICIES: Tuple[str, ...] = (
    "adaptive_probe",
    "probe_oracle",
    "all_agents_adaptive_eta",
    "fixed_q8_adaptive_eta",
    "fixed_q1_adaptive_eta",
)

SCENARIOS: Dict[str, Tuple[float, float]] = {
    "independent": (0.0, 0.0),
    "clustered": (0.0, 0.6),
    "global": (0.6, 0.0),
    "mixed": (0.3, 0.4),
}


@dataclass(frozen=True)
class OnlineConfig:
    num_agents: int = 32
    num_clusters: int = 4
    total_budget: int = 16000
    update_overhead: int = 4
    probe_updates: int = 80
    batch_size: int = 10
    checkpoint_count: int = 101
    eta_min: float = 0.0025
    eta_max: float = 0.08
    eta_count: int = 17
    default_eta: float = 0.02
    curvature: float = 1.0
    initial_error: float = 1.0
    delay_exponent: float = 1.25
    global_ar: float = 0.95
    cluster_ar: float = 0.70
    idiosyncratic_ar: float = 0.20
    stability_tolerance: float = 1e-8

    @property
    def eta_grid(self) -> np.ndarray:
        return np.geomspace(self.eta_min, self.eta_max, self.eta_count)

    @property
    def maximum_updates(self) -> int:
        return self.total_budget // (self.update_overhead + 1)

    @property
    def full_probe_cost(self) -> int:
        return self.probe_updates * (
            self.update_overhead + self.num_agents
        )


def agent_metadata(config: OnlineConfig) -> Dict[str, np.ndarray]:
    """Return interleaved clusters and deterministic heterogeneous loadings."""

    indices = np.arange(config.num_agents, dtype=int)
    clusters = indices % config.num_clusters
    global_rank = (7 * indices) % config.num_agents
    cluster_rank = (11 * indices) % config.num_agents
    global_loadings = 0.8 + 0.4 * global_rank / (config.num_agents - 1)
    cluster_loadings = 0.85 + 0.3 * cluster_rank / (
        config.num_agents - 1
    )
    return {
        "clusters": clusters,
        "global_loadings": global_loadings,
        "cluster_loadings": cluster_loadings,
    }


def _ar_path(
    rng: np.random.RandomState,
    coefficient: float,
    shape: Tuple[int, ...],
) -> np.ndarray:
    values = np.empty(shape, dtype=float)
    values[..., 0] = rng.normal(size=shape[:-1])
    innovation_scale = np.sqrt(1.0 - coefficient**2)
    for index in range(1, shape[-1]):
        values[..., index] = (
            coefficient * values[..., index - 1]
            + innovation_scale * rng.normal(size=shape[:-1])
        )
    return values


def generate_factor_paths(
    seed: int,
    maximum_delay: int,
    config: OnlineConfig,
) -> Dict[str, np.ndarray]:
    length = config.maximum_updates + maximum_delay + 1
    rng = np.random.RandomState(seed)
    return {
        "global": _ar_path(rng, config.global_ar, (length,)),
        "cluster": _ar_path(
            rng, config.cluster_ar, (config.num_clusters, length)
        ),
        "idiosyncratic": _ar_path(
            rng,
            config.idiosyncratic_ar,
            (config.num_agents, length),
        ),
    }


def true_aggregate_lrv(
    selected_indices: Sequence[int],
    rho_global: float,
    rho_cluster: float,
    config: OnlineConfig,
) -> float:
    selected = np.asarray(selected_indices, dtype=int)
    metadata = agent_metadata(config)
    q = float(len(selected))
    global_multiplier = (1.0 + config.global_ar) / (
        1.0 - config.global_ar
    )
    cluster_multiplier = (1.0 + config.cluster_ar) / (
        1.0 - config.cluster_ar
    )
    idiosyncratic_multiplier = (
        1.0 + config.idiosyncratic_ar
    ) / (1.0 - config.idiosyncratic_ar)

    mean_global_loading = float(
        np.mean(metadata["global_loadings"][selected])
    )
    cluster_weight_square = 0.0
    for cluster in range(config.num_clusters):
        members = selected[metadata["clusters"][selected] == cluster]
        if len(members):
            weight = float(
                np.sum(metadata["cluster_loadings"][members]) / q
            )
            cluster_weight_square += weight**2
    residual_fraction = 1.0 - rho_global - rho_cluster
    return float(
        rho_global * global_multiplier * mean_global_loading**2
        + rho_cluster * cluster_multiplier * cluster_weight_square
        + residual_fraction * idiosyncratic_multiplier / q
    )


class FiniteBudgetProxyCache:
    """Cache delayed transition powers and scalar-noise finite-horizon gains."""

    def __init__(self, config: OnlineConfig):
        self.config = config
        self._cache: Dict[
            Tuple[Tuple[int, ...], float, int], Dict[str, object]
        ] = {}

    def get(
        self,
        delays: Iterable[int],
        eta: float,
        horizon: int,
    ) -> Dict[str, object]:
        delay_tuple = tuple(int(value) for value in delays)
        key = (delay_tuple, float(eta), int(horizon))
        if key in self._cache:
            return self._cache[key]
        transition = build_delay_transition(
            eta=eta,
            curvature=self.config.curvature,
            delays=delay_tuple,
        )
        spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(transition)))
        )
        stable = spectral_radius < 1.0 - self.config.stability_tolerance
        result: Dict[str, object] = {
            "stable": stable,
            "spectral_radius": spectral_radius,
            "transition_power": None,
            "noise_gain": float("inf"),
        }
        if stable:
            impulse = np.zeros(transition.shape[0], dtype=float)
            impulse[0] = 1.0
            gain = 0.0
            for _ in range(horizon):
                gain += float(impulse[0] ** 2)
                impulse = transition.dot(impulse)
            result["transition_power"] = np.linalg.matrix_power(
                transition, horizon
            )
            result["noise_gain"] = eta**2 * gain
        self._cache[key] = result
        return result


def choose_exploitation_action(
    candidate_counts: Sequence[int],
    lrv_by_q: Dict[int, float],
    delays: np.ndarray,
    current_history: np.ndarray,
    remaining_budget: int,
    cache: FiniteBudgetProxyCache,
    config: OnlineConfig,
) -> Dict[str, float]:
    best: Optional[Dict[str, float]] = None
    best_key: Optional[Tuple[float, int, float]] = None
    for q in candidate_counts:
        horizon = remaining_budget // (config.update_overhead + int(q))
        if horizon < 1:
            continue
        selected_delays = delays[: int(q)]
        local_history = current_history[: int(np.max(selected_delays)) + 1]
        lrv = max(float(lrv_by_q[int(q)]), 1e-10)
        for eta in config.eta_grid:
            proxy = cache.get(selected_delays, float(eta), horizon)
            if not bool(proxy["stable"]):
                continue
            final_mean = np.asarray(proxy["transition_power"]).dot(
                local_history
            )
            risk = float(
                final_mean[0] ** 2 + lrv * float(proxy["noise_gain"])
            )
            key = (risk, -int(q), float(eta))
            if best_key is None or key < best_key:
                best_key = key
                best = {
                    "num_agents": float(q),
                    "eta": float(eta),
                    "proxy_risk": risk,
                    "lrv_used": lrv,
                    "predicted_exploitation_updates": float(horizon),
                }
    if best is None:
        raise RuntimeError("no stable exploitation action")
    return best


def _noise_vector(
    selected: np.ndarray,
    time_index: int,
    delays: np.ndarray,
    paths: Dict[str, np.ndarray],
    rho_global: float,
    rho_cluster: float,
    metadata: Dict[str, np.ndarray],
) -> np.ndarray:
    residual_fraction = 1.0 - rho_global - rho_cluster
    clusters = metadata["clusters"][selected]
    delayed_indices = time_index - delays[selected]
    return (
        np.sqrt(rho_global)
        * metadata["global_loadings"][selected]
        * paths["global"][time_index]
        + np.sqrt(rho_cluster)
        * metadata["cluster_loadings"][selected]
        * paths["cluster"][clusters, time_index]
        + np.sqrt(residual_fraction)
        * paths["idiosyncratic"][selected, delayed_indices]
    )


def build_noise_table(
    scenario: str,
    max_delay: int,
    paths: Dict[str, np.ndarray],
    config: OnlineConfig,
) -> np.ndarray:
    """Precompute the policy-independent agent noise path for one cell."""

    rho_global, rho_cluster = SCENARIOS[scenario]
    metadata = agent_metadata(config)
    delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=max_delay,
        exponent=config.delay_exponent,
    )
    times = max_delay + np.arange(config.maximum_updates, dtype=int)
    delayed_times = times[None, :] - delays[:, None]
    agent_indices = np.arange(config.num_agents, dtype=int)[:, None]
    cluster_values = paths["cluster"][
        metadata["clusters"][:, None], times[None, :]
    ]
    return (
        np.sqrt(rho_global)
        * metadata["global_loadings"][:, None]
        * paths["global"][times][None, :]
        + np.sqrt(rho_cluster)
        * metadata["cluster_loadings"][:, None]
        * cluster_values
        + np.sqrt(1.0 - rho_global - rho_cluster)
        * paths["idiosyncratic"][agent_indices, delayed_times]
    )


def _current_history(
    x_buffer: np.ndarray,
    time_index: int,
    maximum_delay: int,
) -> np.ndarray:
    return np.asarray(
        [
            x_buffer[time_index - delay]
            for delay in range(maximum_delay + 1)
        ],
        dtype=float,
    )


def simulate_policy(
    policy: str,
    scenario: str,
    max_delay: int,
    paths: Dict[str, np.ndarray],
    config: OnlineConfig,
    proxy_cache: Optional[FiniteBudgetProxyCache] = None,
    noise_table: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    rho_global, rho_cluster = SCENARIOS[scenario]
    metadata = agent_metadata(config)
    delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=max_delay,
        exponent=config.delay_exponent,
    )
    maximum_delay = int(np.max(delays))
    x_buffer = np.full(
        config.maximum_updates + maximum_delay + 1,
        config.initial_error,
        dtype=float,
    )
    budget_values: List[int] = [0]
    squared_errors: List[float] = [config.initial_error**2]
    observed_messages = 0
    budget_used = 0
    update_count = 0
    if noise_table is None:
        noise_table = build_noise_table(
            scenario=scenario,
            max_delay=max_delay,
            paths=paths,
            config=config,
        )

    if policy in (
        "adaptive_probe",
        "probe_oracle",
        "all_agents_adaptive_eta",
    ):
        probe_q = config.num_agents
    elif policy == "fixed_q8_adaptive_eta":
        probe_q = 8
    else:
        probe_q = 1

    if probe_q == config.num_agents:
        observed_streams: Dict[int, List[float]] = {
            q: [] for q in AGENT_COUNTS
        }
    else:
        observed_streams = {probe_q: []}

    def take_update(q: int, eta: float, record_probe: bool) -> None:
        nonlocal budget_used, observed_messages, update_count
        time_index = maximum_delay + update_count
        selected = np.arange(q, dtype=int)
        noise = noise_table[selected, update_count]
        delayed_x = x_buffer[time_index - delays[selected]]
        gradients = config.curvature * delayed_x - noise
        x_buffer[time_index + 1] = (
            x_buffer[time_index] - eta * float(np.mean(gradients))
        )
        if record_probe:
            prefix = np.cumsum(noise)
            for candidate_q in observed_streams:
                observed_streams[candidate_q].append(
                    float(prefix[candidate_q - 1] / candidate_q)
                )
        update_count += 1
        budget_used += config.update_overhead + q
        observed_messages += q
        budget_values.append(budget_used)
        squared_errors.append(float(x_buffer[time_index + 1] ** 2))

    for _ in range(config.probe_updates):
        take_update(probe_q, config.default_eta, True)
    probe_cost = budget_used

    estimated_lrv = {
        q: batch_means_lrv(values, config.batch_size)
        for q, values in observed_streams.items()
    }
    if policy == "probe_oracle":
        candidate_counts: Sequence[int] = AGENT_COUNTS
        lrv_by_q = {
            q: true_aggregate_lrv(
                np.arange(q),
                rho_global,
                rho_cluster,
                config,
            )
            for q in AGENT_COUNTS
        }
    elif policy == "adaptive_probe":
        candidate_counts = AGENT_COUNTS
        lrv_by_q = estimated_lrv
    elif policy == "all_agents_adaptive_eta":
        candidate_counts = (config.num_agents,)
        lrv_by_q = estimated_lrv
    elif policy == "fixed_q8_adaptive_eta":
        candidate_counts = (8,)
        lrv_by_q = estimated_lrv
    else:
        candidate_counts = (1,)
        lrv_by_q = estimated_lrv

    current_time_index = maximum_delay + update_count
    history = _current_history(
        x_buffer, current_time_index, maximum_delay
    )
    action = choose_exploitation_action(
        candidate_counts=candidate_counts,
        lrv_by_q=lrv_by_q,
        delays=delays,
        current_history=history,
        remaining_budget=config.total_budget - budget_used,
        cache=proxy_cache or FiniteBudgetProxyCache(config),
        config=config,
    )
    selected_q = int(action["num_agents"])
    selected_eta = float(action["eta"])
    update_cost = config.update_overhead + selected_q
    while budget_used + update_cost <= config.total_budget:
        take_update(selected_q, selected_eta, False)

    checkpoints = np.linspace(
        0, config.total_budget, config.checkpoint_count
    )
    budget_array = np.asarray(budget_values, dtype=float)
    error_array = np.asarray(squared_errors, dtype=float)
    checkpoint_indices = np.searchsorted(
        budget_array, checkpoints, side="right"
    ) - 1
    checkpoint_errors = error_array[checkpoint_indices]
    result_action = {
        "probe_num_agents": float(probe_q),
        "probe_updates": float(config.probe_updates),
        "probe_cost": float(probe_cost),
        "selected_num_agents": float(selected_q),
        "selected_eta": selected_eta,
        "estimated_selected_lrv": float(estimated_lrv.get(selected_q, np.nan)),
        "true_selected_lrv": true_aggregate_lrv(
            np.arange(selected_q),
            rho_global,
            rho_cluster,
            config,
        ),
        **action,
    }
    return {
        "checkpoint_errors": checkpoint_errors,
        "budget_checkpoints": checkpoints,
        "action": result_action,
        "budget_used": budget_used,
        "observed_messages": observed_messages,
        "total_updates": update_count,
        "finite": bool(np.all(np.isfinite(checkpoint_errors))),
        "within_budget": bool(budget_used <= config.total_budget),
    }
