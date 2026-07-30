"""Observable state-and-correlation participation control for EXP-006B."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from budget_participation import AGENT_COUNTS
from linear_model import make_agent_delays
from online_participation import (
    FiniteBudgetProxyCache,
    OnlineConfig,
    agent_metadata,
    choose_exploitation_action,
    true_aggregate_lrv,
)
from sparse_dynamic import (
    _NUMBA_BLOCK_KERNEL,
    estimate_dependence_components,
)


SCENARIOS: Dict[str, Tuple[float, float]] = {
    "independent": (0.0, 0.0),
    "global_08": (0.8, 0.0),
    "clustered_08": (0.0, 0.8),
    "balanced_08": (0.4, 0.4),
}
POLICIES: Tuple[str, ...] = (
    "state_correlation_adaptive",
    "correlation_only_adaptive",
    "state_oracle",
    "fixed_q1_oracle_eta",
    "fixed_q4_oracle_eta",
    "fixed_q8_oracle_eta",
    "fixed_q32_oracle_eta",
)


@dataclass(frozen=True)
class StateCorrelationConfig(OnlineConfig):
    total_budget: int = 16000
    checkpoint_count: int = 81
    block_budget: int = 2000
    num_blocks: int = 8
    probe_q: int = 8
    probe_updates_per_block: int = 8
    rolling_probe_vectors: int = 32
    state_proxy_min: float = 0.01
    state_proxy_max: float = 1.0
    correlation_only_state: float = 0.3
    initial_error: float = 0.3

    @property
    def probe_cost(self) -> int:
        return self.num_blocks * self.probe_updates_per_block * (
            self.update_overhead + self.probe_q
        )


def build_noise_table_components(
    rho_global: float,
    rho_cluster: float,
    max_delay: int,
    paths: Dict[str, np.ndarray],
    config: StateCorrelationConfig,
) -> np.ndarray:
    """Build one policy-independent noise table from registered components."""

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


def estimate_observable_components(
    gradient_vectors: Sequence[np.ndarray],
    selected_indices: Sequence[int],
    config: StateCorrelationConfig,
) -> Dict[str, float]:
    """Estimate dependence after removing each agent's temporal mean."""

    if len(gradient_vectors) < 2:
        return {
            "rho_global": 0.0,
            "rho_cluster": 0.0,
            "rho_idiosyncratic": 1.0,
        }
    values = np.asarray(gradient_vectors, dtype=float)
    centered = values - np.mean(values, axis=0, keepdims=True)
    return estimate_dependence_components(
        snapshots=list(centered),
        selected_indices=selected_indices,
        config=config,
    )


def observable_state_proxy(
    block_gradient_vectors: Sequence[np.ndarray],
    config: StateCorrelationConfig,
) -> float:
    """Estimate error scale from the charged block-probe gradient mean."""

    values = np.asarray(block_gradient_vectors, dtype=float)
    raw = abs(float(np.mean(values))) / config.curvature
    return float(
        np.clip(raw, config.state_proxy_min, config.state_proxy_max)
    )


def _current_history(
    x_buffer: np.ndarray,
    time_index: int,
    maximum_delay: int,
) -> np.ndarray:
    return x_buffer[
        time_index - maximum_delay : time_index + 1
    ][::-1].copy()


def simulate_state_correlation_policy(
    policy: str,
    scenario: str,
    max_delay: int,
    noise_table: np.ndarray,
    config: StateCorrelationConfig,
    proxy_cache: Optional[FiniteBudgetProxyCache] = None,
) -> Dict[str, object]:
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    if _NUMBA_BLOCK_KERNEL is None:
        raise RuntimeError("EXP-006B requires numba")

    rho_global, rho_cluster = SCENARIOS[scenario]
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
    cache = proxy_cache or FiniteBudgetProxyCache(config)
    nominal_budgets: List[float] = [0.0]
    squared_errors: List[float] = [config.initial_error**2]
    action_rows: List[Dict[str, object]] = []
    rolling_gradients: List[np.ndarray] = []
    update_count = 0
    charged_budget = 0
    observed_messages = 0
    total_probe_cost = 0
    per_block_valid = True

    def take_probe_update(
        block_start: int,
        block_spent: int,
    ) -> Tuple[int, np.ndarray]:
        nonlocal update_count, charged_budget, observed_messages
        q = config.probe_q
        selected = np.arange(q, dtype=int)
        time_index = maximum_delay + update_count
        noise = noise_table[selected, update_count]
        delayed_x = x_buffer[time_index - delays[selected]]
        gradients = config.curvature * delayed_x - noise
        x_buffer[time_index + 1] = (
            x_buffer[time_index]
            - config.default_eta * float(np.mean(gradients))
        )
        cost = config.update_overhead + q
        update_count += 1
        charged_budget += cost
        observed_messages += q
        nominal_budgets.append(float(block_start + block_spent + cost))
        squared_errors.append(float(x_buffer[time_index + 1] ** 2))
        return cost, np.asarray(gradients, dtype=float)

    for block in range(config.num_blocks):
        block_start = block * config.block_budget
        block_spent = 0
        block_probe_cost = 0
        block_gradients: List[np.ndarray] = []

        if policy in (
            "state_correlation_adaptive",
            "correlation_only_adaptive",
        ):
            for _ in range(config.probe_updates_per_block):
                cost, gradients = take_probe_update(
                    block_start, block_spent
                )
                block_spent += cost
                block_probe_cost += cost
                block_gradients.append(gradients)
                rolling_gradients.append(gradients)
                rolling_gradients = rolling_gradients[
                    -config.rolling_probe_vectors :
                ]
            components = estimate_observable_components(
                rolling_gradients,
                np.arange(config.probe_q),
                config,
            )
            measured_state = observable_state_proxy(
                block_gradients, config
            )
            if policy == "state_correlation_adaptive":
                decision_state = measured_state
            else:
                decision_state = config.correlation_only_state
            decision_history = np.full(
                maximum_delay + 1, decision_state, dtype=float
            )
            candidate_counts: Sequence[int] = AGENT_COUNTS
            lrv_by_q = {
                q: true_aggregate_lrv(
                    np.arange(q),
                    components["rho_global"],
                    components["rho_cluster"],
                    config,
                )
                for q in candidate_counts
            }
        else:
            components = {
                "rho_global": rho_global,
                "rho_cluster": rho_cluster,
                "rho_idiosyncratic": 1.0 - rho_global - rho_cluster,
            }
            measured_state = float("nan")
            decision_state = float("nan")
            if policy == "state_oracle":
                candidate_counts = AGENT_COUNTS
            else:
                fixed_q = int(policy.split("_q")[1].split("_")[0])
                candidate_counts = (fixed_q,)
            lrv_by_q = {
                q: true_aggregate_lrv(
                    np.arange(q),
                    rho_global,
                    rho_cluster,
                    config,
                )
                for q in candidate_counts
            }
            time_index = maximum_delay + update_count
            decision_history = _current_history(
                x_buffer, time_index, maximum_delay
            )

        time_index = maximum_delay + update_count
        true_error = abs(float(x_buffer[time_index]))
        action = choose_exploitation_action(
            candidate_counts=candidate_counts,
            lrv_by_q=lrv_by_q,
            delays=delays,
            current_history=decision_history,
            remaining_budget=config.block_budget - block_spent,
            cache=cache,
            config=config,
        )
        selected_q = int(action["num_agents"])
        selected_eta = float(action["eta"])
        exploitation_cost = config.update_overhead + selected_q
        exploitation_updates = (
            config.block_budget - block_spent
        ) // exploitation_cost
        squared_block = _NUMBA_BLOCK_KERNEL(
            x_buffer,
            maximum_delay,
            update_count,
            delays,
            noise_table,
            selected_q,
            selected_eta,
            config.curvature,
            exploitation_updates,
        )
        update_numbers = np.arange(
            1, exploitation_updates + 1, dtype=float
        )
        nominal_budgets.extend(
            (
                block_start
                + block_spent
                + exploitation_cost * update_numbers
            ).tolist()
        )
        squared_errors.extend(squared_block.tolist())
        update_count += exploitation_updates
        exploitation_spend = exploitation_updates * exploitation_cost
        charged_budget += exploitation_spend
        observed_messages += exploitation_updates * selected_q
        block_spent += exploitation_spend
        total_probe_cost += block_probe_cost
        per_block_valid = bool(
            per_block_valid and block_spent <= config.block_budget
        )
        block_end = (block + 1) * config.block_budget
        if nominal_budgets[-1] < block_end:
            nominal_budgets.append(float(block_end))
            squared_errors.append(float(squared_errors[-1]))
        action_rows.append(
            {
                "block": int(block),
                "scenario": scenario,
                "selected_num_agents": selected_q,
                "selected_eta": selected_eta,
                "block_spent": block_spent,
                "probe_cost": block_probe_cost,
                "observable_state_proxy": measured_state,
                "decision_state_proxy": decision_state,
                "true_error_magnitude": true_error,
                "estimated_rho_global": components["rho_global"],
                "estimated_rho_cluster": components["rho_cluster"],
                "estimated_rho_idiosyncratic": components[
                    "rho_idiosyncratic"
                ],
                "true_rho_global": rho_global,
                "true_rho_cluster": rho_cluster,
                **action,
            }
        )

    checkpoints = np.linspace(
        0, config.total_budget, config.checkpoint_count
    )
    checkpoint_indices = np.searchsorted(
        np.asarray(nominal_budgets, dtype=float),
        checkpoints,
        side="right",
    ) - 1
    checkpoint_errors = np.asarray(
        squared_errors, dtype=float
    )[checkpoint_indices]
    return {
        "checkpoint_errors": checkpoint_errors,
        "actions": action_rows,
        "charged_budget": charged_budget,
        "observed_messages": observed_messages,
        "total_probe_cost": total_probe_cost,
        "total_updates": update_count,
        "finite": bool(np.all(np.isfinite(checkpoint_errors))),
        "within_budget": bool(
            charged_budget <= config.total_budget and per_block_valid
        ),
    }
