"""Sparse correlation probing under within-run dependence regime shifts."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from budget_participation import AGENT_COUNTS
from linear_model import make_agent_delays
from online_participation import (
    SCENARIOS,
    FiniteBudgetProxyCache,
    OnlineConfig,
    agent_metadata,
    choose_exploitation_action,
    true_aggregate_lrv,
)


POLICIES: Tuple[str, ...] = (
    "sparse_adaptive",
    "piecewise_oracle",
    "fixed_q1_oracle_eta",
    "fixed_q4_oracle_eta",
    "fixed_q8_oracle_eta",
    "fixed_q32_oracle_eta",
)

REGIME_SEQUENCE: Tuple[str, ...] = (
    "independent",
    "clustered",
    "global",
    "mixed",
)


@dataclass(frozen=True)
class DynamicConfig(OnlineConfig):
    total_budget: int = 32000
    checkpoint_count: int = 161
    block_budget: int = 2000
    blocks_per_regime: int = 4
    sparse_probe_q: int = 8
    sparse_probe_updates: int = 4
    rolling_probe_snapshots: int = 12

    @property
    def num_blocks(self) -> int:
        return len(REGIME_SEQUENCE) * self.blocks_per_regime

    @property
    def sparse_probe_cost(self) -> int:
        return self.num_blocks * self.sparse_probe_updates * (
            self.update_overhead + self.sparse_probe_q
        )


def regime_for_block(block: int, config: DynamicConfig) -> str:
    return REGIME_SEQUENCE[block // config.blocks_per_regime]


def estimate_dependence_components(
    snapshots: Sequence[np.ndarray],
    selected_indices: Sequence[int],
    config: DynamicConfig,
) -> Dict[str, float]:
    """Estimate three nonnegative covariance components from sparse probes."""

    if not snapshots:
        return {
            "rho_global": 0.0,
            "rho_cluster": 0.0,
            "rho_idiosyncratic": 1.0,
        }
    values = np.asarray(snapshots, dtype=float)
    selected = np.asarray(selected_indices, dtype=int)
    metadata = agent_metadata(config)
    design_rows: List[List[float]] = []
    targets: List[float] = []
    for local_i, agent_i in enumerate(selected):
        for local_j in range(local_i, len(selected)):
            agent_j = int(selected[local_j])
            moment = float(
                np.mean(values[:, local_i] * values[:, local_j])
            )
            same_agent = int(agent_i) == agent_j
            same_cluster = (
                metadata["clusters"][agent_i]
                == metadata["clusters"][agent_j]
            )
            design_rows.append(
                [
                    float(
                        metadata["global_loadings"][agent_i]
                        * metadata["global_loadings"][agent_j]
                    ),
                    float(
                        metadata["cluster_loadings"][agent_i]
                        * metadata["cluster_loadings"][agent_j]
                        if same_cluster
                        else 0.0
                    ),
                    float(1.0 if same_agent else 0.0),
                ]
            )
            targets.append(moment)
    estimate, _, _, _ = np.linalg.lstsq(
        np.asarray(design_rows, dtype=float),
        np.asarray(targets, dtype=float),
        rcond=None,
    )
    estimate = np.maximum(estimate, 0.0)
    total = float(np.sum(estimate))
    if total <= 1e-12:
        estimate = np.asarray([0.0, 0.0, 1.0])
    else:
        estimate = estimate / total
    return {
        "rho_global": float(estimate[0]),
        "rho_cluster": float(estimate[1]),
        "rho_idiosyncratic": float(estimate[2]),
    }


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


def simulate_dynamic_policy(
    policy: str,
    max_delay: int,
    noise_tables: Dict[str, np.ndarray],
    config: DynamicConfig,
    proxy_cache: Optional[FiniteBudgetProxyCache] = None,
) -> Dict[str, object]:
    if policy not in POLICIES:
        raise ValueError("unknown policy")
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
    action_rows: List[Dict[str, float]] = []
    rolling_snapshots: List[np.ndarray] = []
    update_count = 0
    charged_budget = 0
    observed_messages = 0
    total_probe_cost = 0
    per_block_valid = True

    def take_update(
        q: int,
        eta: float,
        scenario: str,
        block_start: int,
        block_spent: int,
    ) -> Tuple[int, np.ndarray]:
        nonlocal update_count, charged_budget, observed_messages
        time_index = maximum_delay + update_count
        selected = np.arange(q, dtype=int)
        noise = noise_tables[scenario][selected, update_count]
        delayed_x = x_buffer[time_index - delays[selected]]
        gradients = config.curvature * delayed_x - noise
        x_buffer[time_index + 1] = (
            x_buffer[time_index] - eta * float(np.mean(gradients))
        )
        cost = config.update_overhead + q
        update_count += 1
        charged_budget += cost
        observed_messages += q
        nominal_budgets.append(float(block_start + block_spent + cost))
        squared_errors.append(float(x_buffer[time_index + 1] ** 2))
        return cost, noise

    for block in range(config.num_blocks):
        scenario = regime_for_block(block, config)
        rho_global, rho_cluster = SCENARIOS[scenario]
        block_start = block * config.block_budget
        block_spent = 0
        block_probe_cost = 0

        if policy == "sparse_adaptive":
            for _ in range(config.sparse_probe_updates):
                cost, residuals = take_update(
                    config.sparse_probe_q,
                    config.default_eta,
                    scenario,
                    block_start,
                    block_spent,
                )
                block_spent += cost
                block_probe_cost += cost
                rolling_snapshots.append(np.asarray(residuals, dtype=float))
                rolling_snapshots = rolling_snapshots[
                    -config.rolling_probe_snapshots :
                ]
            components = estimate_dependence_components(
                snapshots=rolling_snapshots,
                selected_indices=np.arange(config.sparse_probe_q),
                config=config,
            )
            candidate_counts: Sequence[int] = AGENT_COUNTS
            lrv_by_q = {
                q: true_aggregate_lrv(
                    np.arange(q),
                    components["rho_global"],
                    components["rho_cluster"],
                    config,
                )
                for q in AGENT_COUNTS
            }
        else:
            components = {
                "rho_global": rho_global,
                "rho_cluster": rho_cluster,
                "rho_idiosyncratic": 1.0 - rho_global - rho_cluster,
            }
            if policy == "piecewise_oracle":
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
        history = _current_history(x_buffer, time_index, maximum_delay)
        action = choose_exploitation_action(
            candidate_counts=candidate_counts,
            lrv_by_q=lrv_by_q,
            delays=delays,
            current_history=history,
            remaining_budget=config.block_budget - block_spent,
            cache=cache,
            config=config,
        )
        selected_q = int(action["num_agents"])
        selected_eta = float(action["eta"])
        exploitation_cost = config.update_overhead + selected_q
        while block_spent + exploitation_cost <= config.block_budget:
            cost, _ = take_update(
                selected_q,
                selected_eta,
                scenario,
                block_start,
                block_spent,
            )
            block_spent += cost
        per_block_valid = bool(
            per_block_valid and block_spent <= config.block_budget
        )
        total_probe_cost += block_probe_cost
        block_end = (block + 1) * config.block_budget
        if nominal_budgets[-1] < block_end:
            nominal_budgets.append(float(block_end))
            squared_errors.append(float(squared_errors[-1]))
        action_rows.append(
            {
                "block": float(block),
                "scenario": scenario,
                "selected_num_agents": float(selected_q),
                "selected_eta": selected_eta,
                "block_spent": float(block_spent),
                "probe_cost": float(block_probe_cost),
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
    nominal_array = np.asarray(nominal_budgets, dtype=float)
    error_array = np.asarray(squared_errors, dtype=float)
    checkpoint_indices = np.searchsorted(
        nominal_array, checkpoints, side="right"
    ) - 1
    checkpoint_errors = error_array[checkpoint_indices]
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
