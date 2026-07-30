"""Delayed linear TD under controlled cross-agent Markov dependence."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from budget_participation import AGENT_COUNTS
from linear_model import make_agent_delays

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


CORRELATIONS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
MAX_DELAYS: Tuple[int, ...] = (0, 8, 32)
BUDGETS: Tuple[int, ...] = (2000, 8000, 32000)


@dataclass(frozen=True)
class LinearTDConfig:
    num_states: int = 7
    num_features: int = 4
    num_agents: int = 32
    gamma: float = 0.9
    update_overhead: int = 4
    max_budget: int = 32000
    delay_exponent: float = 1.25
    eta_min: float = 0.001
    eta_max: float = 0.08
    eta_count: int = 13
    lrv_batch_size: int = 64

    @property
    def eta_grid(self) -> np.ndarray:
        return np.geomspace(self.eta_min, self.eta_max, self.eta_count)

    @property
    def maximum_updates(self) -> int:
        return self.max_budget // (self.update_overhead + 1)


def build_mrp(config: LinearTDConfig) -> Dict[str, np.ndarray]:
    """Construct the registered ring MRP and its projected TD solution."""

    count = config.num_states
    transition = np.full((count, count), 0.05 / count, dtype=float)
    for state in range(count):
        transition[state, state] += 0.65
        transition[state, (state + 1) % count] += 0.20
        transition[state, (state - 1) % count] += 0.10
    stationary = np.full(count, 1.0 / count, dtype=float)
    angles = 2.0 * np.pi * np.arange(count, dtype=float) / count
    reward = (
        0.6 * np.sin(angles)
        + 0.4 * np.cos(2.0 * angles)
        + 0.2
        * (np.arange(count, dtype=float) - (count - 1.0) / 2.0)
        / (count - 1.0)
    )
    raw_features = np.column_stack(
        [
            np.ones(count, dtype=float),
            np.cos(angles),
            np.sin(angles),
            np.cos(2.0 * angles),
        ]
    )
    orthonormal, _ = np.linalg.qr(raw_features)
    features = np.sqrt(float(count)) * orthonormal[:, : config.num_features]
    expected_next = transition.dot(features)
    a_matrix = np.zeros(
        (config.num_features, config.num_features), dtype=float
    )
    b_vector = np.zeros(config.num_features, dtype=float)
    for state in range(count):
        phi = features[state]
        a_matrix += stationary[state] * np.outer(
            phi, phi - config.gamma * expected_next[state]
        )
        b_vector += stationary[state] * phi * reward[state]
    theta_star = np.linalg.solve(a_matrix, b_vector)
    projected_value = features.dot(theta_star)
    return {
        "transition": transition,
        "stationary": stationary,
        "reward": reward,
        "features": features,
        "a_matrix": a_matrix,
        "b_vector": b_vector,
        "theta_star": theta_star,
        "projected_value": projected_value,
    }


def _sample_chain(
    rng: np.random.RandomState,
    transition: np.ndarray,
    length: int,
) -> np.ndarray:
    count = transition.shape[0]
    chain = np.empty(length, dtype=np.int64)
    chain[0] = rng.randint(0, count)
    cumulative = np.cumsum(transition, axis=1)
    uniforms = rng.rand(length - 1)
    for index, uniform in enumerate(uniforms, start=1):
        chain[index] = int(
            np.searchsorted(
                cumulative[chain[index - 1]], uniform, side="right"
            )
        )
    return chain


def generate_base_paths(
    seed: int,
    mrp: Dict[str, np.ndarray],
    config: LinearTDConfig,
) -> Dict[str, np.ndarray]:
    """Generate policy-independent common and idiosyncratic paths."""

    rng = np.random.RandomState(seed)
    length = config.maximum_updates + 1
    common = _sample_chain(rng, mrp["transition"], length)
    idiosyncratic = np.vstack(
        [
            _sample_chain(rng, mrp["transition"], length)
            for _ in range(config.num_agents)
        ]
    )
    masks = rng.rand(config.num_agents, config.maximum_updates)
    return {
        "common": common,
        "idiosyncratic": idiosyncratic,
        "mask_uniforms": masks,
    }


def observed_transition_pairs(
    paths: Dict[str, np.ndarray],
    rho: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return marginally exact MRP pairs with pairwise sharing probability rho."""

    use_common = paths["mask_uniforms"] < np.sqrt(float(rho))
    common_current = paths["common"][:-1][None, :]
    common_next = paths["common"][1:][None, :]
    current = np.where(
        use_common,
        common_current,
        paths["idiosyncratic"][:, :-1],
    )
    following = np.where(
        use_common,
        common_next,
        paths["idiosyncratic"][:, 1:],
    )
    return (
        np.ascontiguousarray(current, dtype=np.int64),
        np.ascontiguousarray(following, dtype=np.int64),
    )


if njit is not None:

    @njit(cache=True, nogil=True)
    def _td_budget_kernel(
        current_states: np.ndarray,
        next_states: np.ndarray,
        features: np.ndarray,
        reward: np.ndarray,
        theta_star: np.ndarray,
        delays: np.ndarray,
        num_agents: int,
        eta: float,
        gamma: float,
        budgets: np.ndarray,
        update_overhead: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dimension = features.shape[1]
        maximum_delay = int(np.max(delays[:num_agents]))
        update_cost = update_overhead + num_agents
        maximum_updates = int(budgets[-1] // update_cost)
        history = np.zeros(
            (maximum_delay + maximum_updates + 1, dimension),
            dtype=np.float64,
        )
        gradient = np.zeros(dimension, dtype=np.float64)
        for update in range(maximum_updates):
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
            for coordinate in range(dimension):
                history[
                    maximum_delay + update + 1, coordinate
                ] = (
                    history[maximum_delay + update, coordinate]
                    + eta * gradient[coordinate] / num_agents
                )
        errors = np.empty(len(budgets), dtype=np.float64)
        updates = np.empty(len(budgets), dtype=np.int64)
        finite = np.empty(len(budgets), dtype=np.bool_)
        for budget_index in range(len(budgets)):
            completed = int(budgets[budget_index] // update_cost)
            updates[budget_index] = completed
            error = 0.0
            for coordinate in range(dimension):
                difference = (
                    history[maximum_delay + completed, coordinate]
                    - theta_star[coordinate]
                )
                error += difference * difference
            errors[budget_index] = error
            finite[budget_index] = np.isfinite(error)
        return errors, updates, finite


    @njit(cache=True, nogil=True)
    def _td_eta_grid_kernel(
        current_states: np.ndarray,
        next_states: np.ndarray,
        features: np.ndarray,
        reward: np.ndarray,
        theta_star: np.ndarray,
        delays: np.ndarray,
        num_agents: int,
        eta_grid: np.ndarray,
        gamma: float,
        budgets: np.ndarray,
        update_overhead: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_errors = np.empty(
            (len(eta_grid), len(budgets)), dtype=np.float64
        )
        all_updates = np.empty(
            (len(eta_grid), len(budgets)), dtype=np.int64
        )
        all_finite = np.empty(
            (len(eta_grid), len(budgets)), dtype=np.bool_
        )
        for eta_index in range(len(eta_grid)):
            errors, updates, finite = _td_budget_kernel(
                current_states,
                next_states,
                features,
                reward,
                theta_star,
                delays,
                num_agents,
                eta_grid[eta_index],
                gamma,
                budgets,
                update_overhead,
            )
            all_errors[eta_index] = errors
            all_updates[eta_index] = updates
            all_finite[eta_index] = finite
        return all_errors, all_updates, all_finite

else:  # pragma: no cover
    _td_budget_kernel = None
    _td_eta_grid_kernel = None


def simulate_td_budget(
    current_states: np.ndarray,
    next_states: np.ndarray,
    mrp: Dict[str, np.ndarray],
    max_delay: int,
    num_agents: int,
    eta: float,
    config: LinearTDConfig,
) -> Dict[str, np.ndarray]:
    if _td_budget_kernel is None:
        raise RuntimeError("EXP-007A requires numba")
    if num_agents not in AGENT_COUNTS:
        raise ValueError("unregistered agent count")
    delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=max_delay,
        exponent=config.delay_exponent,
    )
    errors, updates, finite = _td_budget_kernel(
        current_states,
        next_states,
        mrp["features"],
        mrp["reward"],
        mrp["theta_star"],
        delays,
        int(num_agents),
        float(eta),
        config.gamma,
        np.asarray(BUDGETS, dtype=np.int64),
        config.update_overhead,
    )
    charged = updates * (config.update_overhead + int(num_agents))
    return {
        "errors": errors,
        "updates": updates,
        "charged_budgets": charged,
        "finite": finite,
    }


def simulate_td_eta_grid(
    current_states: np.ndarray,
    next_states: np.ndarray,
    mrp: Dict[str, np.ndarray],
    max_delay: int,
    num_agents: int,
    config: LinearTDConfig,
) -> Dict[str, np.ndarray]:
    """Evaluate the frozen step-size grid inside one compiled call."""

    if _td_eta_grid_kernel is None:
        raise RuntimeError("EXP-007A requires numba")
    if num_agents not in AGENT_COUNTS:
        raise ValueError("unregistered agent count")
    delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=max_delay,
        exponent=config.delay_exponent,
    )
    errors, updates, finite = _td_eta_grid_kernel(
        current_states,
        next_states,
        mrp["features"],
        mrp["reward"],
        mrp["theta_star"],
        delays,
        int(num_agents),
        np.asarray(config.eta_grid, dtype=np.float64),
        config.gamma,
        np.asarray(BUDGETS, dtype=np.int64),
        config.update_overhead,
    )
    update_cost = config.update_overhead + int(num_agents)
    return {
        "errors": errors,
        "updates": updates,
        "charged_budgets": updates * update_cost,
        "finite": finite,
    }


def td_noise_gradients(
    current_states: np.ndarray,
    next_states: np.ndarray,
    mrp: Dict[str, np.ndarray],
    config: LinearTDConfig,
) -> np.ndarray:
    """Evaluate every agent's stationary TD direction at theta_star."""

    features = mrp["features"]
    theta = mrp["theta_star"]
    current_features = features[current_states]
    next_features = features[next_states]
    current_values = np.einsum("atd,d->at", current_features, theta)
    next_values = np.einsum("atd,d->at", next_features, theta)
    td_errors = (
        mrp["reward"][current_states]
        + config.gamma * next_values
        - current_values
    )
    return current_features * td_errors[:, :, None]


def batch_means_trace_lrv(
    values: np.ndarray,
    batch_size: int,
) -> float:
    """Estimate trace long-run covariance using non-overlapping batches."""

    batch_count = values.shape[0] // batch_size
    if batch_count < 2:
        raise ValueError("at least two complete batches are required")
    trimmed = values[: batch_count * batch_size]
    means = trimmed.reshape(
        batch_count, batch_size, values.shape[1]
    ).mean(axis=1)
    return float(batch_size * np.var(means, axis=0, ddof=1).sum())


def effective_participation_rows(
    gradients: np.ndarray,
    rho: float,
    seed: int,
    config: LinearTDConfig,
) -> Tuple[Dict[str, float], ...]:
    rows = []
    lrv_q1 = None
    for q in AGENT_COUNTS:
        aggregate = np.mean(gradients[:q], axis=0)
        lrv = batch_means_trace_lrv(
            aggregate, config.lrv_batch_size
        )
        if q == 1:
            lrv_q1 = lrv
        rows.append(
            {
                "seed": int(seed),
                "rho": float(rho),
                "num_agents": int(q),
                "trace_lrv": float(lrv),
                "effective_participation": float(lrv_q1 / lrv),
            }
        )
    return tuple(rows)
