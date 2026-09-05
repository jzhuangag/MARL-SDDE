"""Seven-state TD certificate tools for EXP-010A."""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from budget_participation import AGENT_COUNTS
from linear_model import make_agent_delays
from linear_td_correlation import LinearTDConfig, build_mrp

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


PERSISTENCES: Tuple[float, ...] = (0.0, 0.9, 0.98)
CORRELATIONS_TRANSFER: Tuple[float, ...] = (0.0, 0.9)
DELAYS_TRANSFER: Tuple[int, ...] = (0, 8)
TARGET_FRACTIONS: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75)
RESOURCE_BUDGET = 128_000
SERVER_OVERHEAD = 4


def build_transfer_mrp(
    persistence: float,
    config: Optional[LinearTDConfig] = None,
) -> Dict[str, np.ndarray]:
    """Build the frozen circulant MRP family and exact TD quantities."""

    if config is None:
        config = LinearTDConfig()
    if not 0.0 <= persistence < 1.0:
        raise ValueError("persistence must lie in [0,1)")
    template = build_mrp(config)
    count = config.num_states
    fast = np.full((count, count), 0.05 / count, dtype=float)
    for state in range(count):
        fast[state, (state + 1) % count] += 0.475
        fast[state, (state - 1) % count] += 0.475
    transition = (
        persistence * np.eye(count)
        + (1.0 - persistence) * fast
    )
    stationary = np.full(count, 1.0 / count, dtype=float)
    features = template["features"].copy()
    reward = template["reward"].copy()
    pairs = np.asarray(
        [(state, following) for state in range(count)
         for following in range(count)],
        dtype=np.int64,
    )
    pair_weights = (
        stationary[pairs[:, 0]]
        * transition[pairs[:, 0], pairs[:, 1]]
    )
    jacobians = np.asarray(
        [
            np.outer(
                features[state],
                features[state] - config.gamma * features[following],
            )
            for state, following in pairs
        ]
    )
    mean = np.einsum("m,mij->ij", pair_weights, jacobians)
    b_vector = np.einsum(
        "s,si,s->i", stationary, features, reward
    )
    theta_star = np.linalg.solve(mean, b_vector)
    td_noise = np.asarray(
        [
            features[state]
            * (
                reward[state]
                + config.gamma * features[following].dot(theta_star)
                - features[state].dot(theta_star)
            )
            for state, following in pairs
        ]
    )
    pair_transition = pair_chain_transition(transition)
    return {
        "transition": transition,
        "stationary": stationary,
        "reward": reward,
        "features": features,
        "pairs": pairs,
        "pair_weights": pair_weights,
        "jacobians": jacobians,
        "mean": mean,
        "b_vector": b_vector,
        "theta_star": theta_star,
        "td_noise": td_noise,
        "pair_transition": pair_transition,
        "gamma": np.asarray(config.gamma),
        "persistence": np.asarray(persistence),
    }


def pair_chain_transition(transition: np.ndarray) -> np.ndarray:
    """Transition matrix for the consecutive-pair chain (S_t,S_{t+1})."""

    count = transition.shape[0]
    result = np.zeros((count * count, count * count), dtype=float)
    for state in range(count):
        for following in range(count):
            source = state * count + following
            for next_state in range(count):
                target = following * count + next_state
                result[source, target] = transition[following, next_state]
    return result


def exact_pair_tv(model: Dict[str, np.ndarray], gap: int) -> float:
    """Worst-case total variation of a retained pair after ``gap`` starts."""

    if gap < 1:
        raise ValueError("gap must be positive")
    powered = np.linalg.matrix_power(model["pair_transition"], int(gap))
    stationary = model["pair_weights"]
    return float(0.5 * np.max(np.sum(np.abs(powered - stationary), axis=1)))


def minimum_joint_gap(
    model: Dict[str, np.ndarray],
    num_agents: int,
    target_delta: float,
    maximum_gap: int = 20_000,
) -> Dict[str, float]:
    """Smallest gap certified by the common-plus-idiosyncratic TV bound."""

    if target_delta <= 0.0:
        raise ValueError("target_delta must be positive")
    for gap in range(1, maximum_gap + 1):
        pair_delta = exact_pair_tv(model, gap)
        joint_delta = min(1.0, (num_agents + 1.0) * pair_delta)
        if joint_delta <= target_delta:
            return {
                "gap": int(gap),
                "pair_delta": pair_delta,
                "joint_delta": float(joint_delta),
            }
    raise RuntimeError("failed to certify a joint mixing gap")


def certificate_constants(
    model: Dict[str, np.ndarray],
    num_agents: int,
    rho: float,
    delays: Iterable[int],
    delta: float,
) -> Dict[str, float]:
    """Compute exact same-time constants and mixing-perturbed margins."""

    weights = model["pair_weights"]
    jacobians = model["jacobians"]
    mean = model["mean"]
    diagonal = np.einsum(
        "m,mji,mjk->ik", weights, jacobians, jacobians, optimize=True
    )
    off_diagonal = (
        rho * diagonal + (1.0 - rho) * mean.T.dot(mean)
    )
    aggregate = (
        diagonal / float(num_agents)
        + (num_agents - 1.0) / float(num_agents) * off_diagonal
    )
    curvature = float(np.max(np.linalg.eigvalsh(aggregate)))
    monotonicity = float(
        np.min(np.linalg.eigvalsh(0.5 * (mean + mean.T)))
    )
    lipschitz = float(
        max(np.linalg.norm(matrix, ord=2) for matrix in jacobians)
    )
    effective_monotonicity = monotonicity - 2.0 * lipschitz * delta
    effective_curvature = curvature + 2.0 * lipschitz ** 2 * delta
    delay_array = np.asarray(tuple(delays), dtype=float)
    rms_delay = float(np.sqrt(np.mean(delay_array ** 2)))
    return {
        "curvature": curvature,
        "monotonicity": monotonicity,
        "lipschitz": lipschitz,
        "effective_monotonicity": effective_monotonicity,
        "effective_curvature": effective_curvature,
        "rms_delay": rms_delay,
        "maximum_actual_delay": int(np.max(delay_array)),
    }


def _sharp_factor(eta: float, constants: Dict[str, float]) -> float:
    base = (
        1.0
        - 2.0 * eta * constants["effective_monotonicity"]
        + eta * eta * constants["effective_curvature"]
    )
    if base < 0.0:
        return float("inf")
    return float(
        np.sqrt(base)
        + eta
        * eta
        * constants["lipschitz"] ** 2
        * constants["rms_delay"]
    )


def rate_optimal_step(constants: Dict[str, float]) -> Dict[str, float]:
    """Solve the sharp scalar boundary and its rate-optimal interior step."""

    if constants["effective_monotonicity"] <= 0.0:
        raise ValueError("mixing removed the monotonicity margin")
    start = min(
        1e-8,
        1e-4
        * constants["effective_monotonicity"]
        / constants["effective_curvature"],
    )
    lower = start
    upper = max(
        2.0
        * constants["effective_monotonicity"]
        / constants["effective_curvature"],
        2.0 * start,
    )
    while _sharp_factor(upper, constants) < 1.0:
        lower = upper
        upper *= 1.5
        if upper > 100.0:
            raise RuntimeError("failed to bracket sharp boundary")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if _sharp_factor(midpoint, constants) < 1.0:
            lower = midpoint
        else:
            upper = midpoint
    boundary = 0.5 * (lower + upper)
    optimized = minimize_scalar(
        lambda value: _sharp_factor(value, constants),
        bounds=(start, boundary * (1.0 - 1e-10)),
        method="bounded",
        options={"xatol": 1e-13, "maxiter": 200},
    )
    eta = float(optimized.x)
    factor = _sharp_factor(eta, constants)
    return {
        "eta": eta,
        "sharp_boundary": float(boundary),
        "sharp_factor": factor,
        "contraction": factor * factor,
    }


def aggregate_td_noise(
    model: Dict[str, np.ndarray], num_agents: int, rho: float
) -> float:
    """Exact same-time E||average TD noise at theta_star||^2."""

    weights = model["pair_weights"]
    noise = model["td_noise"]
    diagonal = float(np.einsum("m,mi,mi->", weights, noise, noise))
    mean_noise = np.einsum("m,mi->i", weights, noise)
    independent = float(mean_noise.dot(mean_noise))
    off = rho * diagonal + (1.0 - rho) * independent
    return float(
        diagonal / num_agents
        + (num_agents - 1.0) / num_agents * off
    )


def candidate_actions(
    model: Dict[str, np.ndarray],
    rho: float,
    maximum_delay: int,
    resource_budget: int = RESOURCE_BUDGET,
    agent_counts: Iterable[int] = AGENT_COUNTS,
) -> Tuple[Dict[str, float], ...]:
    """Enumerate frozen certified actions and their risk surrogates."""

    actions = []
    full_delays = make_agent_delays(32, maximum_delay)
    initial_error = float(model["theta_star"].dot(model["theta_star"]))
    for num_agents in agent_counts:
        delays = full_delays[: int(num_agents)]
        base = certificate_constants(
            model, int(num_agents), rho, delays, delta=0.0
        )
        admissible = base["monotonicity"] / (2.0 * base["lipschitz"])
        for target_fraction in TARGET_FRACTIONS:
            target = target_fraction * admissible
            mixing = minimum_joint_gap(model, int(num_agents), target)
            constants = certificate_constants(
                model,
                int(num_agents),
                rho,
                delays,
                mixing["joint_delta"],
            )
            step = rate_optimal_step(constants)
            cost = SERVER_OVERHEAD + int(num_agents) + int(mixing["gap"])
            updates = int(resource_budget) // cost
            block_length = 2 * constants["maximum_actual_delay"] + 1
            blocks = updates // block_length
            omega = aggregate_td_noise(model, int(num_agents), rho)
            contraction = step["contraction"]
            risk = (
                contraction ** blocks * initial_error
                + step["eta"] ** 2
                * omega
                / max(1.0 - contraction, np.finfo(float).eps)
            )
            actions.append(
                {
                    "num_agents": int(num_agents),
                    "rho": float(rho),
                    "maximum_delay": int(maximum_delay),
                    "target_fraction": float(target_fraction),
                    **mixing,
                    **constants,
                    **step,
                    "update_cost": int(cost),
                    "updates": int(updates),
                    "blocks": int(blocks),
                    "omega": float(omega),
                    "initial_error": initial_error,
                    "risk_surrogate": float(risk),
                    "resource_budget": int(resource_budget),
                }
            )
    return tuple(actions)


def select_action(
    actions: Iterable[Dict[str, float]],
    restricted_q: Optional[int] = None,
) -> Dict[str, float]:
    """Select the deterministic minimum-risk certified action."""

    eligible = [
        action
        for action in actions
        if restricted_q is None or action["num_agents"] == restricted_q
    ]
    if not eligible:
        raise ValueError("no eligible action")
    return min(
        eligible,
        key=lambda row: (
            row["risk_surrogate"],
            row["num_agents"],
            row["gap"],
            row["eta"],
        ),
    ).copy()


if njit is not None:

    @njit(cache=True, nogil=True)
    def _categorical_row(cumulative: np.ndarray, state: int) -> int:
        uniform = np.random.random()
        for following in range(cumulative.shape[1]):
            if uniform <= cumulative[state, following]:
                return following
        return cumulative.shape[1] - 1


    @njit(cache=True, nogil=True)
    def _generate_unit_paths_kernel(
        seed: int,
        cumulative: np.ndarray,
        length: int,
        num_agents: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(seed)
        count = cumulative.shape[0]
        paths = np.empty(
            (num_agents + 1, length + 1), dtype=np.int16
        )
        for source in range(num_agents + 1):
            paths[source, 0] = np.random.randint(0, count)
            for time in range(length):
                paths[source, time + 1] = _categorical_row(
                    cumulative, int(paths[source, time])
                )
        masks = np.empty((num_agents, length), dtype=np.float32)
        for agent in range(num_agents):
            for time in range(length):
                masks[agent, time] = np.random.random()
        return paths, masks


    @njit(cache=True, nogil=True)
    def _simulate_td_paths_kernel(
        paths: np.ndarray,
        masks: np.ndarray,
        features: np.ndarray,
        reward: np.ndarray,
        theta_star: np.ndarray,
        delays: np.ndarray,
        rho: float,
        gap: int,
        eta: float,
        updates: int,
    ) -> Tuple[float, float, bool]:
        dimension = features.shape[1]
        num_agents = len(delays)
        maximum_delay = int(np.max(delays))
        history = np.zeros(
            (maximum_delay + updates + 1, dimension), dtype=np.float64
        )
        gradient = np.zeros(dimension, dtype=np.float64)
        share_probability = np.sqrt(rho)
        maximum_error = 0.0
        for update in range(updates):
            for coordinate in range(dimension):
                gradient[coordinate] = 0.0
            physical_time = update * gap
            for agent in range(num_agents):
                source = agent + 1
                if masks[agent, physical_time] < share_probability:
                    source = 0
                state = int(paths[source, physical_time])
                following = int(paths[source, physical_time + 1])
                stale_index = maximum_delay + update - int(delays[agent])
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
                    reward[state] + 0.9 * next_value - current_value
                )
                for coordinate in range(dimension):
                    gradient[coordinate] += (
                        features[state, coordinate] * td_error
                    )
            error = 0.0
            for coordinate in range(dimension):
                history[
                    maximum_delay + update + 1, coordinate
                ] = (
                    history[maximum_delay + update, coordinate]
                    + eta * gradient[coordinate] / num_agents
                )
                difference = (
                    history[maximum_delay + update + 1, coordinate]
                    - theta_star[coordinate]
                )
                error += difference * difference
            if error > maximum_error:
                maximum_error = error
            if not np.isfinite(error) or error > 1e12:
                return 1e12, maximum_error, True
        final_error = 0.0
        for coordinate in range(dimension):
            difference = (
                history[maximum_delay + updates, coordinate]
                - theta_star[coordinate]
            )
            final_error += difference * difference
        return final_error, maximum_error, False

else:  # pragma: no cover
    _generate_unit_paths_kernel = None
    _simulate_td_paths_kernel = None


def generate_unit_paths(
    seed: int,
    model: Dict[str, np.ndarray],
    length: int = RESOURCE_BUDGET,
    num_agents: int = 32,
) -> Dict[str, np.ndarray]:
    """Generate paired unit-time common/idiosyncratic paths and masks."""

    if _generate_unit_paths_kernel is None:
        raise RuntimeError("EXP-010A simulation requires numba")
    paths, masks = _generate_unit_paths_kernel(
        int(seed),
        np.cumsum(model["transition"], axis=1),
        int(length),
        int(num_agents),
    )
    return {"paths": paths, "masks": masks}


def simulate_certified_action(
    streams: Dict[str, np.ndarray],
    model: Dict[str, np.ndarray],
    action: Dict[str, float],
) -> Dict[str, object]:
    """Simulate one selected action on shared unit-time streams."""

    if _simulate_td_paths_kernel is None:
        raise RuntimeError("EXP-010A simulation requires numba")
    num_agents = int(action["num_agents"])
    full_delays = make_agent_delays(32, int(action["maximum_delay"]))
    delays = np.ascontiguousarray(full_delays[:num_agents], dtype=np.int64)
    final, maximum, diverged = _simulate_td_paths_kernel(
        streams["paths"],
        streams["masks"],
        model["features"],
        model["reward"],
        model["theta_star"],
        delays,
        float(action["rho"]),
        int(action["gap"]),
        float(action["eta"]),
        int(action["updates"]),
    )
    return {
        "squared_parameter_error": float(final),
        "maximum_squared_error": float(maximum),
        "diverged": bool(diverged),
        "finite": bool(np.isfinite(final) and np.isfinite(maximum)),
        "charged_budget": int(action["updates"] * action["update_cost"]),
        "within_budget": bool(
            action["updates"] * action["update_cost"]
            <= action["resource_budget"]
        ),
    }
