"""Anytime latent-sharing certificate from observable sample collisions."""

from typing import Dict

import numpy as np


def stationary_collision_probability(
    rho: float, independent_collision: float
) -> float:
    """Return P(Y1=Y2) in the hidden pair-sharing mixture."""

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if not 0.0 <= independent_collision < 1.0:
        raise ValueError("independent_collision must lie in [0, 1)")
    return float(
        independent_collision
        + (1.0 - independent_collision) * rho
    )


def time_uniform_hoeffding_radius(trials: int, alpha: float) -> float:
    """Return a union-over-time Hoeffding radius for [0,1] observations."""

    if trials < 1:
        raise ValueError("trials must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    inside = np.pi ** 2 * trials ** 2 / (6.0 * alpha)
    return float(np.sqrt(np.log(inside) / (2.0 * trials)))


def latent_rho_upper(
    collisions: int,
    trials: int,
    cumulative_tv_bias: float,
    alpha: float,
    independent_collision_lower: float = 0.0,
) -> Dict[str, float]:
    """Return an anytime upper bound on hidden sharing probability rho."""

    if not 0 <= collisions <= trials or trials < 1:
        raise ValueError("invalid collision counts")
    if not 0.0 <= cumulative_tv_bias <= trials:
        raise ValueError("invalid cumulative_tv_bias")
    if not 0.0 <= independent_collision_lower < 1.0:
        raise ValueError(
            "independent_collision_lower must lie in [0, 1)"
        )
    empirical = collisions / float(trials)
    radius = time_uniform_hoeffding_radius(trials, alpha)
    average_bias = cumulative_tv_bias / float(trials)
    collision_upper = min(1.0, empirical + radius + average_bias)
    rho_upper = (
        collision_upper - independent_collision_lower
    ) / (1.0 - independent_collision_lower)
    return {
        "empirical_collision": float(empirical),
        "hoeffding_radius": float(radius),
        "average_tv_bias": float(average_bias),
        "collision_upper": float(collision_upper),
        "rho_upper": float(np.clip(rho_upper, 0.0, 1.0)),
    }


def symmetric_joint_tv_upper(
    persistence_upper: float,
    gap: int,
    num_hidden_chains: int = 3,
) -> float:
    """Bound joint TV after thinning symmetric two-state hidden chains."""

    if not 0.5 <= persistence_upper <= 1.0:
        raise ValueError("persistence_upper must lie in [0.5, 1]")
    if gap < 1 or num_hidden_chains < 1:
        raise ValueError("gap and num_hidden_chains must be positive")
    eigenvalue = 2.0 * persistence_upper - 1.0
    marginal = 0.5 * eigenvalue ** int(gap)
    return float(min(1.0, num_hidden_chains * marginal))


def minimum_collision_gap(
    persistence_upper: float,
    target_tv: float,
    num_hidden_chains: int = 3,
    maximum_gap: int = 100000,
) -> int:
    """Return the smallest thinning gap meeting a joint-TV target."""

    if not 0.0 < target_tv < 1.0:
        raise ValueError("target_tv must lie in (0, 1)")
    if persistence_upper >= 1.0:
        return int(maximum_gap + 1)
    for gap in range(1, maximum_gap + 1):
        if (
            symmetric_joint_tv_upper(
                persistence_upper, gap, num_hidden_chains
            )
            <= target_tv
        ):
            return int(gap)
    return int(maximum_gap + 1)


def sample_hidden_collision(
    rng: np.random.RandomState,
    states: np.ndarray,
    persistence: float,
    rho: float,
    gap: int,
) -> Dict[str, object]:
    """Advance three stationary chains and reveal only an agent collision."""

    if states.shape != (3,):
        raise ValueError("states must contain common and two private chains")
    eigenvalue = (2.0 * persistence - 1.0) ** int(gap)
    flip_probability = 0.5 * (1.0 - eigenvalue)
    flips = rng.binomial(1, flip_probability, size=3)
    next_states = np.bitwise_xor(states.astype(np.int64), flips)
    share_probability = np.sqrt(rho)
    first_shared = bool(rng.random_sample() < share_probability)
    second_shared = bool(rng.random_sample() < share_probability)
    first = next_states[0] if first_shared else next_states[1]
    second = next_states[0] if second_shared else next_states[2]
    return {
        "states": next_states.astype(np.int64),
        "collision": int(first == second),
        "first": int(first),
        "second": int(second),
    }
