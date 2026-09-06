"""Directional policy-dependency measurements for PettingZoo Pistonball.

The diagnostic uses one smooth scalar policy parameter per piston.  It avoids
training and measures the finite-horizon mixed return derivative for every
ordered pair of distinct policy blocks under common reset randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


def chain_neighbors(n_agents: int) -> dict[int, tuple[int, ...]]:
    if n_agents < 2:
        raise ValueError("n_agents must be at least two")
    return {
        owner: tuple(j for j in (owner - 1, owner + 1) if 0 <= j < n_agents)
        for owner in range(n_agents)
    }


def initial_ball_index(ball_x: float, n_agents: int, wall_width: float = 40.0, piston_width: float = 40.0) -> int:
    raw = int(np.floor((float(ball_x) - wall_width) / piston_width))
    return int(np.clip(raw, 0, n_agents - 1))


def active_window_support(
    n_agents: int,
    ball_index: int,
    radius: int,
) -> dict[int, tuple[int, ...]]:
    active = set(range(max(0, ball_index - radius), min(n_agents, ball_index + radius + 1)))
    return {
        owner: tuple(sorted(j for j in active if j != owner)) if owner in active else tuple()
        for owner in range(n_agents)
    }


def predictive_tube_support(
    n_agents: int,
    launch_x: float,
    launch_velocity_x: float,
    horizon: int,
    radius: int,
    wall_width: float = 40.0,
    piston_width: float = 40.0,
    joint_cycle_seconds: float = 1.0,
) -> dict[int, tuple[int, ...]]:
    """Complete directed support over a launch-time ballistic interaction tube."""

    start = initial_ball_index(launch_x, n_agents, wall_width, piston_width)
    predicted_x = float(launch_x) + float(launch_velocity_x) * int(horizon) * joint_cycle_seconds
    end = initial_ball_index(predicted_x, n_agents, wall_width, piston_width)
    lower = max(0, min(start, end) - int(radius))
    upper = min(n_agents - 1, max(start, end) + int(radius))
    active = set(range(lower, upper + 1))
    return {
        owner: tuple(sorted(j for j in active if j != owner)) if owner in active else tuple()
        for owner in range(n_agents)
    }


def _apply_joint_action(raw_env, actions: np.ndarray) -> None:
    if not raw_env.agents:
        return
    for actor, agent in enumerate(raw_env.possible_agents):
        if raw_env.terminations.get(agent, True) or raw_env.truncations.get(agent, True):
            raw_env.step(None)
        else:
            raw_env.step(np.asarray([actions[actor]], dtype=np.float32))


def heuristic_burn_in_action(raw_env) -> np.ndarray:
    """Public state-generation controller; it is never a learned comparator."""

    n_agents = raw_env.n_pistons
    ball_index = initial_ball_index(float(raw_env.ball.position.x), n_agents)
    actions = np.full(n_agents, -1.0, dtype=np.float32)
    actions[min(n_agents - 1, ball_index + 1)] = 1.0
    return actions


def finite_horizon_return(
    raw_env,
    theta: np.ndarray,
    reset_seed: int,
    horizon: int,
    burn_in: int = 0,
    action_noise: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Run a raw AEC Pistonball environment without materializing observations."""

    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (raw_env.n_pistons,):
        raise ValueError(f"theta must have shape {(raw_env.n_pistons,)}, got {theta.shape}")
    raw_env.reset(seed=int(reset_seed))
    for _ in range(int(burn_in)):
        if not raw_env.agents:
            break
        _apply_joint_action(raw_env, heuristic_burn_in_action(raw_env))
    launch_x = float(raw_env.ball.position.x)
    launch_velocity_x = float(raw_env.ball.velocity.x)
    total = 0.0
    if action_noise is None:
        action_noise = np.zeros((int(horizon), raw_env.n_pistons), dtype=np.float64)
    action_noise = np.asarray(action_noise, dtype=np.float64)
    if action_noise.shape != (int(horizon), raw_env.n_pistons):
        raise ValueError(
            f"action_noise must have shape {(int(horizon), raw_env.n_pistons)}, got {action_noise.shape}"
        )
    for time_index in range(int(horizon)):
        if not raw_env.agents:
            break
        actions = np.tanh(theta + action_noise[time_index]).astype(np.float32)
        _apply_joint_action(raw_env, actions)
        total += float(raw_env.rewards.get(raw_env.possible_agents[0], 0.0))
        if not raw_env.agents or all(
            raw_env.terminations[a] or raw_env.truncations[a] for a in raw_env.possible_agents
        ):
            break
    return total, launch_x, launch_velocity_x


def directional_influence_matrix(
    objective: Callable[[np.ndarray], float],
    theta: np.ndarray,
    step: float,
) -> np.ndarray:
    """Absolute central mixed derivative for all ordered policy-block pairs."""

    theta = np.asarray(theta, dtype=np.float64)
    n_agents = len(theta)
    influence = np.zeros((n_agents, n_agents), dtype=np.float64)
    for owner in range(n_agents):
        for donor in range(n_agents):
            if owner == donor:
                continue
            values: dict[tuple[int, int], float] = {}
            for owner_sign in (-1, 1):
                for donor_sign in (-1, 1):
                    candidate = np.array(theta, copy=True)
                    candidate[owner] += owner_sign * step
                    candidate[donor] += donor_sign * step
                    values[(owner_sign, donor_sign)] = float(objective(candidate))
            mixed = (
                values[(1, 1)]
                - values[(1, -1)]
                - values[(-1, 1)]
                + values[(-1, -1)]
            ) / (4.0 * step * step)
            influence[owner, donor] = abs(mixed)
    return influence


def score_cross_influence_matrix(
    raw_env,
    theta: np.ndarray,
    reset_seed: int,
    burn_in: int,
    horizon: int,
    donor_step: float,
    policy_noise_std: float,
    trajectory_seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate cross-policy gradient sensitivity with paired score functions.

    For owner ``i != j``, the returned signed quantity estimates
    ``d/d theta_j grad_i E[R]``.  The transformed-Gaussian policy is
    ``a_i,t=tanh(theta_i+sigma*z_i,t)``.  Common ``z`` and reset randomness
    are used for the donor's plus/minus perturbations.
    """

    theta = np.asarray(theta, dtype=np.float64)
    n_agents = len(theta)
    if donor_step <= 0.0 or policy_noise_std <= 0.0:
        raise ValueError("donor_step and policy_noise_std must be positive")
    samples = np.zeros((n_agents, len(trajectory_seeds), n_agents), dtype=np.float64)
    for donor in range(n_agents):
        plus = np.array(theta, copy=True)
        minus = np.array(theta, copy=True)
        plus[donor] += donor_step
        minus[donor] -= donor_step
        for sample_index, trajectory_seed in enumerate(trajectory_seeds):
            rng = np.random.default_rng(int(trajectory_seed))
            standard_noise = rng.normal(size=(int(horizon), n_agents))
            action_noise = policy_noise_std * standard_noise
            plus_return = finite_horizon_return(
                raw_env,
                plus,
                reset_seed,
                horizon,
                burn_in=burn_in,
                action_noise=action_noise,
            )[0]
            minus_return = finite_horizon_return(
                raw_env,
                minus,
                reset_seed,
                horizon,
                burn_in=burn_in,
                action_noise=action_noise,
            )[0]
            paired_return_derivative = (plus_return - minus_return) / (2.0 * donor_step)
            owner_scores = np.sum(standard_noise, axis=0) / policy_noise_std
            samples[donor, sample_index] = owner_scores * paired_return_derivative

    signed = np.zeros((n_agents, n_agents), dtype=np.float64)
    standard_error = np.zeros_like(signed)
    for donor in range(n_agents):
        signed[:, donor] = np.mean(samples[donor], axis=0)
        if len(trajectory_seeds) > 1:
            standard_error[:, donor] = np.std(samples[donor], axis=0, ddof=1) / np.sqrt(
                len(trajectory_seeds)
            )
    np.fill_diagonal(signed, 0.0)
    np.fill_diagonal(standard_error, 0.0)
    return signed, standard_error


def spsa_hessian_from_objectives(
    theta: np.ndarray,
    step: float,
    sample_seeds: Sequence[int],
    objective_builder: Callable[[int], Callable[[np.ndarray], float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a smoothed Hessian using paired bilinear Rademacher probes."""

    theta = np.asarray(theta, dtype=np.float64)
    if step <= 0.0:
        raise ValueError("step must be positive")
    dimension = len(theta)
    estimates = np.empty((len(sample_seeds), dimension, dimension), dtype=np.float64)
    for sample_index, sample_seed in enumerate(sample_seeds):
        rng = np.random.default_rng(int(sample_seed))
        owner_direction = rng.choice((-1.0, 1.0), size=dimension)
        donor_direction = rng.choice((-1.0, 1.0), size=dimension)
        objective = objective_builder(int(sample_seed))
        values: dict[tuple[int, int], float] = {}
        for owner_sign in (-1, 1):
            for donor_sign in (-1, 1):
                candidate = (
                    theta
                    + owner_sign * step * owner_direction
                    + donor_sign * step * donor_direction
                )
                values[(owner_sign, donor_sign)] = float(objective(candidate))
        bilinear = (
            values[(1, 1)]
            - values[(1, -1)]
            - values[(-1, 1)]
            + values[(-1, -1)]
        ) / (4.0 * step * step)
        estimates[sample_index] = bilinear * np.outer(owner_direction, donor_direction)
    mean = np.mean(estimates, axis=0)
    mean = 0.5 * (mean + mean.T)
    standard_error = np.zeros_like(mean)
    if len(sample_seeds) > 1:
        standard_error = np.std(estimates, axis=0, ddof=1) / np.sqrt(len(sample_seeds))
        standard_error = 0.5 * np.sqrt(standard_error**2 + standard_error.T**2)
    np.fill_diagonal(mean, 0.0)
    np.fill_diagonal(standard_error, 0.0)
    return mean, standard_error


def retained_fraction(influence: np.ndarray, support: Mapping[int, Iterable[int]]) -> float:
    total = float(np.sum(influence))
    if total <= 0.0:
        return float("nan")
    retained = sum(float(influence[i, j]) for i, donors in support.items() for j in donors)
    return retained / total


def random_degree_matched_support(
    rng: np.random.Generator,
    degrees: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    n_agents = len(degrees)
    result: dict[int, tuple[int, ...]] = {}
    for owner, degree in enumerate(degrees):
        candidates = np.asarray([j for j in range(n_agents) if j != owner])
        selected = rng.choice(candidates, size=int(degree), replace=False)
        result[owner] = tuple(sorted(int(j) for j in selected))
    return result


@dataclass(frozen=True)
class PistonTailSummary:
    ball_index: int
    active_radius: int
    active_size: int
    active_edge_fraction: float
    active_edge_cost_fraction: float
    chain_edge_fraction: float
    chain_edge_cost_fraction: float
    random_active_median_fraction: float
    active_minus_random_median: float
    nonzero_fraction: float


def summarize_support(
    influence: np.ndarray,
    support: Mapping[int, Iterable[int]],
    graph_seed: int,
    random_graphs: int = 256,
) -> tuple[float, float, float, float]:
    n_agents = influence.shape[0]
    degrees = [len(tuple(support[i])) for i in range(n_agents)]
    edge_count = sum(degrees)
    total_edges = n_agents * (n_agents - 1)
    rng = np.random.default_rng(graph_seed)
    random_values = np.asarray(
        [
            retained_fraction(influence, random_degree_matched_support(rng, degrees))
            for _ in range(random_graphs)
        ]
    )
    fraction = retained_fraction(influence, support)
    random_median = float(np.nanmedian(random_values))
    return float(fraction), float(edge_count / total_edges), random_median, float(fraction - random_median)


def summarize_piston_influence(
    influence: np.ndarray,
    ball_index: int,
    active_radius: int,
    graph_seed: int,
    random_graphs: int = 256,
) -> PistonTailSummary:
    n_agents = influence.shape[0]
    active_support = active_window_support(n_agents, ball_index, active_radius)
    chain_support = chain_neighbors(n_agents)
    active_degrees = [len(active_support[i]) for i in range(n_agents)]
    total_edges = n_agents * (n_agents - 1)
    active_edges = sum(active_degrees)
    chain_edges = sum(len(chain_support[i]) for i in range(n_agents))
    rng = np.random.default_rng(graph_seed)
    random_values = np.asarray(
        [
            retained_fraction(influence, random_degree_matched_support(rng, active_degrees))
            for _ in range(random_graphs)
        ]
    )
    active_fraction = retained_fraction(influence, active_support)
    random_median = float(np.nanmedian(random_values))
    scale = max(float(np.max(influence)), np.finfo(np.float64).tiny)
    return PistonTailSummary(
        ball_index=int(ball_index),
        active_radius=int(active_radius),
        active_size=int(sum(degree > 0 for degree in active_degrees)),
        active_edge_fraction=float(active_fraction),
        active_edge_cost_fraction=float(active_edges / total_edges),
        chain_edge_fraction=float(retained_fraction(influence, chain_support)),
        chain_edge_cost_fraction=float(chain_edges / total_edges),
        random_active_median_fraction=random_median,
        active_minus_random_median=float(active_fraction - random_median),
        nonzero_fraction=float(np.mean(influence[~np.eye(n_agents, dtype=bool)] > 1e-8 * scale)),
    )
