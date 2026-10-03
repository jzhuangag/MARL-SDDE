"""Finite-horizon cross-policy dependency measurements for HalfCheetah 6x1.

The module deliberately contains no learning controller.  It measures mixed
finite differences of a shared finite-horizon return under six distinct smooth
policy blocks.  The resulting block Frobenius norms are a design-stage proxy
for the true policy-gradient dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


ACTION_DIM = 6
PARAM_DIM = 4

# Gymnasium HalfCheetah action order is bthigh, bshin, bfoot, fthigh,
# fshin, ffoot.  Multi-Agent MuJoCo's physical chain is
# bfoot-bshin-bthigh-fthigh-fshin-ffoot.
PHYSICAL_CHAIN = (2, 1, 0, 3, 4, 5)


def physical_neighbors() -> dict[int, tuple[int, ...]]:
    neighbors: dict[int, set[int]] = {i: set() for i in range(ACTION_DIM)}
    for left, right in zip(PHYSICAL_CHAIN[:-1], PHYSICAL_CHAIN[1:]):
        neighbors[left].add(right)
        neighbors[right].add(left)
    return {i: tuple(sorted(values)) for i, values in neighbors.items()}


def local_features(observation: np.ndarray, actor: int) -> np.ndarray:
    """Return a fixed four-dimensional local feature vector.

    HalfCheetah-v5 exposes qpos[1:] followed by qvel.  For actuator ``actor``,
    indices 2+actor and 11+actor are its joint angle and angular velocity.
    Torso pitch is shared proprioception rather than teammate-policy data.
    """

    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (17,):
        raise ValueError(f"expected HalfCheetah observation shape (17,), got {observation.shape}")
    if not 0 <= actor < ACTION_DIM:
        raise ValueError(f"actor must be in [0,{ACTION_DIM}), got {actor}")
    return np.asarray(
        [1.0, observation[1], observation[2 + actor], observation[11 + actor]],
        dtype=np.float64,
    )


def smooth_action(observation: np.ndarray, theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (ACTION_DIM, PARAM_DIM):
        raise ValueError(f"theta must have shape {(ACTION_DIM, PARAM_DIM)}, got {theta.shape}")
    action = np.empty(ACTION_DIM, dtype=np.float64)
    for actor in range(ACTION_DIM):
        action[actor] = np.tanh(np.dot(theta[actor], local_features(observation, actor)))
    return action


def fixed_policy_parameters(policy_seed: int, scale: float = 0.12) -> np.ndarray:
    rng = np.random.default_rng(policy_seed)
    theta = rng.normal(scale=scale, size=(ACTION_DIM, PARAM_DIM))
    theta[:, 0] *= 0.5
    return theta.astype(np.float64)


def finite_horizon_return(env, theta: np.ndarray, reset_seed: int, horizon: int) -> float:
    observation, _ = env.reset(seed=int(reset_seed))
    total = 0.0
    for _ in range(int(horizon)):
        observation, reward, terminated, truncated, _ = env.step(smooth_action(observation, theta))
        total += float(reward)
        if terminated or truncated:
            break
    return total


def mixed_block(
    objective: Callable[[np.ndarray], float],
    theta: np.ndarray,
    owner: int,
    donor: int,
    owner_step: float,
    donor_step: float,
) -> np.ndarray:
    """Central mixed finite-difference block d^2 J/(d theta_i d theta_j)."""

    if owner == donor:
        raise ValueError("owner and donor must be distinct")
    block = np.empty((PARAM_DIM, PARAM_DIM), dtype=np.float64)
    for owner_coordinate in range(PARAM_DIM):
        for donor_coordinate in range(PARAM_DIM):
            values: dict[tuple[int, int], float] = {}
            for owner_sign in (-1, 1):
                for donor_sign in (-1, 1):
                    candidate = np.array(theta, copy=True)
                    candidate[owner, owner_coordinate] += owner_sign * owner_step
                    candidate[donor, donor_coordinate] += donor_sign * donor_step
                    values[(owner_sign, donor_sign)] = float(objective(candidate))
            block[owner_coordinate, donor_coordinate] = (
                values[(1, 1)]
                - values[(1, -1)]
                - values[(-1, 1)]
                + values[(-1, -1)]
            ) / (4.0 * owner_step * donor_step)
    return block


def influence_matrix(
    objective: Callable[[np.ndarray], float],
    theta: np.ndarray,
    owner_step: float,
    donor_step: float,
) -> np.ndarray:
    influence = np.zeros((ACTION_DIM, ACTION_DIM), dtype=np.float64)
    for owner in range(ACTION_DIM):
        for donor in range(ACTION_DIM):
            if owner == donor:
                continue
            block = mixed_block(objective, theta, owner, donor, owner_step, donor_step)
            influence[owner, donor] = np.linalg.norm(block, ord="fro")
    return influence


def retained_fraction(
    influence: np.ndarray,
    supports: Mapping[int, Iterable[int]],
) -> float:
    influence = np.asarray(influence, dtype=np.float64)
    total = float(influence.sum())
    if total <= 0.0:
        return float("nan")
    retained = sum(float(influence[owner, donor]) for owner, donors in supports.items() for donor in donors)
    return retained / total


def owner_retained_fractions(
    influence: np.ndarray,
    supports: Mapping[int, Iterable[int]],
) -> np.ndarray:
    values = []
    for owner, donors in supports.items():
        denominator = float(np.sum(influence[owner]))
        numerator = sum(float(influence[owner, donor]) for donor in donors)
        values.append(numerator / denominator if denominator > 0.0 else np.nan)
    return np.asarray(values, dtype=np.float64)


def random_degree_matched_supports(
    rng: np.random.Generator,
    degrees: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    supports: dict[int, tuple[int, ...]] = {}
    for owner, degree in enumerate(degrees):
        candidates = np.asarray([j for j in range(ACTION_DIM) if j != owner], dtype=np.int64)
        selected = rng.choice(candidates, size=int(degree), replace=False)
        supports[owner] = tuple(sorted(int(value) for value in selected))
    return supports


@dataclass(frozen=True)
class TailSummary:
    physical_fraction: float
    owner_median_fraction: float
    random_median_fraction: float
    random_p90_fraction: float
    physical_minus_random_median: float
    neighbor_top_rate: float


def summarize_influence(
    influence: np.ndarray,
    random_graphs: int,
    graph_seed: int,
) -> TailSummary:
    physical = physical_neighbors()
    degrees = [len(physical[i]) for i in range(ACTION_DIM)]
    physical_fraction = retained_fraction(influence, physical)
    owner_median = float(np.nanmedian(owner_retained_fractions(influence, physical)))

    rng = np.random.default_rng(graph_seed)
    random_values = np.asarray(
        [
            retained_fraction(influence, random_degree_matched_supports(rng, degrees))
            for _ in range(int(random_graphs))
        ],
        dtype=np.float64,
    )
    top_hits = []
    for owner, neighbors in physical.items():
        best_donor = int(np.argmax(influence[owner]))
        top_hits.append(best_donor in neighbors)

    random_median = float(np.nanmedian(random_values))
    return TailSummary(
        physical_fraction=float(physical_fraction),
        owner_median_fraction=owner_median,
        random_median_fraction=random_median,
        random_p90_fraction=float(np.nanquantile(random_values, 0.9)),
        physical_minus_random_median=float(physical_fraction - random_median),
        neighbor_top_rate=float(np.mean(top_hits)),
    )
