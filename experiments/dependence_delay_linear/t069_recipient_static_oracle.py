"""Batched exact moments for recipient-specific static mixing vectors."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Sequence

import numpy as np

from experiments.dependence_delay_linear.t068_safe_personalized_mixing import (
    block_noise_multiplier,
)


@dataclass(frozen=True)
class StaticComponents:
    alpha_vectors: np.ndarray
    mean_initial: np.ndarray
    mean_heterogeneity: np.ndarray
    variance_independent: np.ndarray
    variance_common: np.ndarray


def registered_alpha_vectors(alpha: Sequence[float], agents: int) -> np.ndarray:
    grid = np.asarray(alpha, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("alpha must be a nonempty finite vector")
    if np.any(grid < 0.0) or np.any(grid > 1.0):
        raise ValueError("alpha values must lie in [0, 1]")
    if int(agents) != agents or agents < 1:
        raise ValueError("agents must be a positive integer")
    return np.asarray(list(product(grid.tolist(), repeat=agents)), dtype=float)


def _batched_block_maps(
    *,
    alpha_vectors: np.ndarray,
    agents: int,
    delay: int,
    block_contraction: float,
    target_pattern: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policies = alpha_vectors.shape[0]
    dimension = (delay + 2) * agents
    pre_linear = np.zeros((3 * agents, dimension), dtype=float)
    pre_noise = np.zeros((3 * agents, agents), dtype=float)
    current = slice(0, agents)
    shadow = slice((delay + 1) * agents, (delay + 2) * agents)
    pre_linear[0:agents, current] = block_contraction * np.eye(agents)
    pre_noise[0:agents] = np.eye(agents)
    if delay == 0:
        pre_linear[agents : 2 * agents, current] = block_contraction * np.eye(agents)
        pre_noise[agents : 2 * agents] = np.eye(agents)
    else:
        donor = slice((delay - 1) * agents, delay * agents)
        pre_linear[agents : 2 * agents, donor] = np.eye(agents)
    pre_linear[2 * agents : 3 * agents, shadow] = block_contraction * np.eye(agents)
    pre_noise[2 * agents : 3 * agents] = np.eye(agents)

    rows = np.zeros((policies, agents, 3 * agents), dtype=float)
    for recipient in range(agents):
        rows[:, recipient, recipient] = 1.0 - alpha_vectors[:, recipient]
        rows[:, recipient, agents : 2 * agents] = (
            alpha_vectors[:, recipient, None] / agents
        )
    block_map = np.zeros((policies, dimension, dimension), dtype=float)
    block_noise = np.zeros((policies, dimension, agents), dtype=float)
    block_shift = np.zeros((policies, dimension), dtype=float)
    block_map[:, 0:agents] = rows @ pre_linear
    block_noise[:, 0:agents] = rows @ pre_noise
    block_shift[:, 0:agents] = alpha_vectors * (
        float(np.mean(target_pattern)) - target_pattern[None, :]
    )
    for lag in range(1, delay + 1):
        destination = slice(lag * agents, (lag + 1) * agents)
        source = slice((lag - 1) * agents, lag * agents)
        block_map[:, destination, source] = np.eye(agents)
    shadow_destination = slice((delay + 1) * agents, (delay + 2) * agents)
    block_map[:, shadow_destination] = pre_linear[2 * agents : 3 * agents]
    block_noise[:, shadow_destination] = pre_noise[2 * agents : 3 * agents]
    return block_map, block_noise, block_shift


def fixed_vector_components(
    config: dict[str, Any], delay: int, alpha_vectors: np.ndarray
) -> StaticComponents:
    model = config["model"]
    agents = int(model["agents"])
    vectors = np.asarray(alpha_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != agents:
        raise ValueError("alpha_vectors have incompatible shape")
    if int(delay) != delay or delay < 0:
        raise ValueError("delay must be a nonnegative integer")
    pattern = np.asarray(model["target_pattern"], dtype=float)
    one_step = 1.0 - model["gain"] * model["curvature"]
    block_contraction = one_step ** model["learning_steps_per_block"]
    mixed_map, mixed_noise, mixed_shift = _batched_block_maps(
        alpha_vectors=vectors,
        agents=agents,
        delay=delay,
        block_contraction=block_contraction,
        target_pattern=pattern,
    )
    zero = np.zeros((1, agents), dtype=float)
    local_map, local_noise, local_shift = _batched_block_maps(
        alpha_vectors=zero,
        agents=agents,
        delay=delay,
        block_contraction=block_contraction,
        target_pattern=pattern,
    )
    policies = vectors.shape[0]
    dimension = (delay + 2) * agents
    local_map = np.broadcast_to(local_map, (policies, dimension, dimension))
    local_noise = np.broadcast_to(local_noise, (policies, dimension, agents))
    local_shift = np.broadcast_to(local_shift, (policies, dimension))
    initial_basis = np.tile(np.ones(agents), delay + 2)
    heterogeneity_basis = np.tile(-pattern, delay + 2)
    mean_initial = np.repeat(initial_basis[None, :], policies, axis=0)
    mean_heterogeneity = np.repeat(heterogeneity_basis[None, :], policies, axis=0)
    covariance_independent = np.zeros((policies, dimension, dimension), dtype=float)
    covariance_common = np.zeros_like(covariance_independent)
    decision_blocks = set(model["decision_blocks"])
    for block in range(model["blocks"]):
        if block in decision_blocks:
            matrix = mixed_map
            noise = mixed_noise
            shift = mixed_shift
        else:
            matrix = local_map
            noise = local_noise
            shift = local_shift
        mean_initial = np.einsum("pij,pj->pi", matrix, mean_initial)
        mean_heterogeneity = np.einsum(
            "pij,pj->pi", matrix, mean_heterogeneity
        ) + shift
        covariance_independent = (
            matrix @ covariance_independent @ np.swapaxes(matrix, 1, 2)
            + noise @ np.swapaxes(noise, 1, 2)
        )
        common_direction = np.sum(noise, axis=2)
        covariance_common = (
            matrix @ covariance_common @ np.swapaxes(matrix, 1, 2)
            + common_direction[:, :, None] * common_direction[:, None, :]
        )
    return StaticComponents(
        alpha_vectors=vectors,
        mean_initial=mean_initial[:, :agents],
        mean_heterogeneity=mean_heterogeneity[:, :agents],
        variance_independent=np.diagonal(
            covariance_independent[:, :agents, :agents], axis1=1, axis2=2
        ),
        variance_common=np.diagonal(
            covariance_common[:, :agents, :agents], axis1=1, axis2=2
        ),
    )


def terminal_risks_from_components(
    config: dict[str, Any], scenario: dict[str, Any], components: StaticComponents
) -> np.ndarray:
    model = config["model"]
    one_step = 1.0 - model["gain"] * model["curvature"]
    multiplier = model["gain"] ** 2 * block_noise_multiplier(
        one_step,
        model["learning_steps_per_block"],
        scenario["temporal_correlation"],
    )
    mean = (
        scenario["initial_common_parameter"] * components.mean_initial
        + scenario["target_heterogeneity"] * components.mean_heterogeneity
    )
    variance = multiplier * scenario["noise_scale"] * (
        (1.0 - scenario["spatial_correlation"]) * components.variance_independent
        + scenario["spatial_correlation"] * components.variance_common
    )
    risks = np.mean(np.square(mean) + variance, axis=1)
    if not np.all(np.isfinite(risks)) or np.any(risks <= 0.0):
        raise ValueError("terminal risks must be positive and finite")
    return risks
