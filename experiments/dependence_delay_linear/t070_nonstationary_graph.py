"""Exact moments for delayed collaboration graphs with changing targets.

This module is an outcome-free feasibility tool.  It propagates the first two
moments of scalar affine Markov-TD errors exactly.  Each recipient either keeps
its locally updated model or mixes it with one delayed donor.  A same-data local
shadow is propagated in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Sequence

import numpy as np

from experiments.dependence_delay_linear.t068_safe_personalized_mixing import (
    MomentState,
    block_noise_multiplier,
    initial_moment_state,
    spatial_noise_covariance,
)


@dataclass(frozen=True)
class RecipientAction:
    donor: int
    alpha: float


@dataclass(frozen=True)
class GraphBlockResult:
    state: MomentState
    action_indices: np.ndarray
    used_shadow: np.ndarray
    personalized_risk: np.ndarray
    shadow_risk: np.ndarray


@dataclass(frozen=True)
class StaticGraphComponents:
    graph_indices: np.ndarray
    mean_initial_path: np.ndarray
    mean_target_path: np.ndarray
    variance_independent_path: np.ndarray
    variance_common_path: np.ndarray


def recipient_actions(agents: int, recipient: int, alpha: Sequence[float]) -> list[RecipientAction]:
    """Return local plus every registered single-donor action."""

    if int(agents) != agents or agents < 2:
        raise ValueError("agents must be an integer of at least two")
    if not 0 <= recipient < agents:
        raise ValueError("recipient lies outside the agent set")
    grid = np.asarray(alpha, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("alpha must be a nonempty finite vector")
    if np.any(grid <= 0.0) or np.any(grid > 1.0):
        raise ValueError("nonlocal alpha values must lie in (0, 1]")
    actions = [RecipientAction(donor=recipient, alpha=0.0)]
    for donor in range(agents):
        if donor == recipient:
            continue
        actions.extend(RecipientAction(donor=donor, alpha=float(value)) for value in grid)
    return actions


def registered_static_graphs(actions_per_recipient: int, agents: int) -> np.ndarray:
    if int(actions_per_recipient) != actions_per_recipient or actions_per_recipient < 1:
        raise ValueError("actions_per_recipient must be positive")
    if int(agents) != agents or agents < 1:
        raise ValueError("agents must be positive")
    return np.asarray(
        list(product(range(actions_per_recipient), repeat=agents)), dtype=np.int16
    )


def retarget_state(
    state: MomentState, old_targets: Sequence[float], new_targets: Sequence[float]
) -> MomentState:
    """Express every stored parameter snapshot relative to new targets."""

    old = np.asarray(old_targets, dtype=float)
    new = np.asarray(new_targets, dtype=float)
    if old.shape != (state.agents,) or new.shape != (state.agents,):
        raise ValueError("target vectors have incompatible shape")
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError("targets must be finite")
    shift = np.tile(old - new, state.delay + 2)
    return MomentState(
        mean=state.mean + shift,
        covariance=state.covariance.copy(),
        agents=state.agents,
        delay=state.delay,
    )


def _pre_block(
    state: MomentState,
    *,
    gain: float,
    curvature: float,
    local_steps: int,
    noise_scale: float,
    spatial_correlation: float,
    temporal_correlation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    one_step = 1.0 - gain * curvature
    if not 0.0 < one_step < 1.0:
        raise ValueError("gain and curvature must define a contraction")
    if int(local_steps) != local_steps or local_steps < 1:
        raise ValueError("local_steps must be positive")
    agents = state.agents
    dimension = state.mean.size
    contraction = one_step**local_steps
    innovation = gain**2 * block_noise_multiplier(
        one_step, local_steps, temporal_correlation
    ) * spatial_noise_covariance(agents, noise_scale, spatial_correlation)
    linear = np.zeros((3 * agents, dimension), dtype=float)
    noise = np.zeros((3 * agents, agents), dtype=float)
    current = slice(0, agents)
    shadow = slice((state.delay + 1) * agents, (state.delay + 2) * agents)
    linear[:agents, current] = contraction * np.eye(agents)
    noise[:agents] = np.eye(agents)
    if state.delay == 0:
        linear[agents : 2 * agents, current] = contraction * np.eye(agents)
        noise[agents : 2 * agents] = np.eye(agents)
    else:
        donor = slice((state.delay - 1) * agents, state.delay * agents)
        linear[agents : 2 * agents, donor] = np.eye(agents)
    linear[2 * agents :, shadow] = contraction * np.eye(agents)
    noise[2 * agents :] = np.eye(agents)
    mean = linear @ state.mean
    covariance = linear @ state.covariance @ linear.T + noise @ innovation @ noise.T
    return mean, 0.5 * (covariance + covariance.T), linear, noise, innovation


def _action_row(
    agents: int, recipient: int, action: RecipientAction
) -> np.ndarray:
    row = np.zeros(3 * agents, dtype=float)
    row[recipient] = 1.0 - action.alpha
    row[agents + action.donor] = action.alpha
    return row


def _action_shift(targets: np.ndarray, recipient: int, action: RecipientAction) -> float:
    return float(action.alpha * (targets[action.donor] - targets[recipient]))


def _row_risk(row: np.ndarray, shift: float, mean: np.ndarray, covariance: np.ndarray) -> float:
    candidate_mean = float(row @ mean + shift)
    return candidate_mean**2 + float(row @ covariance @ row)


def propagate_graph_block(
    state: MomentState,
    *,
    targets: Sequence[float],
    gain: float,
    curvature: float,
    local_steps: int,
    noise_scale: float,
    spatial_correlation: float,
    temporal_correlation: float,
    alpha_grid: Sequence[float],
    fixed_action_indices: Sequence[int] | None = None,
    safe_oracle: bool = False,
) -> GraphBlockResult:
    """Propagate one fixed graph block or one exact shadow-safe oracle block."""

    if (fixed_action_indices is None) == (not safe_oracle):
        raise ValueError("select exactly one of a fixed graph or safe_oracle")
    target = np.asarray(targets, dtype=float)
    if target.shape != (state.agents,) or not np.all(np.isfinite(target)):
        raise ValueError("targets have incompatible shape")
    pre_mean, pre_cov, pre_linear, pre_noise, innovation = _pre_block(
        state,
        gain=gain,
        curvature=curvature,
        local_steps=local_steps,
        noise_scale=noise_scale,
        spatial_correlation=spatial_correlation,
        temporal_correlation=temporal_correlation,
    )
    agents = state.agents
    shadow_rows = np.zeros((agents, 3 * agents), dtype=float)
    shadow_rows[:, 2 * agents :] = np.eye(agents)
    shadow_risk = np.asarray(
        [_row_risk(row, 0.0, pre_mean, pre_cov) for row in shadow_rows]
    )
    rows = shadow_rows.copy()
    shifts = np.zeros(agents, dtype=float)
    selected = np.zeros(agents, dtype=np.int16)
    used_shadow = np.zeros(agents, dtype=bool)
    chosen_risk = shadow_risk.copy()
    for recipient in range(agents):
        catalogue = recipient_actions(agents, recipient, alpha_grid)
        if safe_oracle:
            feasible = []
            for index, action in enumerate(catalogue):
                row = _action_row(agents, recipient, action)
                shift = _action_shift(target, recipient, action)
                risk = _row_risk(row, shift, pre_mean, pre_cov)
                if risk <= shadow_risk[recipient] + 1e-12:
                    feasible.append((risk, index, row, shift))
            risk, index, row, shift = min(feasible, key=lambda item: (item[0], item[1]))
            used_shadow[recipient] = index == 0
        else:
            indices = np.asarray(fixed_action_indices, dtype=int)
            if indices.shape != (agents,):
                raise ValueError("fixed_action_indices have incompatible shape")
            index = int(indices[recipient])
            if not 0 <= index < len(catalogue):
                raise ValueError("fixed action index lies outside catalogue")
            action = catalogue[index]
            row = _action_row(agents, recipient, action)
            shift = _action_shift(target, recipient, action)
            risk = _row_risk(row, shift, pre_mean, pre_cov)
        rows[recipient] = row
        shifts[recipient] = shift
        selected[recipient] = index
        chosen_risk[recipient] = risk

    dimension = state.mean.size
    block_map = np.zeros((dimension, dimension), dtype=float)
    block_noise = np.zeros((dimension, agents), dtype=float)
    block_shift = np.zeros(dimension, dtype=float)
    block_map[:agents] = rows @ pre_linear
    block_noise[:agents] = rows @ pre_noise
    block_shift[:agents] = shifts
    for lag in range(1, state.delay + 1):
        destination = slice(lag * agents, (lag + 1) * agents)
        source = slice((lag - 1) * agents, lag * agents)
        block_map[destination, source] = np.eye(agents)
    shadow = slice((state.delay + 1) * agents, (state.delay + 2) * agents)
    block_map[shadow] = pre_linear[2 * agents :]
    block_noise[shadow] = pre_noise[2 * agents :]
    mean = block_map @ state.mean + block_shift
    covariance = block_map @ state.covariance @ block_map.T + block_noise @ innovation @ block_noise.T
    covariance = 0.5 * (covariance + covariance.T)
    return GraphBlockResult(
        state=MomentState(mean=mean, covariance=covariance, agents=agents, delay=state.delay),
        action_indices=selected,
        used_shadow=used_shadow,
        personalized_risk=np.square(mean[:agents]) + np.diag(covariance[:agents, :agents]),
        shadow_risk=np.square(mean[shadow]) + np.diag(covariance[shadow, shadow]),
    )


def _batched_maps(
    *,
    graphs: np.ndarray,
    agents: int,
    delay: int,
    block_contraction: float,
    target_pattern: np.ndarray,
    alpha_grid: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policies = graphs.shape[0]
    dimension = (delay + 2) * agents
    pre_linear = np.zeros((3 * agents, dimension), dtype=float)
    pre_noise = np.zeros((3 * agents, agents), dtype=float)
    pre_linear[:agents, :agents] = block_contraction * np.eye(agents)
    pre_noise[:agents] = np.eye(agents)
    if delay == 0:
        pre_linear[agents : 2 * agents, :agents] = block_contraction * np.eye(agents)
        pre_noise[agents : 2 * agents] = np.eye(agents)
    else:
        pre_linear[agents : 2 * agents, (delay - 1) * agents : delay * agents] = np.eye(agents)
    shadow = slice((delay + 1) * agents, (delay + 2) * agents)
    pre_linear[2 * agents :, shadow] = block_contraction * np.eye(agents)
    pre_noise[2 * agents :] = np.eye(agents)
    rows = np.zeros((policies, agents, 3 * agents), dtype=float)
    shifts = np.zeros((policies, agents), dtype=float)
    for recipient in range(agents):
        catalogue = recipient_actions(agents, recipient, alpha_grid)
        for action_index, action in enumerate(catalogue):
            mask = graphs[:, recipient] == action_index
            rows[mask, recipient, recipient] = 1.0 - action.alpha
            rows[mask, recipient, agents + action.donor] = action.alpha
            shifts[mask, recipient] = action.alpha * (
                target_pattern[action.donor] - target_pattern[recipient]
            )
    block_map = np.zeros((policies, dimension, dimension), dtype=float)
    block_noise = np.zeros((policies, dimension, agents), dtype=float)
    block_shift = np.zeros((policies, dimension), dtype=float)
    block_map[:, :agents] = rows @ pre_linear
    block_noise[:, :agents] = rows @ pre_noise
    block_shift[:, :agents] = shifts
    for lag in range(1, delay + 1):
        block_map[:, lag * agents : (lag + 1) * agents, (lag - 1) * agents : lag * agents] = np.eye(agents)
    block_map[:, shadow] = pre_linear[2 * agents :]
    block_noise[:, shadow] = pre_noise[2 * agents :]
    return block_map, block_noise, block_shift


def static_graph_components(
    *,
    graphs: np.ndarray,
    agents: int,
    delay: int,
    blocks: int,
    decision_blocks: Sequence[int],
    gain: float,
    curvature: float,
    local_steps: int,
    alpha_grid: Sequence[float],
    unit_target_schedule: np.ndarray,
) -> StaticGraphComponents:
    """Precompute exact paths shared by a grid of scalar scenarios."""

    graph_array = np.asarray(graphs, dtype=np.int16)
    schedule = np.asarray(unit_target_schedule, dtype=float)
    if graph_array.ndim != 2 or graph_array.shape[1] != agents:
        raise ValueError("graphs have incompatible shape")
    if schedule.shape != (blocks, agents):
        raise ValueError("unit_target_schedule has incompatible shape")
    policies = graph_array.shape[0]
    dimension = (delay + 2) * agents
    one_step = 1.0 - gain * curvature
    contraction = one_step**local_steps
    zero_graph = np.zeros((1, agents), dtype=np.int16)
    local_map, local_noise, local_shift = _batched_maps(
        graphs=zero_graph,
        agents=agents,
        delay=delay,
        block_contraction=contraction,
        target_pattern=schedule[0],
        alpha_grid=alpha_grid,
    )
    local_map = np.broadcast_to(local_map, (policies, dimension, dimension))
    local_noise = np.broadcast_to(local_noise, (policies, dimension, agents))
    local_shift = np.broadcast_to(local_shift, (policies, dimension))
    mean_initial = np.tile(np.ones(agents), delay + 2)[None, :]
    mean_initial = np.repeat(mean_initial, policies, axis=0)
    mean_target = np.tile(-schedule[0], delay + 2)[None, :]
    mean_target = np.repeat(mean_target, policies, axis=0)
    covariance_independent = np.zeros((policies, dimension, dimension), dtype=float)
    covariance_common = np.zeros_like(covariance_independent)
    initial_path = []
    target_path = []
    independent_path = []
    common_path = []
    decisions = set(decision_blocks)
    previous = schedule[0]
    for block in range(blocks):
        current = schedule[block]
        if block > 0 and not np.array_equal(current, previous):
            mean_target += np.tile(previous - current, delay + 2)[None, :]
        if block in decisions:
            matrix, noise, shift = _batched_maps(
                graphs=graph_array,
                agents=agents,
                delay=delay,
                block_contraction=contraction,
                target_pattern=current,
                alpha_grid=alpha_grid,
            )
        else:
            matrix, noise, shift = local_map, local_noise, local_shift
        mean_initial = np.einsum("pij,pj->pi", matrix, mean_initial)
        mean_target = np.einsum("pij,pj->pi", matrix, mean_target) + shift
        covariance_independent = matrix @ covariance_independent @ np.swapaxes(matrix, 1, 2) + noise @ np.swapaxes(noise, 1, 2)
        common_direction = np.sum(noise, axis=2)
        covariance_common = matrix @ covariance_common @ np.swapaxes(matrix, 1, 2) + common_direction[:, :, None] * common_direction[:, None, :]
        initial_path.append(mean_initial[:, :agents].copy())
        target_path.append(mean_target[:, :agents].copy())
        independent_path.append(np.diagonal(covariance_independent[:, :agents, :agents], axis1=1, axis2=2).copy())
        common_path.append(np.diagonal(covariance_common[:, :agents, :agents], axis1=1, axis2=2).copy())
        previous = current
    return StaticGraphComponents(
        graph_indices=graph_array,
        mean_initial_path=np.stack(initial_path, axis=1),
        mean_target_path=np.stack(target_path, axis=1),
        variance_independent_path=np.stack(independent_path, axis=1),
        variance_common_path=np.stack(common_path, axis=1),
    )


def static_graph_risks(
    components: StaticGraphComponents,
    *,
    initial_parameter: float,
    target_scale: float,
    gain: float,
    curvature: float,
    local_steps: int,
    noise_scale: float,
    spatial_correlation: float,
    temporal_correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    one_step = 1.0 - gain * curvature
    multiplier = gain**2 * block_noise_multiplier(
        one_step, local_steps, temporal_correlation
    )
    mean = initial_parameter * components.mean_initial_path + target_scale * components.mean_target_path
    variance = multiplier * noise_scale * (
        (1.0 - spatial_correlation) * components.variance_independent_path
        + spatial_correlation * components.variance_common_path
    )
    path = np.mean(np.square(mean) + variance, axis=2)
    auc = np.mean(path, axis=1)
    terminal = path[:, -1]
    if np.any(auc <= 0.0) or not np.all(np.isfinite(auc)):
        raise ValueError("static risks must be positive and finite")
    return auc, terminal
