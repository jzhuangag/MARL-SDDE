"""Exact moment tools for shadow-anchored personalized model mixing.

The module is intentionally limited to a scalar affine Markov-noise model.  It
is used to falsify the proposed transfer mechanism before sampled TD or neural
benchmarks are authorized.  All policies evolve a collaborative model and a
same-data local shadow model for every agent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class MomentState:
    """First two moments of delayed collaborative errors and shadow errors."""

    mean: np.ndarray
    covariance: np.ndarray
    agents: int
    delay: int


@dataclass(frozen=True)
class BlockResult:
    state: MomentState
    selected_alpha: np.ndarray
    used_shadow: np.ndarray
    personalized_risk: np.ndarray
    shadow_risk: np.ndarray


@dataclass(frozen=True)
class TrajectoryResult:
    state: MomentState
    terminal_risk: float
    terminal_shadow_risk: float
    normalized_auc: float
    selected_alpha: np.ndarray
    used_shadow: np.ndarray
    risk_path: np.ndarray
    shadow_risk_path: np.ndarray


def _as_finite_vector(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def spatial_noise_covariance(agents: int, scale: float, correlation: float) -> np.ndarray:
    if int(agents) != agents or agents < 1:
        raise ValueError("agents must be a positive integer")
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and nonnegative")
    if not math.isfinite(correlation) or not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must lie in [0, 1]")
    return scale * (
        correlation * np.ones((agents, agents))
        + (1.0 - correlation) * np.eye(agents)
    )


def block_noise_multiplier(
    contraction: float, local_steps: int, temporal_correlation: float
) -> float:
    """Variance multiplier for a geometrically correlated affine-noise block."""

    if not math.isfinite(contraction) or abs(contraction) >= 1.0:
        raise ValueError("contraction must be finite with absolute value below one")
    if int(local_steps) != local_steps or local_steps < 1:
        raise ValueError("local_steps must be a positive integer")
    if not math.isfinite(temporal_correlation) or not 0.0 <= temporal_correlation < 1.0:
        raise ValueError("temporal_correlation must lie in [0, 1)")
    weights = contraction ** np.arange(local_steps - 1, -1, -1, dtype=float)
    lags = np.abs(np.subtract.outer(np.arange(local_steps), np.arange(local_steps)))
    temporal = temporal_correlation**lags
    return float(weights @ temporal @ weights)


def initial_moment_state(
    targets: Sequence[float], initial_parameters: Sequence[float], delay: int
) -> MomentState:
    target = _as_finite_vector(targets, "targets")
    initial = _as_finite_vector(initial_parameters, "initial_parameters")
    if target.shape != initial.shape:
        raise ValueError("targets and initial_parameters must have the same shape")
    if int(delay) != delay or delay < 0:
        raise ValueError("delay must be a nonnegative integer")
    error = initial - target
    mean = np.concatenate([*[error.copy() for _ in range(delay + 1)], error.copy()])
    return MomentState(
        mean=mean,
        covariance=np.zeros((mean.size, mean.size), dtype=float),
        agents=target.size,
        delay=delay,
    )


def residual_transfer_quadratic(
    *, drift: np.ndarray, residual: np.ndarray, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact coefficients of the affine residual-energy change.

    For ``L(theta)=0.5*||F(theta)||^2`` and
    ``F(theta + D w)=F(theta)+A D w``, the change is
    ``g.T @ w + 0.5*w.T @ H @ w``.
    """

    matrix = np.asarray(drift, dtype=float)
    vector = np.asarray(residual, dtype=float)
    direction_matrix = np.asarray(directions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("drift must be square")
    if vector.shape != (matrix.shape[0],):
        raise ValueError("residual has incompatible dimension")
    if direction_matrix.ndim != 2 or direction_matrix.shape[0] != matrix.shape[1]:
        raise ValueError("directions have incompatible dimension")
    if not all(np.all(np.isfinite(item)) for item in (matrix, vector, direction_matrix)):
        raise ValueError("quadratic inputs must be finite")
    transformed = matrix @ direction_matrix
    linear = transformed.T @ vector
    hessian = transformed.T @ transformed
    return linear, 0.5 * (hessian + hessian.T)


def _risk(mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return np.square(mean) + np.diag(covariance)


def _pre_block_map(
    state: MomentState,
    *,
    gain: float,
    curvature: float,
    local_steps: int,
    noise_scale: float,
    spatial_correlation: float,
    temporal_correlation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("gain must be positive and finite")
    if not math.isfinite(curvature) or curvature <= 0.0:
        raise ValueError("curvature must be positive and finite")
    one_step = 1.0 - gain * curvature
    if not 0.0 < one_step < 1.0:
        raise ValueError("gain and curvature must define a contraction")
    agents = state.agents
    dimension = state.mean.size
    block_contraction = one_step**local_steps
    innovation = gain**2 * block_noise_multiplier(
        one_step, local_steps, temporal_correlation
    ) * spatial_noise_covariance(agents, noise_scale, spatial_correlation)

    # z = [current collaborative model after local learning,
    #      stale donor snapshot, local shadow after local learning].
    linear = np.zeros((3 * agents, dimension), dtype=float)
    noise_map = np.zeros((3 * agents, agents), dtype=float)
    current = slice(0, agents)
    shadow = slice((state.delay + 1) * agents, (state.delay + 2) * agents)
    linear[0:agents, current] = block_contraction * np.eye(agents)
    noise_map[0:agents] = np.eye(agents)
    if state.delay == 0:
        linear[agents : 2 * agents, current] = block_contraction * np.eye(agents)
        noise_map[agents : 2 * agents] = np.eye(agents)
    else:
        donor = slice((state.delay - 1) * agents, state.delay * agents)
        linear[agents : 2 * agents, donor] = np.eye(agents)
    linear[2 * agents : 3 * agents, shadow] = block_contraction * np.eye(agents)
    noise_map[2 * agents : 3 * agents] = np.eye(agents)
    mean = linear @ state.mean
    covariance = linear @ state.covariance @ linear.T + noise_map @ innovation @ noise_map.T
    covariance = 0.5 * (covariance + covariance.T)
    return mean, covariance, linear, noise_map, innovation


def _candidate_row(agents: int, recipient: int, alpha: float) -> np.ndarray:
    row = np.zeros(3 * agents, dtype=float)
    row[recipient] = 1.0 - alpha
    row[agents : 2 * agents] = alpha / agents
    return row


def _candidate_shift(targets: np.ndarray, recipient: int, alpha: float) -> float:
    return float(alpha * (np.mean(targets) - targets[recipient]))


def _scalar_affine_risk(
    row: np.ndarray, shift: float, mean: np.ndarray, covariance: np.ndarray
) -> float:
    candidate_mean = float(row @ mean + shift)
    candidate_variance = float(row @ covariance @ row)
    return candidate_mean**2 + candidate_variance


def propagate_personalized_block(
    state: MomentState,
    *,
    targets: Sequence[float],
    gain: float,
    curvature: float,
    local_steps: int,
    noise_scale: float,
    spatial_correlation: float,
    temporal_correlation: float,
    alpha: float | Sequence[float] | None = None,
    safe_alpha_grid: Sequence[float] | None = None,
    safety_slack: float = 0.0,
) -> BlockResult:
    """Propagate one block under fixed mixing or an exact shadow-safe oracle.

    Exactly one of ``alpha`` and ``safe_alpha_grid`` must be supplied.  The
    safe oracle minimizes recipient risk over predictable scalar mixing
    candidates and falls back to the same-data local shadow whenever needed.
    """

    target = _as_finite_vector(targets, "targets")
    if target.size != state.agents:
        raise ValueError("targets have incompatible agent count")
    if (alpha is None) == (safe_alpha_grid is None):
        raise ValueError("supply exactly one of alpha or safe_alpha_grid")
    if not math.isfinite(safety_slack) or safety_slack < 0.0:
        raise ValueError("safety_slack must be finite and nonnegative")
    pre_mean, pre_covariance, pre_linear, pre_noise, innovation = _pre_block_map(
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
    shadow_rows[:, 2 * agents : 3 * agents] = np.eye(agents)
    shadow_risk = np.asarray(
        [_scalar_affine_risk(row, 0.0, pre_mean, pre_covariance) for row in shadow_rows]
    )

    if safe_alpha_grid is not None:
        grid = np.asarray(safe_alpha_grid, dtype=float)
        if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
            raise ValueError("safe_alpha_grid must be a nonempty finite vector")
        if np.any(grid < 0.0) or np.any(grid > 1.0):
            raise ValueError("mixing weights must lie in [0, 1]")
        selected = np.zeros(agents, dtype=float)
        use_shadow = np.ones(agents, dtype=bool)
        rows = shadow_rows.copy()
        shifts = np.zeros(agents, dtype=float)
        chosen_risk = shadow_risk.copy()
        for recipient in range(agents):
            candidates = []
            for value in grid:
                row = _candidate_row(agents, recipient, float(value))
                shift = _candidate_shift(target, recipient, float(value))
                risk = _scalar_affine_risk(row, shift, pre_mean, pre_covariance)
                if risk <= shadow_risk[recipient] + safety_slack:
                    candidates.append((risk, float(value), row, shift))
            if candidates:
                risk, value, row, shift = min(candidates, key=lambda item: (item[0], item[1]))
                selected[recipient] = value
                use_shadow[recipient] = False
                rows[recipient] = row
                shifts[recipient] = shift
                chosen_risk[recipient] = risk
    else:
        values = np.asarray(alpha, dtype=float)
        if values.ndim == 0:
            values = np.repeat(values.item(), agents)
        if values.shape != (agents,) or not np.all(np.isfinite(values)):
            raise ValueError("alpha must be scalar or one finite value per agent")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("mixing weights must lie in [0, 1]")
        selected = values
        use_shadow = np.zeros(agents, dtype=bool)
        rows = np.vstack(
            [_candidate_row(agents, recipient, values[recipient]) for recipient in range(agents)]
        )
        shifts = np.asarray(
            [_candidate_shift(target, recipient, values[recipient]) for recipient in range(agents)]
        )
        chosen_risk = np.asarray(
            [
                _scalar_affine_risk(rows[i], shifts[i], pre_mean, pre_covariance)
                for i in range(agents)
            ]
        )

    new_dimension = state.mean.size
    block_map = np.zeros((new_dimension, new_dimension), dtype=float)
    block_noise = np.zeros((new_dimension, agents), dtype=float)
    block_shift = np.zeros(new_dimension, dtype=float)
    block_map[0:agents] = rows @ pre_linear
    block_noise[0:agents] = rows @ pre_noise
    block_shift[0:agents] = shifts
    for lag in range(1, state.delay + 1):
        destination = slice(lag * agents, (lag + 1) * agents)
        source = slice((lag - 1) * agents, lag * agents)
        block_map[destination, source] = np.eye(agents)
    shadow_destination = slice((state.delay + 1) * agents, (state.delay + 2) * agents)
    block_map[shadow_destination] = pre_linear[2 * agents : 3 * agents]
    block_noise[shadow_destination] = pre_noise[2 * agents : 3 * agents]

    mean = block_map @ state.mean + block_shift
    covariance = block_map @ state.covariance @ block_map.T + block_noise @ innovation @ block_noise.T
    covariance = 0.5 * (covariance + covariance.T)
    next_state = MomentState(mean=mean, covariance=covariance, agents=agents, delay=state.delay)
    next_shadow_mean = mean[shadow_destination]
    next_shadow_covariance = covariance[shadow_destination, shadow_destination]
    return BlockResult(
        state=next_state,
        selected_alpha=selected,
        used_shadow=use_shadow,
        personalized_risk=_risk(mean[:agents], covariance[:agents, :agents]),
        shadow_risk=_risk(next_shadow_mean, next_shadow_covariance),
    )


def simulate_personalized_mixing(
    *,
    targets: Sequence[float],
    initial_parameters: Sequence[float],
    delay: int,
    blocks: int,
    gain: float,
    curvature: float,
    local_steps: int,
    noise_scale: float,
    spatial_correlation: float,
    temporal_correlation: float,
    policy: Literal["fixed", "two_phase", "safe_oracle"],
    alpha: float | None = None,
    early_alpha: float | None = None,
    late_alpha: float | None = None,
    switch_block: int | None = None,
    safe_alpha_grid: Sequence[float] | None = None,
) -> TrajectoryResult:
    if int(blocks) != blocks or blocks < 1:
        raise ValueError("blocks must be a positive integer")
    state = initial_moment_state(targets, initial_parameters, delay)
    selected = []
    fallbacks = []
    risks = []
    shadow_risks = []
    for block in range(blocks):
        if policy == "fixed":
            result = propagate_personalized_block(
                state,
                targets=targets,
                gain=gain,
                curvature=curvature,
                local_steps=local_steps,
                noise_scale=noise_scale,
                spatial_correlation=spatial_correlation,
                temporal_correlation=temporal_correlation,
                alpha=alpha,
            )
        elif policy == "two_phase":
            if early_alpha is None or late_alpha is None or switch_block is None:
                raise ValueError("two_phase requires both gains and switch_block")
            value = early_alpha if block < switch_block else late_alpha
            result = propagate_personalized_block(
                state,
                targets=targets,
                gain=gain,
                curvature=curvature,
                local_steps=local_steps,
                noise_scale=noise_scale,
                spatial_correlation=spatial_correlation,
                temporal_correlation=temporal_correlation,
                alpha=value,
            )
        elif policy == "safe_oracle":
            result = propagate_personalized_block(
                state,
                targets=targets,
                gain=gain,
                curvature=curvature,
                local_steps=local_steps,
                noise_scale=noise_scale,
                spatial_correlation=spatial_correlation,
                temporal_correlation=temporal_correlation,
                safe_alpha_grid=safe_alpha_grid,
            )
        else:
            raise ValueError(f"unknown policy: {policy}")
        state = result.state
        selected.append(result.selected_alpha)
        fallbacks.append(result.used_shadow)
        risks.append(result.personalized_risk)
        shadow_risks.append(result.shadow_risk)
    risk_path = np.asarray(risks)
    shadow_path = np.asarray(shadow_risks)
    initial_error = _as_finite_vector(initial_parameters, "initial_parameters") - _as_finite_vector(
        targets, "targets"
    )
    normalization = max(float(np.mean(np.square(initial_error))), 1e-12)
    return TrajectoryResult(
        state=state,
        terminal_risk=float(np.mean(risk_path[-1])),
        terminal_shadow_risk=float(np.mean(shadow_path[-1])),
        normalized_auc=float(np.mean(risk_path) / normalization),
        selected_alpha=np.asarray(selected),
        used_shadow=np.asarray(fallbacks),
        risk_path=risk_path,
        shadow_risk_path=shadow_path,
    )
