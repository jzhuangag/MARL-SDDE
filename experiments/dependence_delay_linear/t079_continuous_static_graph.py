"""Exact-moment continuous collaboration graphs for the T-079 audit.

The dynamic policy minimizes the next-block personalized risk over each row
simplex.  The static comparator uses one row-stochastic matrix at every
decision block and is strengthened by deterministic multistart optimization
of its full-horizon exact risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from experiments.dependence_delay_linear.t068_safe_personalized_mixing import (
    MomentState,
    initial_moment_state,
)
from experiments.dependence_delay_linear.t070_nonstationary_graph import (
    _pre_block,
    recipient_actions,
    retarget_state,
)


@dataclass(frozen=True)
class ContinuousGraphBlockResult:
    state: MomentState
    weights: np.ndarray
    used_shadow: np.ndarray
    personalized_risk: np.ndarray
    shadow_risk: np.ndarray
    row_kkt_residuals: np.ndarray


@dataclass(frozen=True)
class ContinuousGraphTrajectory:
    auc_risk: float
    terminal_risk: float
    risk_path: np.ndarray
    weights_path: np.ndarray
    shadow_path: np.ndarray
    used_shadow: np.ndarray
    maximum_row_kkt_residual: float
    learning_transitions: int
    extra_probe_transitions: int
    message_units: int


@dataclass(frozen=True)
class StaticOptimizationResult:
    weights: np.ndarray
    auc_risk: float
    terminal_risk: float
    risk_path: np.ndarray
    successful_starts: int
    total_starts: int
    best_start_index: int
    row_sum_residual: float
    minimum_weight: float


def solve_simplex_quadratic(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Globally minimize ``w.T @ matrix @ w`` on a small simplex.

    Every nonempty support is enumerated.  On each face the equality-constrained
    quadratic has a closed form, and lower-dimensional supports include all
    boundary optima.  This is exact up to floating-point linear algebra for the
    four-agent audit.
    """

    hessian = np.asarray(matrix, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("quadratic matrix must be square")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("quadratic matrix must be finite")
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    hessian = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    dimension = hessian.shape[0]
    best_weights: np.ndarray | None = None
    best_value = float("inf")
    tolerance = 1e-10
    for size in range(1, dimension + 1):
        for support_tuple in combinations(range(dimension), size):
            support = np.asarray(support_tuple, dtype=int)
            face = hessian[np.ix_(support, support)]
            inverse = np.linalg.pinv(face, rcond=1e-12)
            ones = np.ones(size, dtype=float)
            denominator = float(ones @ inverse @ ones)
            if denominator <= tolerance:
                candidate_face = np.repeat(1.0 / size, size)
            else:
                candidate_face = inverse @ ones / denominator
            if np.any(candidate_face < -tolerance):
                continue
            candidate = np.zeros(dimension, dtype=float)
            candidate[support] = np.maximum(candidate_face, 0.0)
            total = float(np.sum(candidate))
            if total <= tolerance:
                continue
            candidate /= total
            value = float(candidate @ hessian @ candidate)
            if value < best_value - 1e-14:
                best_value = value
                best_weights = candidate
    if best_weights is None:
        raise RuntimeError("simplex quadratic did not produce a feasible point")
    return best_weights, simplex_quadratic_kkt_residual(hessian, best_weights)


def simplex_quadratic_kkt_residual(matrix: np.ndarray, weights: np.ndarray) -> float:
    hessian = np.asarray(matrix, dtype=float)
    value = np.asarray(weights, dtype=float)
    gradient = 2.0 * hessian @ value
    active = value > 1e-8
    if not np.any(active):
        return float("inf")
    multiplier = float(np.mean(gradient[active]))
    active_error = float(np.max(np.abs(gradient[active] - multiplier)))
    inactive_error = 0.0
    if np.any(~active):
        inactive_error = float(np.max(np.maximum(multiplier - gradient[~active], 0.0)))
    feasibility = max(abs(float(np.sum(value)) - 1.0), float(max(0.0, -np.min(value))))
    return max(active_error, inactive_error, feasibility)


def _recipient_error_moments(
    pre_mean: np.ndarray,
    pre_covariance: np.ndarray,
    targets: np.ndarray,
    recipient: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    agents = targets.size
    indices = np.asarray([
        recipient if donor == recipient else agents + donor
        for donor in range(agents)
    ], dtype=int)
    mean = pre_mean[indices].copy()
    mean += targets - targets[recipient]
    mean[recipient] = pre_mean[recipient]
    covariance = pre_covariance[np.ix_(indices, indices)]
    second_moment = covariance + np.outer(mean, mean)
    return mean, covariance, 0.5 * (second_moment + second_moment.T)


def _weight_row(agents: int, recipient: int, weights: np.ndarray) -> np.ndarray:
    row = np.zeros(3 * agents, dtype=float)
    for donor, weight in enumerate(weights):
        index = recipient if donor == recipient else agents + donor
        row[index] += float(weight)
    return row


def propagate_continuous_graph_block(
    state: MomentState,
    *,
    targets: Sequence[float],
    gain: float,
    curvature: float,
    local_steps: int,
    noise_scale: float,
    spatial_correlation: float,
    temporal_correlation: float,
    fixed_weights: np.ndarray | None = None,
    safe_oracle: bool = False,
) -> ContinuousGraphBlockResult:
    """Propagate one block under a fixed matrix or exact row-simplex oracle."""

    if (fixed_weights is None) == (not safe_oracle):
        raise ValueError("select exactly one of fixed_weights and safe_oracle")
    target = np.asarray(targets, dtype=float)
    agents = state.agents
    if target.shape != (agents,) or not np.all(np.isfinite(target)):
        raise ValueError("targets have incompatible shape")
    pre_mean, pre_covariance, pre_linear, pre_noise, innovation = _pre_block(
        state,
        gain=gain,
        curvature=curvature,
        local_steps=local_steps,
        noise_scale=noise_scale,
        spatial_correlation=spatial_correlation,
        temporal_correlation=temporal_correlation,
    )
    identity = np.eye(agents)
    if fixed_weights is not None:
        matrix = np.asarray(fixed_weights, dtype=float)
        if matrix.shape != (agents, agents):
            raise ValueError("fixed graph has incompatible shape")
        if np.min(matrix) < -1e-10 or not np.allclose(np.sum(matrix, axis=1), 1.0, atol=1e-9):
            raise ValueError("fixed graph must be row stochastic")
        selected_weights = np.maximum(matrix, 0.0)
        selected_weights /= np.sum(selected_weights, axis=1, keepdims=True)
    else:
        selected_weights = np.empty((agents, agents), dtype=float)
    shadow_rows = np.zeros((agents, 3 * agents), dtype=float)
    shadow_rows[:, 2 * agents:] = identity
    shadow_mean = shadow_rows @ pre_mean
    shadow_risk = np.square(shadow_mean) + np.einsum(
        "ij,jk,ik->i", shadow_rows, pre_covariance, shadow_rows
    )
    rows = np.empty_like(shadow_rows)
    shifts = np.zeros(agents, dtype=float)
    used_shadow = np.zeros(agents, dtype=bool)
    chosen_risk = np.empty(agents, dtype=float)
    residuals = np.zeros(agents, dtype=float)
    for recipient in range(agents):
        mean, _, second_moment = _recipient_error_moments(
            pre_mean, pre_covariance, target, recipient
        )
        if safe_oracle:
            weights, residual = solve_simplex_quadratic(second_moment)
            risk = float(weights @ second_moment @ weights)
            if risk > shadow_risk[recipient] + 1e-12:
                selected_weights[recipient] = identity[recipient]
                rows[recipient] = shadow_rows[recipient]
                chosen_risk[recipient] = shadow_risk[recipient]
                used_shadow[recipient] = True
                residuals[recipient] = residual
                continue
            selected_weights[recipient] = weights
            residuals[recipient] = residual
        weights = selected_weights[recipient]
        rows[recipient] = _weight_row(agents, recipient, weights)
        shifts[recipient] = float(weights @ (target - target[recipient]))
        shifts[recipient] -= float(weights[recipient] * (target[recipient] - target[recipient]))
        chosen_risk[recipient] = float(weights @ second_moment @ weights)
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
        block_map[destination, source] = identity
    shadow = slice((state.delay + 1) * agents, (state.delay + 2) * agents)
    block_map[shadow] = pre_linear[2 * agents:]
    block_noise[shadow] = pre_noise[2 * agents:]
    next_mean = block_map @ state.mean + block_shift
    next_covariance = (
        block_map @ state.covariance @ block_map.T
        + block_noise @ innovation @ block_noise.T
    )
    next_covariance = 0.5 * (next_covariance + next_covariance.T)
    return ContinuousGraphBlockResult(
        state=MomentState(
            mean=next_mean,
            covariance=next_covariance,
            agents=agents,
            delay=state.delay,
        ),
        weights=selected_weights,
        used_shadow=used_shadow,
        personalized_risk=np.square(next_mean[:agents])
        + np.diag(next_covariance[:agents, :agents]),
        shadow_risk=shadow_risk,
        row_kkt_residuals=residuals,
    )


def catalogue_graph_to_weights(
    graph_indices: Sequence[int], *, agents: int, alpha_grid: Sequence[float]
) -> np.ndarray:
    indices = np.asarray(graph_indices, dtype=int)
    if indices.shape != (agents,):
        raise ValueError("graph indices have incompatible shape")
    matrix = np.zeros((agents, agents), dtype=float)
    for recipient, action_index in enumerate(indices):
        catalogue = recipient_actions(agents, recipient, alpha_grid)
        action = catalogue[int(action_index)]
        matrix[recipient, recipient] = 1.0 - action.alpha
        matrix[recipient, action.donor] += action.alpha
    return matrix


def simulate_continuous_graph(
    *,
    target_schedule: np.ndarray,
    initial_parameter: float,
    delay: int,
    decision_blocks: Sequence[int],
    gain: float,
    curvature: float,
    local_steps: int,
    noise_scale: float,
    spatial_correlation: float,
    temporal_correlation: float,
    fixed_weights: np.ndarray | None = None,
    safe_dynamic_oracle: bool = False,
    fingerprint_message_units: int = 1,
    mixing_message_units: int = 1,
) -> ContinuousGraphTrajectory:
    """Propagate a fixed continuous graph or the exact myopic dynamic oracle."""

    if (fixed_weights is None) == (not safe_dynamic_oracle):
        raise ValueError("select exactly one graph mode")
    schedule = np.asarray(target_schedule, dtype=float)
    if schedule.ndim != 2 or schedule.shape[1] < 2:
        raise ValueError("target schedule must be blocks by agents")
    blocks, agents = schedule.shape
    initial = np.repeat(float(initial_parameter), agents)
    state = initial_moment_state(schedule[0], initial, delay)
    identity = np.eye(agents)
    decisions = set(int(block) for block in decision_blocks)
    previous = schedule[0]
    risks: list[float] = []
    shadow_risks: list[float] = []
    weights_path: list[np.ndarray] = []
    shadow_path: list[np.ndarray] = []
    residuals: list[float] = []
    messages = 0
    for block in range(blocks):
        target = schedule[block]
        if block > 0 and not np.array_equal(previous, target):
            state = retarget_state(state, previous, target)
        if block in decisions:
            result = propagate_continuous_graph_block(
                state,
                targets=target,
                gain=gain,
                curvature=curvature,
                local_steps=local_steps,
                noise_scale=noise_scale,
                spatial_correlation=spatial_correlation,
                temporal_correlation=temporal_correlation,
                fixed_weights=fixed_weights,
                safe_oracle=safe_dynamic_oracle,
            )
            if safe_dynamic_oracle:
                messages += int(fingerprint_message_units)
            accepted_nonlocal = np.any(
                np.max(np.abs(result.weights - identity), axis=1) > 1e-10
            ) and not bool(np.all(result.used_shadow))
            if accepted_nonlocal:
                messages += int(mixing_message_units)
            weights_path.append(result.weights.copy())
            shadow_path.append(result.used_shadow.copy())
            residuals.extend(result.row_kkt_residuals.tolist())
        else:
            result = propagate_continuous_graph_block(
                state,
                targets=target,
                gain=gain,
                curvature=curvature,
                local_steps=local_steps,
                noise_scale=noise_scale,
                spatial_correlation=spatial_correlation,
                temporal_correlation=temporal_correlation,
                fixed_weights=identity,
            )
        state = result.state
        risks.append(float(np.mean(result.personalized_risk)))
        shadow_risks.append(float(np.mean(result.shadow_risk)))
        previous = target
    risk_path = np.asarray(risks, dtype=float)
    return ContinuousGraphTrajectory(
        auc_risk=float(np.mean(risk_path)),
        terminal_risk=float(risk_path[-1]),
        risk_path=risk_path,
        weights_path=np.asarray(weights_path),
        shadow_path=np.asarray(shadow_path),
        used_shadow=np.asarray(shadow_path),
        maximum_row_kkt_residual=float(max(residuals, default=0.0)),
        learning_transitions=blocks * int(local_steps),
        extra_probe_transitions=0,
        message_units=messages,
    )


def deterministic_static_starts(
    agents: int, discrete_start: np.ndarray | None = None
) -> list[np.ndarray]:
    identity = np.eye(agents)
    uniform = np.repeat(1.0 / agents, agents * agents).reshape(agents, agents)
    starts = [identity, uniform]
    starts.extend((1.0 - alpha) * identity + alpha * uniform for alpha in (0.25, 0.5, 0.75))
    for shift in range(1, agents):
        starts.append(np.roll(identity, shift=shift, axis=1))
    if discrete_start is not None:
        start = np.asarray(discrete_start, dtype=float)
        starts.append(start)
        starts.append(0.9 * start + 0.1 * uniform)
    return [start.copy() for start in starts]


def optimize_static_graph(
    objective: Callable[[np.ndarray], ContinuousGraphTrajectory],
    *,
    agents: int,
    discrete_start: np.ndarray | None = None,
    maximum_iterations: int = 250,
) -> StaticOptimizationResult:
    """Deterministically strengthen a static graph with multistart SLSQP."""

    starts = deterministic_static_starts(agents, discrete_start)
    constraints = [
        {
            "type": "eq",
            "fun": lambda flattened, row=row: float(
                np.sum(flattened.reshape(agents, agents)[row]) - 1.0
            ),
        }
        for row in range(agents)
    ]
    bounds = [(0.0, 1.0)] * (agents * agents)
    best = None
    best_trajectory = None
    best_index = -1
    successful = 0
    for index, start in enumerate(starts):
        def scalar(flattened: np.ndarray) -> float:
            matrix = flattened.reshape(agents, agents)
            return objective(matrix).auc_risk

        result = minimize(
            scalar,
            start.reshape(-1),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maximum_iterations, "ftol": 1e-12, "disp": False},
        )
        matrix = result.x.reshape(agents, agents)
        matrix = np.maximum(matrix, 0.0)
        matrix /= np.sum(matrix, axis=1, keepdims=True)
        trajectory = objective(matrix)
        successful += int(bool(result.success))
        if best_trajectory is None or trajectory.auc_risk < best_trajectory.auc_risk:
            best = matrix
            best_trajectory = trajectory
            best_index = index
    if best is None or best_trajectory is None:
        raise RuntimeError("continuous static optimization failed")
    return StaticOptimizationResult(
        weights=best,
        auc_risk=best_trajectory.auc_risk,
        terminal_risk=best_trajectory.terminal_risk,
        risk_path=best_trajectory.risk_path,
        successful_starts=successful,
        total_starts=len(starts),
        best_start_index=best_index,
        row_sum_residual=float(np.max(np.abs(np.sum(best, axis=1) - 1.0))),
        minimum_weight=float(np.min(best)),
    )
