"""Exact matrix-free lifted mean-square operator for EXP-008A."""

from typing import Dict, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from linear_td_correlation import LinearTDConfig


def td_jacobian_distribution(
    mrp: Dict[str, np.ndarray],
    config: LinearTDConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Enumerate H=phi(s)(phi(s)-gamma phi(s'))' and its probability."""

    matrices = []
    weights = []
    for state in range(config.num_states):
        phi = mrp["features"][state]
        for following in range(config.num_states):
            difference = (
                phi - config.gamma * mrp["features"][following]
            )
            matrices.append(np.outer(phi, difference))
            weights.append(
                mrp["stationary"][state]
                * mrp["transition"][state, following]
            )
    return np.asarray(matrices), np.asarray(weights)


def delay_counts(delays: np.ndarray) -> np.ndarray:
    delays = np.asarray(delays, dtype=np.int64)
    return np.bincount(delays, minlength=int(np.max(delays)) + 1)


def build_shift_matrix(dimension: int, maximum_delay: int) -> np.ndarray:
    blocks = maximum_delay + 1
    size = dimension * blocks
    shift = np.zeros((size, size), dtype=float)
    shift[:dimension, :dimension] = np.eye(dimension)
    for lag in range(1, blocks):
        row = lag * dimension
        previous = (lag - 1) * dimension
        shift[row : row + dimension, previous : previous + dimension] = (
            np.eye(dimension)
        )
    return shift


def diagonal_moment_map(
    matrix: np.ndarray,
    jacobians: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "sij,jk,slk,s->il",
        jacobians,
        matrix,
        jacobians,
        weights,
        optimize=True,
    )


def build_operator_components(
    a_matrix: np.ndarray,
    jacobians: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dimension = a_matrix.shape[0]
    maximum_delay = len(counts) - 1
    shift = build_shift_matrix(dimension, maximum_delay)
    selector = np.zeros(
        (dimension, dimension * (maximum_delay + 1)), dtype=float
    )
    for lag, count in enumerate(counts):
        start = lag * dimension
        selector[:, start : start + dimension] = (
            int(count) * np.eye(dimension)
        )
    mean_lifted = np.zeros_like(shift)
    mean_lifted[:dimension, :] = a_matrix.dot(selector)
    moment_tensor = np.zeros(
        (dimension * dimension, dimension * dimension), dtype=float
    )
    for jacobian, weight in zip(jacobians, weights):
        moment_tensor += weight * np.kron(jacobian, jacobian)
    return shift, selector, mean_lifted, moment_tensor


def _apply_precomputed_operator(
    covariance: np.ndarray,
    eta: float,
    a_matrix: np.ndarray,
    counts: np.ndarray,
    rho: float,
    shift: np.ndarray,
    selector: np.ndarray,
    mean_lifted: np.ndarray,
    moment_tensor: np.ndarray,
) -> np.ndarray:
    dimension = a_matrix.shape[0]
    num_agents = int(np.sum(counts))
    result = (
        shift.dot(covariance).dot(shift.T)
        - eta
        / num_agents
        * (
            mean_lifted.dot(covariance).dot(shift.T)
            + shift.dot(covariance).dot(mean_lifted.T)
        )
    )
    all_blocks = selector.dot(covariance).dot(selector.T)
    diagonal_blocks = np.zeros(
        (dimension, dimension), dtype=covariance.dtype
    )
    for lag, count in enumerate(counts):
        if count == 0:
            continue
        block = slice(lag * dimension, (lag + 1) * dimension)
        diagonal_blocks += int(count) * covariance[block, block]

    def moment_map(matrix: np.ndarray) -> np.ndarray:
        vector = matrix.reshape(-1, order="F")
        return moment_tensor.dot(vector).reshape(
            (dimension, dimension), order="F"
        )

    diagonal_image = moment_map(diagonal_blocks)
    all_image = moment_map(all_blocks)
    off_blocks = all_blocks - diagonal_blocks
    second_top = (
        rho * all_image
        + (1.0 - rho) * diagonal_image
        + (1.0 - rho) * a_matrix.dot(off_blocks).dot(a_matrix.T)
    )
    result[:dimension, :dimension] += (
        eta * eta / (num_agents * num_agents) * second_top
    )
    return result


def apply_lifted_operator(
    covariance: np.ndarray,
    eta: float,
    a_matrix: np.ndarray,
    jacobians: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Apply P -> E[M P M'] without forming E[M kron M]."""

    components = build_operator_components(
        a_matrix, jacobians, weights, counts
    )
    return _apply_precomputed_operator(
        covariance,
        eta,
        a_matrix,
        counts,
        rho,
        *components,
    )


def dense_lifted_matrix(
    eta: float,
    a_matrix: np.ndarray,
    jacobians: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    rho: float,
) -> np.ndarray:
    dimension = a_matrix.shape[0] * len(counts)
    operator_dimension = dimension * dimension
    result = np.empty(
        (operator_dimension, operator_dimension), dtype=float
    )
    for index in range(operator_dimension):
        basis = np.zeros(operator_dimension, dtype=float)
        basis[index] = 1.0
        matrix = basis.reshape((dimension, dimension), order="F")
        image = apply_lifted_operator(
            matrix,
            eta,
            a_matrix,
            jacobians,
            weights,
            counts,
            rho,
        )
        result[:, index] = image.reshape(-1, order="F")
    return result


def lifted_spectral_radius(
    eta: float,
    a_matrix: np.ndarray,
    jacobians: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    rho: float,
    tolerance: float = 1e-10,
    max_iterations: int = 4000,
) -> Tuple[float, float, int]:
    """Return dominant radius, normalized residual, and ARPACK count proxy."""

    lifted_dimension = a_matrix.shape[0] * len(counts)
    operator_dimension = lifted_dimension * lifted_dimension
    components = build_operator_components(
        a_matrix, jacobians, weights, counts
    )

    def matvec(vector: np.ndarray) -> np.ndarray:
        matrix = vector.reshape(
            (lifted_dimension, lifted_dimension), order="F"
        )
        image = _apply_precomputed_operator(
            matrix,
            eta,
            a_matrix,
            counts,
            rho,
            *components,
        )
        return image.reshape(-1, order="F")

    operator = LinearOperator(
        (operator_dimension, operator_dimension),
        matvec=matvec,
        dtype=np.float64,
    )
    initial = np.eye(lifted_dimension).reshape(-1, order="F")
    initial /= np.linalg.norm(initial)
    values, vectors = eigs(
        operator,
        k=1,
        which="LM",
        v0=initial,
        tol=tolerance,
        maxiter=max_iterations,
        ncv=min(30, operator_dimension),
    )
    value = values[0]
    vector = vectors[:, 0]
    residual = np.linalg.norm(matvec(vector) - value * vector) / max(
        np.linalg.norm(vector), np.finfo(float).eps
    )
    return float(np.abs(value)), float(residual), int(operator_dimension)


def first_stability_boundary(
    mean_boundary: float,
    joint_step: float,
    a_matrix: np.ndarray,
    jacobians: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    rho: float,
    relative_tolerance: float = 2e-5,
) -> Dict[str, float]:
    """Locate the first loss of exact lifted mean-square stability."""

    cache: Dict[float, Tuple[float, float, int]] = {}

    def evaluate(value: float) -> Tuple[float, float, int]:
        key = float(value)
        if key not in cache:
            cache[key] = lifted_spectral_radius(
                key,
                a_matrix,
                jacobians,
                weights,
                counts,
                rho,
            )
        return cache[key]

    start = min(joint_step, mean_boundary) * 1e-3
    radius_start, _, _ = evaluate(start)
    for _ in range(12):
        if radius_start < 1.0:
            break
        start *= 0.5
        radius_start, _, _ = evaluate(start)
    if radius_start >= 1.0:
        raise RuntimeError("failed to locate the small-step stable region")
    upper_limit = mean_boundary * (1.0 + 1e-3)
    scan = np.geomspace(start, upper_limit, 24)
    lower = start
    upper = upper_limit
    found = False
    previous_eta = start
    previous_radius = radius_start
    for eta in scan[1:]:
        radius, _, _ = evaluate(float(eta))
        if previous_radius < 1.0 and radius >= 1.0:
            lower = previous_eta
            upper = float(eta)
            found = True
            break
        previous_eta = float(eta)
        previous_radius = radius
    if not found:
        raise RuntimeError("failed to bracket first mean-square boundary")
    while (
        upper - lower
        > relative_tolerance * max(1.0, 0.5 * (upper + lower))
    ):
        midpoint = 0.5 * (lower + upper)
        radius, _, _ = evaluate(midpoint)
        if radius < 1.0:
            lower = midpoint
        else:
            upper = midpoint
    boundary = 0.5 * (lower + upper)
    below_eta = lower * (1.0 - 5e-5)
    above_eta = upper * (1.0 + 5e-5)
    below_radius, below_residual, dimension = evaluate(below_eta)
    above_radius, above_residual, _ = evaluate(above_eta)
    joint_radius, joint_residual, _ = evaluate(joint_step)
    return {
        "exact_boundary": float(boundary),
        "below_eta": float(below_eta),
        "below_radius": float(below_radius),
        "below_residual": float(below_residual),
        "above_eta": float(above_eta),
        "above_radius": float(above_radius),
        "above_residual": float(above_residual),
        "joint_radius": float(joint_radius),
        "joint_residual": float(joint_residual),
        "operator_dimension": int(dimension),
        "operator_evaluations": int(len(cache)),
    }
