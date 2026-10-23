"""Exact two-regime Markov-jump mean-square tools for EXP-008B."""

from itertools import product
from math import ceil, log
from typing import Dict, Iterable, Tuple

import numpy as np

from exact_lifted_ms import dense_lifted_matrix, delay_counts
from td_delay_stability import build_mean_delay_transition


REGIME_PERSISTENCES: Tuple[float, ...] = (0.5, 0.9, 0.98)
AGENT_COUNTS_MARKOV: Tuple[int, ...] = (1, 2, 3)
MAX_DELAYS_MARKOV: Tuple[int, ...] = (0, 2)
CORRELATIONS_MARKOV: Tuple[float, ...] = (0.0, 0.9)
AGENT_COUNTS_EXPANDING: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)


def registered_td_model(
    gamma: float = 0.9,
) -> Dict[str, np.ndarray]:
    """Return the frozen two-feature, two-regime TD Jacobian model."""

    scale = 1.0 / np.sqrt(2.0)
    features = scale * np.asarray(((1.0, 1.0), (1.0, -1.0)))
    jacobians = []
    pairs = []
    for state in range(2):
        for following in range(2):
            phi = features[state]
            difference = phi - gamma * features[following]
            jacobians.append(np.outer(phi, difference))
            pairs.append((state, following))
    weights = np.asarray(
        (
            (0.05, 0.85, 0.05, 0.05),
            (0.05, 0.05, 0.85, 0.05),
        ),
        dtype=float,
    )
    matrices = np.asarray(jacobians)
    conditional_means = np.einsum(
        "zs,sij->zij", weights, matrices, optimize=True
    )
    return {
        "features": features,
        "pairs": np.asarray(pairs, dtype=np.int64),
        "jacobians": matrices,
        "weights": weights,
        "conditional_means": conditional_means,
        "stationary_mean": np.mean(conditional_means, axis=0),
        "gamma": np.asarray(gamma),
    }


def registered_expanding_td_model(
    gamma: float = 0.9,
) -> Dict[str, np.ndarray]:
    """Return the frozen scalar TD emission model for EXP-008C."""

    features = np.asarray(((1.0,), (2.0,)), dtype=float)
    jacobians = []
    pairs = []
    for state in range(2):
        for following in range(2):
            phi = features[state]
            difference = phi - gamma * features[following]
            jacobians.append(np.outer(phi, difference))
            pairs.append((state, following))
    weights = np.asarray(
        (
            (0.05, 0.85, 0.05, 0.05),
            (0.05, 0.05, 0.85, 0.05),
        ),
        dtype=float,
    )
    matrices = np.asarray(jacobians)
    conditional_means = np.einsum(
        "zs,sij->zij", weights, matrices, optimize=True
    )
    return {
        "features": features,
        "pairs": np.asarray(pairs, dtype=np.int64),
        "jacobians": matrices,
        "weights": weights,
        "conditional_means": conditional_means,
        "stationary_mean": np.mean(conditional_means, axis=0),
        "gamma": np.asarray(gamma),
    }


def registered_delays(num_agents: int, maximum_delay: int) -> np.ndarray:
    if num_agents not in AGENT_COUNTS_MARKOV:
        raise ValueError("unregistered agent count")
    if maximum_delay not in MAX_DELAYS_MARKOV:
        raise ValueError("unregistered maximum delay")
    if maximum_delay == 0:
        return np.zeros(num_agents, dtype=np.int64)
    profiles = {
        1: np.asarray((2,), dtype=np.int64),
        2: np.asarray((0, 2), dtype=np.int64),
        3: np.asarray((0, 1, 2), dtype=np.int64),
    }
    return profiles[num_agents].copy()


def homogeneous_delays(
    num_agents: int, delay: int
) -> np.ndarray:
    if num_agents not in AGENT_COUNTS_EXPANDING:
        raise ValueError("unregistered EXP-008C agent count")
    if delay not in MAX_DELAYS_MARKOV:
        raise ValueError("unregistered homogeneous delay")
    return np.full(num_agents, delay, dtype=np.int64)


def regime_transition(persistence: float) -> np.ndarray:
    if not 0.5 <= persistence < 1.0:
        raise ValueError("persistence must lie in [0.5, 1)")
    return np.asarray(
        (
            (persistence, 1.0 - persistence),
            (1.0 - persistence, persistence),
        ),
        dtype=float,
    )


def mixing_tv_after_gap(persistence: float, gap: int) -> float:
    if gap < 1:
        raise ValueError("decorrelation gap must be positive")
    eigenvalue = abs(2.0 * persistence - 1.0)
    return float(0.5 * eigenvalue ** int(gap))


def minimum_decorrelation_gap(
    persistence: float, target_delta: float
) -> int:
    if not 0.0 < target_delta < 0.5:
        raise ValueError("target delta must lie in (0, 0.5)")
    eigenvalue = abs(2.0 * persistence - 1.0)
    if eigenvalue == 0.0:
        return 1
    raw = log(2.0 * target_delta) / log(eigenvalue)
    gap = max(1, int(ceil(raw)))
    while mixing_tv_after_gap(persistence, gap) > target_delta:
        gap += 1
    while (
        gap > 1
        and mixing_tv_after_gap(persistence, gap - 1) <= target_delta
    ):
        gap -= 1
    return gap


def thinned_persistence(persistence: float, gap: int) -> float:
    eigenvalue = 2.0 * persistence - 1.0
    return float(0.5 * (1.0 + eigenvalue ** int(gap)))


def theorem_safe_step(
    model: Dict[str, np.ndarray],
    num_agents: int,
    rho: float,
    delays: np.ndarray,
    delta: float,
    safety_fraction: float = 1.0 - 1e-8,
) -> Dict[str, float]:
    """Return the scalar root of Theorem 3's sufficient condition."""

    _, curvature, monotonicity = aggregate_same_time_curvature(
        model, num_agents, rho
    )
    jacobians = model["jacobians"]
    lipschitz = float(
        max(np.linalg.norm(matrix, ord=2) for matrix in jacobians)
    )
    effective_monotonicity = monotonicity - 2.0 * lipschitz * delta
    if effective_monotonicity <= 0.0:
        raise ValueError("mixing error removes the monotonicity margin")
    effective_curvature = curvature + 2.0 * lipschitz ** 2 * delta
    rms_delay = float(np.sqrt(np.mean(np.asarray(delays, dtype=float) ** 2)))
    linear_coefficient = (
        effective_curvature + 4.0 * lipschitz ** 2 * rms_delay
    )
    cubic_coefficient = lipschitz ** 4 * rms_delay ** 2
    right_hand_side = 2.0 * effective_monotonicity

    def polynomial(eta: float) -> float:
        return (
            eta * linear_coefficient
            + eta ** 3 * cubic_coefficient
        )

    lower = 0.0
    upper = 1.0 / lipschitz
    if polynomial(upper) <= right_hand_side:
        root = upper
    else:
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            if polynomial(midpoint) < right_hand_side:
                lower = midpoint
            else:
                upper = midpoint
        root = 0.5 * (lower + upper)
    eta = safety_fraction * root
    contraction = (
        1.0
        - 2.0 * eta * effective_monotonicity
        + eta ** 2 * linear_coefficient
        + eta ** 4 * cubic_coefficient
    )
    return {
        "eta": float(eta),
        "root": float(root),
        "lipschitz": lipschitz,
        "monotonicity": monotonicity,
        "effective_monotonicity": effective_monotonicity,
        "curvature": curvature,
        "effective_curvature": effective_curvature,
        "rms_delay": rms_delay,
        "linear_coefficient": linear_coefficient,
        "cubic_coefficient": cubic_coefficient,
        "right_hand_side": right_hand_side,
        "polynomial_at_eta": float(polynomial(eta)),
        "contraction_coefficient": float(contraction),
    }


def sharp_theorem_steps(
    model: Dict[str, np.ndarray],
    num_agents: int,
    rho: float,
    delays: np.ndarray,
    delta: float,
) -> Dict[str, float]:
    """Return the sharp boundary and rate-optimal Theorem 3 steps."""

    coarse = theorem_safe_step(
        model,
        num_agents,
        rho,
        delays,
        delta,
    )
    effective_monotonicity = coarse["effective_monotonicity"]
    effective_curvature = coarse["effective_curvature"]
    lipschitz = coarse["lipschitz"]
    rms_delay = coarse["rms_delay"]

    def base(eta: float) -> float:
        return (
            1.0
            - 2.0 * eta * effective_monotonicity
            + eta * eta * effective_curvature
        )

    def factor(eta: float) -> float:
        return float(
            np.sqrt(max(base(eta), 0.0))
            + eta * eta * lipschitz * lipschitz * rms_delay
        )

    start = min(
        1e-6,
        effective_monotonicity
        / max(effective_curvature, np.finfo(float).eps)
        * 1e-3,
    )
    if factor(start) >= 1.0:
        raise RuntimeError("failed to locate sharp small-step region")
    lower = start
    upper = max(
        2.0 * effective_monotonicity / effective_curvature,
        2.0 * start,
    )
    while factor(upper) < 1.0:
        lower = upper
        upper *= 1.5
        if upper > 100.0:
            raise RuntimeError("failed to bracket sharp boundary")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if factor(midpoint) < 1.0:
            lower = midpoint
        else:
            upper = midpoint
    boundary = 0.5 * (lower + upper)
    safe_boundary_eta = (1.0 - 1e-8) * boundary

    left = 0.0
    right = boundary
    inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
    first = right - inverse_phi * (right - left)
    second = left + inverse_phi * (right - left)
    first_value = factor(first)
    second_value = factor(second)
    for _ in range(120):
        if first_value <= second_value:
            right = second
            second = first
            second_value = first_value
            first = right - inverse_phi * (right - left)
            first_value = factor(first)
        else:
            left = first
            first = second
            first_value = second_value
            second = left + inverse_phi * (right - left)
            second_value = factor(second)
    rate_eta = 0.5 * (left + right)
    rate_factor = factor(rate_eta)
    return {
        **coarse,
        "sharp_root": float(boundary),
        "sharp_safe_eta": float(safe_boundary_eta),
        "sharp_safe_factor": factor(safe_boundary_eta),
        "sharp_root_factor": factor(boundary),
        "rate_eta": float(rate_eta),
        "rate_factor": rate_factor,
        "rate_contraction_coefficient": rate_factor ** 2,
        "coarse_eta": coarse["eta"],
    }


def aggregate_same_time_curvature(
    model: Dict[str, np.ndarray],
    num_agents: int,
    rho: float,
) -> Tuple[np.ndarray, float, float]:
    """Return exact stationary E[Hbar' Hbar], K, and monotonicity."""

    jacobians = model["jacobians"]
    weights = model["weights"]
    means = model["conditional_means"]
    dimension = jacobians.shape[1]
    diagonal = np.zeros((dimension, dimension), dtype=float)
    off_diagonal = np.zeros((dimension, dimension), dtype=float)
    for mode in range(2):
        second = np.einsum(
            "s,sji,sjk->ik",
            weights[mode],
            jacobians,
            jacobians,
            optimize=True,
        )
        diagonal += 0.5 * second
        off_diagonal += 0.5 * (
            rho * second
            + (1.0 - rho) * means[mode].T.dot(means[mode])
        )
    aggregate = (
        diagonal / float(num_agents)
        + (num_agents - 1.0) / float(num_agents) * off_diagonal
    )
    curvature = float(np.max(np.linalg.eigvalsh(aggregate)))
    mean = model["stationary_mean"]
    monotonicity = float(
        np.min(np.linalg.eigvalsh(0.5 * (mean + mean.T)))
    )
    return aggregate, curvature, monotonicity


def _conditional_coefficients(
    model: Dict[str, np.ndarray],
    counts: np.ndarray,
    rho: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Return L_z(eta)=L0+eta L1+eta^2 L2 for each regime."""

    result = []
    for mode in range(2):
        mean = model["conditional_means"][mode]
        jacobians = model["jacobians"]
        weights = model["weights"][mode]
        at_zero = dense_lifted_matrix(
            0.0, mean, jacobians, weights, counts, rho
        )
        at_plus = dense_lifted_matrix(
            1.0, mean, jacobians, weights, counts, rho
        )
        at_minus = dense_lifted_matrix(
            -1.0, mean, jacobians, weights, counts, rho
        )
        linear = 0.5 * (at_plus - at_minus)
        quadratic = 0.5 * (at_plus + at_minus) - at_zero
        result.append((at_zero, linear, quadratic))
    return tuple(result)


def _combine_markov_coefficients(
    conditional: Tuple[Tuple[np.ndarray, ...], ...],
    transition: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Build blocks (next,current)=p(current,next)*L_current."""

    coefficients = []
    for degree in range(len(conditional[0])):
        rows = []
        for following in range(2):
            row = []
            for current in range(2):
                row.append(
                    transition[current, following]
                    * conditional[current][degree]
                )
            rows.append(row)
        coefficients.append(np.block(rows))
    return tuple(coefficients)


def covariance_operator_coefficients(
    model: Dict[str, np.ndarray],
    delays: np.ndarray,
    rho: float,
    persistence: float,
) -> Dict[str, Tuple[np.ndarray, ...]]:
    counts = delay_counts(delays)
    conditional = _conditional_coefficients(model, counts, rho)
    transition = regime_transition(persistence)
    markov = _combine_markov_coefficients(conditional, transition)
    iid = tuple(
        0.5 * (conditional[0][degree] + conditional[1][degree])
        for degree in range(3)
    )
    return {"markov": markov, "iid": iid}


def mean_operator_coefficients(
    model: Dict[str, np.ndarray],
    delays: np.ndarray,
    persistence: float,
) -> Dict[str, Tuple[np.ndarray, ...]]:
    conditional = []
    for mode in range(2):
        at_zero = build_mean_delay_transition(
            model["conditional_means"][mode], delays, 0.0
        )
        at_one = build_mean_delay_transition(
            model["conditional_means"][mode], delays, 1.0
        )
        conditional.append((at_zero, at_one - at_zero))
    conditional_tuple = tuple(conditional)
    transition = regime_transition(persistence)
    markov = _combine_markov_coefficients(
        conditional_tuple, transition
    )
    iid = tuple(
        0.5
        * (
            conditional_tuple[0][degree]
            + conditional_tuple[1][degree]
        )
        for degree in range(2)
    )
    return {"markov": markov, "iid": iid}


def polynomial_matrix(
    coefficients: Iterable[np.ndarray], eta: float
) -> np.ndarray:
    coefficients = tuple(coefficients)
    result = np.zeros_like(coefficients[0])
    power = 1.0
    for coefficient in coefficients:
        result += power * coefficient
        power *= eta
    return result


def spectral_radius_with_residual(
    coefficients: Iterable[np.ndarray], eta: float
) -> Tuple[float, float]:
    matrix = polynomial_matrix(coefficients, eta)
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmax(np.abs(values)))
    value = values[index]
    vector = vectors[:, index]
    residual = np.linalg.norm(matrix.dot(vector) - value * vector) / max(
        np.linalg.norm(vector), np.finfo(float).eps
    )
    return float(np.abs(value)), float(residual)


def first_stability_boundary(
    coefficients: Iterable[np.ndarray],
    initial_scale: float = 0.1,
    relative_tolerance: float = 2e-9,
) -> Dict[str, float]:
    """Locate the first loss of stability from the connected small-step set."""

    coefficients = tuple(coefficients)
    start = max(1e-8, float(initial_scale) * 1e-4)
    radius, _ = spectral_radius_with_residual(coefficients, start)
    for _ in range(30):
        if radius < 1.0 - 1e-11:
            break
        start *= 2.0
        radius, _ = spectral_radius_with_residual(coefficients, start)
    if radius >= 1.0 - 1e-11:
        raise RuntimeError("failed to locate a small-step stable point")

    lower = start
    lower_radius = radius
    upper = max(float(initial_scale), 2.0 * start)
    upper_radius, _ = spectral_radius_with_residual(coefficients, upper)
    evaluations = 2
    while upper_radius < 1.0:
        lower = upper
        lower_radius = upper_radius
        upper *= 1.5
        upper_radius, _ = spectral_radius_with_residual(
            coefficients, upper
        )
        evaluations += 1
        if upper > 100.0:
            raise RuntimeError("failed to bracket stability boundary")

    while upper - lower > relative_tolerance * max(
        1.0, 0.5 * (upper + lower)
    ):
        midpoint = 0.5 * (lower + upper)
        midpoint_radius, _ = spectral_radius_with_residual(
            coefficients, midpoint
        )
        evaluations += 1
        if midpoint_radius < 1.0:
            lower = midpoint
            lower_radius = midpoint_radius
        else:
            upper = midpoint
            upper_radius = midpoint_radius

    boundary = 0.5 * (lower + upper)
    below_eta = lower * (1.0 - 1e-7)
    above_eta = upper * (1.0 + 1e-7)
    below_radius, below_residual = spectral_radius_with_residual(
        coefficients, below_eta
    )
    above_radius, above_residual = spectral_radius_with_residual(
        coefficients, above_eta
    )
    return {
        "boundary": float(boundary),
        "below_eta": float(below_eta),
        "below_radius": float(below_radius),
        "below_residual": float(below_residual),
        "above_eta": float(above_eta),
        "above_radius": float(above_radius),
        "above_residual": float(above_residual),
        "operator_dimension": int(coefficients[0].shape[0]),
        "operator_evaluations": int(evaluations + 2),
    }


def direct_conditional_operator(
    model: Dict[str, np.ndarray],
    delays: np.ndarray,
    rho: float,
    mode: int,
    eta: float,
) -> np.ndarray:
    """Enumerate common draw, idiosyncratic draws, and masks independently."""

    jacobians = model["jacobians"]
    weights = model["weights"][mode]
    num_agents = len(delays)
    maximum_delay = int(np.max(delays))
    dimension = jacobians.shape[1]
    lifted_dimension = dimension * (maximum_delay + 1)
    shift = build_mean_delay_transition(
        np.zeros((dimension, dimension)), delays, 0.0
    )
    result = np.zeros(
        (lifted_dimension * lifted_dimension,) * 2, dtype=float
    )
    share_probability = np.sqrt(float(rho))
    choices = range(len(jacobians))
    for common in choices:
        for idiosyncratic in product(choices, repeat=num_agents):
            base_weight = weights[common]
            for choice in idiosyncratic:
                base_weight *= weights[choice]
            for masks in product((0, 1), repeat=num_agents):
                mask_weight = 1.0
                update = shift.copy()
                for agent, use_common in enumerate(masks):
                    mask_weight *= (
                        share_probability
                        if use_common
                        else 1.0 - share_probability
                    )
                    sample = (
                        common if use_common else idiosyncratic[agent]
                    )
                    column = int(delays[agent]) * dimension
                    update[
                        :dimension, column : column + dimension
                    ] -= eta * jacobians[sample] / num_agents
                result += (
                    base_weight
                    * mask_weight
                    * np.kron(update, update)
                )
    return result
