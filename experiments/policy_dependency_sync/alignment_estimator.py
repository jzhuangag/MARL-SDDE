"""Error accounting for sparse signed policy-dependency scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GradientErrorBudget:
    base_rms: float
    jvp_rms: float
    displacement_norm: float
    taylor_remainder: float
    sparse_tail: float
    off_policy_bias: float = 0.0

    @property
    def candidate_rms(self) -> float:
        values = (
            self.base_rms,
            self.jvp_rms,
            self.displacement_norm,
            self.taylor_remainder,
            self.sparse_tail,
            self.off_policy_bias,
        )
        if min(values) < 0.0:
            raise ValueError("gradient error-budget entries must be nonnegative")
        return float(
            self.base_rms
            + self.jvp_rms * self.displacement_norm
            + self.taylor_remainder
            + self.sparse_tail
            + self.off_policy_bias
        )


def geometric_mean_variance_factor(sample_count: int, correlation: float) -> float:
    """Finite-sample variance inflation for covariance bounded by ``rho^lag``."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must lie in [0, 1)")
    weighted_sum = sum(
        (1.0 - lag / sample_count) * correlation**lag
        for lag in range(1, sample_count)
    )
    return float(1.0 + 2.0 * weighted_sum)


def markov_mean_rms_bound(
    sample_count: int,
    correlation: float,
    stationary_coordinate_variance_sum: float,
    mean_bias_norm: float = 0.0,
) -> float:
    """RMS norm error of a vector Markov sample mean plus declared bias."""
    if stationary_coordinate_variance_sum < 0.0 or mean_bias_norm < 0.0:
        raise ValueError("variance and bias must be nonnegative")
    factor = geometric_mean_variance_factor(sample_count, correlation)
    stochastic = np.sqrt(
        stationary_coordinate_variance_sum * factor / sample_count
    )
    return float(mean_bias_norm + stochastic)


def optimized_quadratic_score(
    current_gradient: np.ndarray,
    candidate_gradient: np.ndarray,
    curvature: float,
    maximum_step: float,
) -> float:
    """Minimum smooth quadratic score over a scalar step interval."""
    current = np.asarray(current_gradient, dtype=float)
    candidate = np.asarray(candidate_gradient, dtype=float)
    if current.shape != candidate.shape:
        raise ValueError("gradient arrays must have matching shapes")
    if curvature <= 0.0 or maximum_step < 0.0:
        raise ValueError("curvature must be positive and maximum_step nonnegative")
    norm_squared = float(candidate @ candidate)
    if norm_squared == 0.0 or maximum_step == 0.0:
        return 0.0
    alignment = float(current @ candidate)
    step = min(max(alignment / (curvature * norm_squared), 0.0), maximum_step)
    return float(-step * alignment + 0.5 * curvature * step * step * norm_squared)


def optimized_score_error_bound(
    maximum_step: float,
    curvature: float,
    current_gradient_norm_bound: float,
    candidate_gradient_norm_bound: float,
    current_gradient_rms_error: float,
    candidate_gradient_rms_error: float,
) -> float:
    """Expected absolute error bound for an optimized quadratic drift score.

    The minimum-value error is bounded by the uniform fixed-step error because
    ``|min f - min g| <= sup |f-g|``.
    """
    values = (
        maximum_step,
        current_gradient_norm_bound,
        candidate_gradient_norm_bound,
        current_gradient_rms_error,
        candidate_gradient_rms_error,
    )
    if min(values) < 0.0 or curvature <= 0.0:
        raise ValueError("bounds must be nonnegative and curvature positive")
    alignment_error = (
        current_gradient_norm_bound * candidate_gradient_rms_error
        + candidate_gradient_norm_bound * current_gradient_rms_error
        + current_gradient_rms_error * candidate_gradient_rms_error
    )
    squared_norm_error = candidate_gradient_rms_error * (
        2.0 * candidate_gradient_norm_bound + candidate_gradient_rms_error
    )
    return float(
        maximum_step * alignment_error
        + 0.5 * curvature * maximum_step**2 * squared_norm_error
    )


def gaussian_max_norm_rms_bound(
    action_count: int,
    dimension: int,
    maximum_coordinate_standard_deviation: float,
    maximum_bias_norm: float = 0.0,
) -> float:
    """RMS bound for the maximum norm of finitely many Gaussian errors.

    Dependence among action errors is allowed.  A union bound on their
    individual Gaussian norm tails gives

    ``sqrt(E max ||Z_j||^2) <= s * sqrt(a^2 + 2 a sqrt(pi/2) + 2)``,

    where ``a=sqrt(d)+sqrt(2 log m)``.  A deterministic bias is then added by
    Minkowski's inequality.
    """
    if action_count <= 0 or dimension <= 0:
        raise ValueError("action_count and dimension must be positive")
    if maximum_coordinate_standard_deviation < 0.0 or maximum_bias_norm < 0.0:
        raise ValueError("standard deviation and bias must be nonnegative")
    threshold = np.sqrt(dimension) + np.sqrt(2.0 * np.log(action_count))
    second_moment_factor = np.sqrt(
        threshold * threshold
        + 2.0 * threshold * np.sqrt(np.pi / 2.0)
        + 2.0
    )
    return float(
        maximum_bias_norm
        + maximum_coordinate_standard_deviation * second_moment_factor
    )


def uniform_optimized_score_error_bound(
    maximum_step: float,
    curvature: float,
    current_gradient_norm_bound: float,
    candidate_gradient_norm_bound: float,
    current_gradient_rms_error: float,
    maximum_candidate_error_rms: float,
) -> float:
    """Expected supremum score error from joint RMS maximum bounds."""
    return optimized_score_error_bound(
        maximum_step=maximum_step,
        curvature=curvature,
        current_gradient_norm_bound=current_gradient_norm_bound,
        candidate_gradient_norm_bound=candidate_gradient_norm_bound,
        current_gradient_rms_error=current_gradient_rms_error,
        candidate_gradient_rms_error=maximum_candidate_error_rms,
    )


def sparse_candidate_gradient(
    base_gradient_samples: np.ndarray,
    local_jvp_samples: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    """Estimate one candidate gradient from a base mean and local JVP mean.

    ``local_jvp_samples`` has shape ``(samples, output_dim, input_dim)`` and
    contains only one declared dependency block.
    """
    base = np.asarray(base_gradient_samples, dtype=float)
    jvps = np.asarray(local_jvp_samples, dtype=float)
    shift = np.asarray(displacement, dtype=float)
    if base.ndim != 2 or jvps.ndim != 3 or shift.ndim != 1:
        raise ValueError("invalid base, JVP, or displacement rank")
    if base.shape[0] != jvps.shape[0]:
        raise ValueError("base and JVP sample counts must match")
    if base.shape[1] != jvps.shape[1] or jvps.shape[2] != shift.shape[0]:
        raise ValueError("base, JVP, and displacement dimensions must match")
    return np.asarray(base.mean(axis=0) + jvps.mean(axis=0) @ shift)


def split_fully_charged_rollouts(
    rollout_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split distinct rollout IDs into control/update halves."""
    ids = np.asarray(rollout_ids)
    if ids.ndim != 1 or ids.size < 2 or ids.size % 2:
        raise ValueError("an even one-dimensional rollout set is required")
    if np.unique(ids).size != ids.size:
        raise ValueError("rollout IDs must be distinct")
    midpoint = ids.size // 2
    return ids[:midpoint].copy(), ids[midpoint:].copy()
