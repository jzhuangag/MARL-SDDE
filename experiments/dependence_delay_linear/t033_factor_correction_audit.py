"""Outcome-free algebraic audit of low-rank factor-corrected aggregation.

This module does not run a learning experiment.  It checks identifiability,
compares exact factor cancellation with the full-covariance BLUE/GLS oracle,
and audits pathwise cancellation after observations arrive with heterogeneous
delays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _as_loading_matrix(loadings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(loadings, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("loadings must be a vector or matrix")
    return matrix


def cancellation_identifiable(loadings: np.ndarray, tol: float = 1e-10) -> bool:
    """Return whether unbiased exact factor cancellation is feasible.

    Feasibility is equivalent to one not belonging to the column span of B.
    """

    matrix = _as_loading_matrix(loadings)
    ones = np.ones((matrix.shape[0], 1))
    return bool(
        np.linalg.matrix_rank(np.column_stack((ones, matrix)), tol=tol)
        > np.linalg.matrix_rank(matrix, tol=tol)
    )


def factor_cancel_weights(loadings: np.ndarray, idiosyncratic_variance: np.ndarray) -> np.ndarray:
    """Minimum-idiosyncratic-variance weights subject to exact cancellation."""

    matrix = _as_loading_matrix(loadings)
    variance = np.asarray(idiosyncratic_variance, dtype=float)
    if variance.shape != (matrix.shape[0],) or np.any(variance <= 0):
        raise ValueError("idiosyncratic variances must be positive and match agents")
    if not cancellation_identifiable(matrix):
        raise ValueError("factor cancellation is not identifiable")
    design = np.column_stack((np.ones(matrix.shape[0]), matrix))
    inverse_d_design = design / variance[:, None]
    gram = design.T @ inverse_d_design
    target = np.zeros(design.shape[1])
    target[0] = 1.0
    return inverse_d_design @ np.linalg.solve(gram, target)


def low_rank_gls_weights(
    loadings: np.ndarray,
    idiosyncratic_variance: np.ndarray,
    factor_covariance: np.ndarray,
) -> np.ndarray:
    """BLUE/GLS weights using a low-rank Woodbury solve."""

    matrix = _as_loading_matrix(loadings)
    variance = np.asarray(idiosyncratic_variance, dtype=float)
    omega = np.asarray(factor_covariance, dtype=float)
    if omega.ndim == 0:
        omega = omega.reshape(1, 1)
    if variance.shape != (matrix.shape[0],) or np.any(variance <= 0):
        raise ValueError("idiosyncratic variances must be positive and match agents")
    if omega.shape != (matrix.shape[1], matrix.shape[1]):
        raise ValueError("factor covariance has incompatible shape")
    if np.min(np.linalg.eigvalsh(omega)) <= 0:
        raise ValueError("factor covariance must be positive definite")
    inverse_d_b = matrix / variance[:, None]
    middle = np.linalg.inv(omega) + matrix.T @ inverse_d_b
    inverse_sigma_ones = 1.0 / variance - inverse_d_b @ np.linalg.solve(
        middle, matrix.T @ (1.0 / variance)
    )
    return inverse_sigma_ones / np.sum(inverse_sigma_ones)


def aggregation_risk(
    weights: np.ndarray,
    loadings: np.ndarray,
    idiosyncratic_variance: np.ndarray,
    factor_covariance: np.ndarray,
) -> float:
    matrix = _as_loading_matrix(loadings)
    weight = np.asarray(weights, dtype=float)
    variance = np.asarray(idiosyncratic_variance, dtype=float)
    omega = np.asarray(factor_covariance, dtype=float)
    if omega.ndim == 0:
        omega = omega.reshape(1, 1)
    exposure = matrix.T @ weight
    return float(np.sum(variance * weight**2) + exposure @ omega @ exposure)


def delay_expanded_loadings(loadings: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """Make one factor-loading block per distinct observation time.

    Pathwise cancellation of arbitrary factor values at distinct delays
    requires cancelling every block separately.
    """

    matrix = _as_loading_matrix(loadings)
    delay = np.asarray(delays)
    if delay.shape != (matrix.shape[0],):
        raise ValueError("delays must match agents")
    cohorts = np.unique(delay)
    expanded = np.zeros((matrix.shape[0], matrix.shape[1] * len(cohorts)))
    for cohort_index, cohort in enumerate(cohorts):
        rows = delay == cohort
        start = cohort_index * matrix.shape[1]
        expanded[rows, start : start + matrix.shape[1]] = matrix[rows]
    return expanded


def delayed_scalar_spectral_radius(alpha: float, weights: np.ndarray, delays: np.ndarray) -> float:
    """Spectral radius of e[t+1]=e[t]-alpha*sum_i w_i e[t-delay_i]."""

    weight = np.asarray(weights, dtype=float)
    delay = np.asarray(delays, dtype=int)
    if weight.shape != delay.shape or np.any(delay < 0):
        raise ValueError("weights and nonnegative integer delays must match")
    max_delay = int(np.max(delay))
    state = np.zeros((max_delay + 1, max_delay + 1))
    state[0, 0] = 1.0
    for value, lag in zip(weight, delay):
        state[0, lag] -= alpha * value
    if max_delay:
        state[1:, :-1] = np.eye(max_delay)
    return float(np.max(np.abs(np.linalg.eigvals(state))))


def run_audit() -> dict[str, object]:
    variance = np.ones(2)
    omega = np.array([[10.0]])
    unequal = np.array([0.5, 1.5])
    cancel = factor_cancel_weights(unequal, variance)
    gls = low_rank_gls_weights(unequal, variance, omega)
    equal_loading = np.ones(2)
    same_delay = np.array([0, 0])
    different_delay = np.array([0, 1])
    unstable_loading = np.array([1.0, 2.0])
    unstable_weights = factor_cancel_weights(unstable_loading, variance)

    return {
        "task": "T-033",
        "scientific_trajectories": 0,
        "known_factor": {
            "loadings": unequal.tolist(),
            "factor_cancel_weights": cancel.tolist(),
            "low_rank_gls_weights": gls.tolist(),
            "factor_cancel_risk": aggregation_risk(cancel, unequal, variance, omega),
            "low_rank_gls_risk": aggregation_risk(gls, unequal, variance, omega),
            "gls_weakly_dominates": aggregation_risk(gls, unequal, variance, omega)
            <= aggregation_risk(cancel, unequal, variance, omega) + 1e-12,
        },
        "identifiability": {
            "unequal_loading_feasible": cancellation_identifiable(unequal),
            "equal_loading_feasible": cancellation_identifiable(equal_loading),
        },
        "delay": {
            "same_time_pathwise_feasible": cancellation_identifiable(
                delay_expanded_loadings(unstable_loading, same_delay)
            ),
            "different_time_pathwise_feasible": cancellation_identifiable(
                delay_expanded_loadings(unstable_loading, different_delay)
            ),
            "same_time_spectral_radius_alpha_0_9": delayed_scalar_spectral_radius(
                0.9, unstable_weights, same_delay
            ),
            "different_time_spectral_radius_alpha_0_9": delayed_scalar_spectral_radius(
                0.9, unstable_weights, different_delay
            ),
            "weights": unstable_weights.tolist(),
        },
        "complexity": "O(m r^2 + r^3) arithmetic and O(m r) storage",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
