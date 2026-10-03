"""Tests for the exact lifted mean-square operator."""

import numpy as np

from exact_lifted_ms import (
    apply_lifted_operator,
    build_shift_matrix,
    dense_lifted_matrix,
    lifted_spectral_radius,
    td_jacobian_distribution,
)
from linear_td_correlation import LinearTDConfig, build_mrp


def test_shift_matrix_matches_lifted_state_definition() -> None:
    shift = build_shift_matrix(2, 2)
    state = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expected = np.asarray([1.0, 2.0, 1.0, 2.0, 3.0, 4.0])
    assert np.allclose(shift.dot(state), expected)


def test_matrix_free_operator_matches_direct_monte_carlo_expectation() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    jacobians, weights = td_jacobian_distribution(mrp, config)
    counts = np.asarray([2], dtype=np.int64)
    covariance = np.arange(16, dtype=float).reshape(4, 4) / 17.0
    eta = 0.03
    rho = 0.4
    analytic = apply_lifted_operator(
        covariance,
        eta,
        mrp["a_matrix"],
        jacobians,
        weights,
        counts,
        rho,
    )
    direct = np.zeros_like(covariance)
    for first, weight_first in zip(jacobians, weights):
        for second, weight_second in zip(jacobians, weights):
            independent_weight = weight_first * weight_second
            mean_matrix = np.eye(4) - 0.5 * eta * (first + second)
            direct += (
                (1.0 - rho)
                * independent_weight
                * mean_matrix.dot(covariance).dot(mean_matrix.T)
            )
        shared_matrix = np.eye(4) - eta * first
        direct += (
            rho
            * weight_first
            * shared_matrix.dot(covariance).dot(shared_matrix.T)
        )
    assert np.allclose(analytic, direct, atol=1e-12)


def test_dense_and_matrix_free_radii_agree() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    jacobians, weights = td_jacobian_distribution(mrp, config)
    counts = np.asarray([2], dtype=np.int64)
    eta = 0.04
    rho = 0.7
    dense = dense_lifted_matrix(
        eta,
        mrp["a_matrix"],
        jacobians,
        weights,
        counts,
        rho,
    )
    dense_radius = float(np.max(np.abs(np.linalg.eigvals(dense))))
    matrix_free_radius, residual, _ = lifted_spectral_radius(
        eta,
        mrp["a_matrix"],
        jacobians,
        weights,
        counts,
        rho,
    )
    assert abs(dense_radius - matrix_free_radius) <= 1e-9
    assert residual <= 1e-8

