from __future__ import annotations

import numpy as np
import pytest

from experiments.policy_update_backpressure.transport_theory import (
    PendingTransportDebt,
    TransportCertificate,
    audit,
    certified_gain,
    equicoupled_matrix,
    gain_optimal_step,
    lyapunov_optimal_step,
    public_curvature_bound,
    quadratic_gradient,
    quadratic_hessian,
    raw_staleness_radius,
    transport_lyapunov_drift_bound,
    transport_drift_second_derivative,
    transported_gradient,
    transported_radius,
)


def test_quadratic_transport_is_exact() -> None:
    matrix = equicoupled_matrix(3, 0.85)
    birth = np.array([0.8, -0.4, 0.2])
    displacement = np.array([-0.2, 0.1, 0.05])
    transported = transported_gradient(
        quadratic_gradient(birth, matrix),
        quadratic_hessian(matrix),
        displacement,
    )
    assert transported == pytest.approx(quadratic_gradient(birth+displacement, matrix))


def test_transport_radius_is_second_order_without_hvp_noise() -> None:
    small = transported_radius(0.0, 0.0, 0.5, 0.1)
    raw = raw_staleness_radius(0.0, 2.0, 0.1)
    assert small == pytest.approx(0.0025)
    assert small < raw


def test_gain_optimal_step_matches_grid() -> None:
    cert = TransportCertificate(1.2, 0.1, 1.5, 1.0)
    optimum = gain_optimal_step(cert)
    grid = np.linspace(0.0, 1.0, 100_001)
    numeric = float(grid[np.argmax([certified_gain(cert, x) for x in grid])])
    assert optimum == pytest.approx(numeric, abs=1e-5)


def test_large_transport_radius_rejects() -> None:
    cert = TransportCertificate(0.2, 0.3, 1.0, 1.0)
    assert gain_optimal_step(cert) == 0.0


def test_lyapunov_step_matches_dense_grid() -> None:
    cert = TransportCertificate(1.0, 0.05, 1.0, 1.0)
    pending = (
        PendingTransportDebt(0.1, 0.2, 0.03, 0.4),
        PendingTransportDebt(0.2, 0.4, 0.02, 0.6),
    )
    optimum = lyapunov_optimal_step(cert, pending, potential_weight=3.0)
    grid = np.linspace(0.0, 1.0, 100_001)
    numeric = float(grid[np.argmin([
        transport_lyapunov_drift_bound(
            cert, pending, x, potential_weight=3.0
        )
        for x in grid
    ])])
    assert optimum == pytest.approx(numeric, abs=1e-5)


def test_no_pending_debt_recovers_gain_optimum() -> None:
    cert = TransportCertificate(0.9, 0.1, 1.3, 0.8)
    lyapunov = lyapunov_optimal_step(cert, (), potential_weight=7.0)
    assert lyapunov == pytest.approx(gain_optimal_step(cert), abs=1e-12)


def test_public_curvature_bound_covers_exact_second_derivative() -> None:
    cert = TransportCertificate(1.1, 0.2, 1.4, 0.8)
    pending = (
        PendingTransportDebt(0.1, 0.3, 0.02, 0.2),
        PendingTransportDebt(0.2, 0.4, 0.04, 0.3, weight=1.5),
    )
    second = transport_drift_second_derivative(
        cert, pending, 0.7, potential_weight=3.0
    )
    bound = public_curvature_bound(
        potential_weight=3.0,
        smoothness_max=1.4,
        max_pending=2,
        weight_max=1.5,
        gradient_radius_max=0.2,
        hvp_radius_max=0.04,
        hessian_lipschitz_max=0.3,
        path_length_max=0.4,
        proposal_norm_max=1.1,
        max_step=0.8,
    )
    assert second <= bound*cert.proposal_norm**2


def test_invalid_radius_rejected() -> None:
    with pytest.raises(ValueError):
        transported_radius(0.0, -0.1, 1.0, 0.2)


def test_outcome_free_audit_closes_all_algebra_checks() -> None:
    result = audit()
    assert result["exact_quadratic_checks"] == 384
    assert result["nonlinear_radius_checks"] == 576
    assert result["transport_radius_improvement_checks"] == 576
    assert result["lyapunov_optimizer_checks"] == 162
    assert result["public_curvature_bound_checks"] == 162
    assert result["scientific_population_generated"] is False
