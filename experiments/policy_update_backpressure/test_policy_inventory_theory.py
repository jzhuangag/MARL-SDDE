from __future__ import annotations

import math

import numpy as np
import pytest

from .policy_inventory_theory import (
    CompletedInventoryProposal,
    PendingInventory,
    gaussian_joint_log_second_moment,
    importance_variance_inflation,
    inventory_drift_second_derivative,
    inventory_lyapunov_drift,
    inventory_optimal_step,
    rms_importance_gradient_radius,
)


def test_factorized_gaussian_second_moment_is_additive() -> None:
    current = np.asarray([0.5, -0.2, 0.7])
    birth = np.asarray([-0.1, -0.4, 0.2])
    variance = np.asarray([0.5, 1.0, 2.0])
    joint = gaussian_joint_log_second_moment(current, birth, variance)
    separate = sum(
        gaussian_joint_log_second_moment(
            current[i:i+1], birth[i:i+1], variance[i:i+1]
        )
        for i in range(current.size)
    )
    assert joint == pytest.approx(separate)
    assert importance_variance_inflation(joint) == pytest.approx(math.expm1(joint))


def test_gauss_hermite_integration_matches_second_moment_identity() -> None:
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    current = np.asarray([0.65, -0.35])
    birth = np.asarray([-0.15, 0.25])
    variance = np.asarray([0.8, 1.4])
    numerical = 1.0
    for target, behavior, var in zip(current, birth, variance, strict=True):
        actions = behavior+math.sqrt(2.0*var)*nodes
        log_ratio = -(
            (actions-target)**2-(actions-behavior)**2
        )/(2.0*var)
        numerical *= float(np.sum(weights*np.exp(2.0*log_ratio))/math.sqrt(math.pi))
    analytic = math.exp(
        gaussian_joint_log_second_moment(current, birth, variance)
    )
    assert numerical == pytest.approx(analytic, rel=2e-12)


def test_rms_radius_scales_with_batch_and_inventory() -> None:
    base = rms_importance_gradient_radius(
        sample_norm_bound=3.0, batch_size=64, log_second_moment=0.0
    )
    doubled_batch = rms_importance_gradient_radius(
        sample_norm_bound=3.0, batch_size=256, log_second_moment=0.0
    )
    stale = rms_importance_gradient_radius(
        sample_norm_bound=3.0,
        batch_size=64,
        log_second_moment=math.log(4.0),
    )
    assert doubled_batch == pytest.approx(base/2.0)
    assert stale == pytest.approx(2.0*base)


def test_pending_quadratic_matches_direct_gaussian_geometry() -> None:
    birth = np.asarray([0.2, -0.4, 0.1])
    current = np.asarray([0.5, 0.3, -0.2])
    direction = np.asarray([0.0, -0.6, 0.0])
    variance = np.asarray([0.7, 1.3, 0.9])
    z0 = gaussian_joint_log_second_moment(current, birth, variance)
    delta = current[1]-birth[1]
    item = PendingInventory(
        log_second_moment=z0,
        linear=2.0*delta*direction[1]/variance[1],
        quadratic=direction[1]**2/variance[1],
    )
    for step in np.linspace(0.0, 1.2, 25):
        updated = current+step*direction
        direct = gaussian_joint_log_second_moment(updated, birth, variance)
        assert item.post_step_log_second_moment(float(step)) == pytest.approx(direct)


def test_scalar_inventory_drift_is_convex_and_solver_matches_grid() -> None:
    proposal = CompletedInventoryProposal(
        direction_norm=1.4,
        error_radius=0.15,
        block_smoothness=1.2,
        log_second_moment=0.4,
        max_step=1.1,
    )
    pending = (
        PendingInventory(0.3, 0.8, 0.7, 1.0),
        PendingInventory(0.5, -0.4, 0.6, 0.8),
        PendingInventory(0.2, 0.2, 0.5, 1.3),
    )
    for step in np.linspace(0.0, proposal.max_step, 101):
        assert inventory_drift_second_derivative(
            proposal, pending, float(step), potential_weight=4.0
        ) > 0
    optimum = inventory_optimal_step(
        proposal, pending, potential_weight=4.0
    )
    grid = np.linspace(0.0, proposal.max_step, 20001)
    values = np.asarray([
        inventory_lyapunov_drift(
            proposal, pending, float(step), potential_weight=4.0
        )
        for step in grid
    ])
    grid_optimum = float(grid[int(np.argmin(values))])
    assert abs(optimum-grid_optimum) <= proposal.max_step/20000+1e-9


def test_no_pending_inventory_recovers_certified_gain_step() -> None:
    proposal = CompletedInventoryProposal(
        direction_norm=2.0,
        error_radius=0.5,
        block_smoothness=1.25,
        log_second_moment=0.0,
        max_step=1.0,
    )
    expected = (1.0-proposal.error_radius/proposal.direction_norm)/proposal.block_smoothness
    actual = inventory_optimal_step(proposal, (), potential_weight=2.0)
    assert actual == pytest.approx(expected, abs=1e-10)
