from __future__ import annotations

import numpy as np
import pytest

from .theory_gate import (
    first_arrival_noise_mean,
    interaction_weighted_drift_bound,
    krasovskii_history_drift,
    quadratic_block_gradient,
    quadratic_potential,
    quadratic_smooth_gain_lower_bound,
)


def test_quadratic_block_gradient_matches_finite_difference() -> None:
    curvature = np.asarray([[2.0, -0.4], [-0.4, 1.3]])
    theta = np.asarray([0.7, -0.2])
    epsilon = 1e-6
    for block in range(2):
        direction = np.zeros(2)
        direction[block] = epsilon
        finite = (
            quadratic_potential(theta+direction, curvature)
            -quadratic_potential(theta-direction, curvature)
        )/(2.0*epsilon)
        assert finite == pytest.approx(
            quadratic_block_gradient(theta, curvature, block), abs=1e-9
        )


def test_interaction_weighted_bound_covers_quadratic_mismatch() -> None:
    rng = np.random.default_rng(20260901)
    base = rng.normal(size=(5, 5))
    curvature = base.T@base+0.2*np.eye(5)
    current = rng.normal(size=5)
    birth = rng.normal(size=5)
    for block in range(5):
        mismatch = abs(
            quadratic_block_gradient(current, curvature, block)
            -quadratic_block_gradient(birth, curvature, block)
        )
        bound = interaction_weighted_drift_bound(
            np.abs(curvature), current, birth, block
        )
        assert mismatch <= bound+1e-12


def test_same_scalar_age_can_have_different_teammate_mismatch() -> None:
    curvature = np.asarray(
        [[2.0, 1.5, 0.0], [1.5, 2.0, 0.0], [0.0, 0.0, 1.0]]
    )
    birth = np.zeros(3)
    high_coupling_change = np.asarray([0.0, 1.0, 0.0])
    irrelevant_change = np.asarray([0.0, 0.0, 1.0])
    high_error = abs(
        quadratic_block_gradient(high_coupling_change, curvature, 0)
        -quadratic_block_gradient(birth, curvature, 0)
    )
    zero_error = abs(
        quadratic_block_gradient(irrelevant_change, curvature, 0)
        -quadratic_block_gradient(birth, curvature, 0)
    )
    assert high_error == pytest.approx(1.5)
    assert zero_error == pytest.approx(0.0)


def test_quadratic_smooth_gain_certificate_is_exact() -> None:
    curvature = np.asarray([[2.0, 0.7], [0.7, 1.5]])
    current = np.asarray([0.3, -0.8])
    stale = np.asarray([-0.2, 0.4])
    for block in range(2):
        actual, certified = quadratic_smooth_gain_lower_bound(
            current, stale, curvature, block, step_size=0.17
        )
        assert actual == pytest.approx(certified, abs=1e-12)


def test_krasovskii_history_drift_identity() -> None:
    energy = np.asarray([0.7, 0.2, 1.3, 0.4])
    direct, telescope = krasovskii_history_drift(energy, 0.9)
    assert direct == pytest.approx(telescope, abs=1e-12)


def test_endogenous_completion_creates_arrival_order_bias() -> None:
    assert first_arrival_noise_mean(endogenous=False) == pytest.approx(0.0)
    assert first_arrival_noise_mean(endogenous=True) == pytest.approx(0.5)


def test_theory_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        quadratic_potential(np.ones(2), np.asarray([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        interaction_weighted_drift_bound(
            np.asarray([[1.0, -1.0], [0.0, 1.0]]),
            np.zeros(2),
            np.zeros(2),
            0,
        )
    with pytest.raises(ValueError):
        krasovskii_history_drift(np.asarray([]), 1.0)

