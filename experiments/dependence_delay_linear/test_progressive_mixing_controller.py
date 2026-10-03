"""Tests for exact progressive covariance propagation."""

import numpy as np

from predictable_mixing_controller import select_joint_action
from progressive_mixing_controller import (
    advance_action,
    advance_observations,
    final_expected_error,
    initial_covariance_state,
)


def test_observation_only_preserves_constant_initial_covariance() -> None:
    initial = initial_covariance_state(delay=2)
    advanced = advance_observations(initial, 0.98, 137, delay=2)
    assert np.allclose(initial, advanced)


def test_safe_action_reduces_exact_homogeneous_error_without_noise_blowup() -> None:
    action = select_joint_action(
        0.985,
        rho=0.0,
        delay=0,
        pilot_cost=0,
        resource_budget=2_000,
    )
    initial = initial_covariance_state(delay=0)
    result = advance_action(
        initial,
        action,
        persistence=0.98,
        rho=0.0,
        delay=0,
        updates=10,
    )
    assert result["radius"] < 1.0
    assert np.isfinite(final_expected_error(result["state"], 0))
