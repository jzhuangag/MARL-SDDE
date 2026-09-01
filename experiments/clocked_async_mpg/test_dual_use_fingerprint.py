from __future__ import annotations

import math

import numpy as np
import pytest

from .dual_use_fingerprint import directional_geometry_fingerprint


@pytest.mark.parametrize("seed", range(16))
def test_potential_fingerprint_is_aligned_without_rotation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=7)
    gradient = state
    lookahead = (1.0 - 0.3) * state
    fingerprint = directional_geometry_fingerprint(
        gradient, lookahead, lookahead_step=0.3
    )
    assert fingerprint.informative
    assert fingerprint.symmetric_alignment == pytest.approx(1.0, abs=1e-12)
    assert fingerprint.rotational_residual == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("seed", range(16))
def test_skew_fingerprint_is_orthogonal_and_rotational(seed: int) -> None:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2)
    operator = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    gradient = operator @ state
    lookahead_state = state - 0.4 * gradient
    lookahead_gradient = operator @ lookahead_state
    fingerprint = directional_geometry_fingerprint(
        gradient, lookahead_gradient, lookahead_step=0.4
    )
    assert fingerprint.informative
    assert fingerprint.symmetric_alignment == pytest.approx(0.0, abs=1e-12)
    assert fingerprint.rotational_residual == pytest.approx(1.0, abs=1e-12)
    assert fingerprint.jacobian_action_energy_ratio == pytest.approx(1.0)


def test_fingerprint_matches_a_general_linear_jacobian_action() -> None:
    operator = np.asarray([[1.0, 2.0], [-0.5, 0.25]])
    state = np.asarray([0.4, -0.8])
    gradient = operator @ state
    lookahead_gradient = operator @ (state - 0.2 * gradient)
    fingerprint = directional_geometry_fingerprint(
        gradient, lookahead_gradient, lookahead_step=0.2
    )
    action = operator @ gradient
    expected_alignment = float(gradient @ action / (gradient @ gradient))
    expected_ratio = float(action @ action / (gradient @ gradient))
    assert fingerprint.symmetric_alignment == pytest.approx(expected_alignment)
    assert fingerprint.jacobian_action_energy_ratio == pytest.approx(expected_ratio)
    assert fingerprint.rotational_residual == pytest.approx(
        math.sqrt(max(0.0, expected_ratio - expected_alignment**2))
    )


def test_small_gradient_fails_uninformatively() -> None:
    fingerprint = directional_geometry_fingerprint(
        np.zeros(3),
        np.zeros(3),
        lookahead_step=0.5,
        minimum_gradient_energy=1e-6,
    )
    assert not fingerprint.informative
    assert math.isnan(fingerprint.symmetric_alignment)
    assert math.isnan(fingerprint.rotational_residual)


@pytest.mark.parametrize(
    ("current", "lookahead", "step"),
    [
        (np.zeros((2, 1)), np.zeros((2, 1)), 0.1),
        (np.zeros(2), np.zeros(3), 0.1),
        (np.zeros(2), np.zeros(2), 0.0),
    ],
)
def test_fingerprint_validates_inputs(
    current: np.ndarray, lookahead: np.ndarray, step: float
) -> None:
    with pytest.raises(ValueError):
        directional_geometry_fingerprint(
            current, lookahead, lookahead_step=step
        )
