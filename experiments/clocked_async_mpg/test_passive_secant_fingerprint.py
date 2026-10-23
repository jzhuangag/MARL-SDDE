from __future__ import annotations

import numpy as np
import pytest

from .passive_secant_fingerprint import passive_secant_fingerprint


def test_identity_operator_is_pure_symmetric_alignment() -> None:
    x0 = np.asarray([0.2, -0.7])
    x1 = np.asarray([-0.4, 0.5])
    fingerprint = passive_secant_fingerprint(x0, x0, x1, x1)
    assert fingerprint.informative
    assert fingerprint.symmetric_alignment == pytest.approx(1.0)
    assert fingerprint.rotational_residual == pytest.approx(0.0, abs=1e-12)


def test_planar_rotation_is_pure_orthogonal_response() -> None:
    operator = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    x0 = np.asarray([0.2, -0.7])
    x1 = np.asarray([-0.4, 0.5])
    fingerprint = passive_secant_fingerprint(
        x0, operator @ x0, x1, operator @ x1
    )
    assert fingerprint.informative
    assert fingerprint.symmetric_alignment == pytest.approx(0.0, abs=1e-12)
    assert fingerprint.rotational_residual == pytest.approx(1.0)


def test_random_linear_secant_matches_direct_jacobian_action() -> None:
    rng = np.random.default_rng(92061)
    for dimension in (2, 7, 19):
        operator = rng.normal(size=(dimension, dimension))
        x0 = rng.normal(size=dimension)
        x1 = rng.normal(size=dimension)
        result = passive_secant_fingerprint(
            x0, operator @ x0, x1, operator @ x1
        )
        displacement = x1 - x0
        action = operator @ displacement
        energy = float(displacement @ displacement)
        alignment = float(displacement @ action / energy)
        residual = np.linalg.norm(action - alignment * displacement) / np.linalg.norm(
            displacement
        )
        assert result.symmetric_alignment == pytest.approx(alignment)
        assert result.rotational_residual == pytest.approx(residual)


def test_zero_displacement_fails_closed() -> None:
    x = np.asarray([1.0, 2.0])
    result = passive_secant_fingerprint(x, x, x, x, minimum_displacement_energy=0.0)
    assert not result.informative
    assert np.isnan(result.symmetric_alignment)


def test_invalid_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        passive_secant_fingerprint(
            np.ones(2), np.ones(2), np.ones(3), np.ones(2)
        )
