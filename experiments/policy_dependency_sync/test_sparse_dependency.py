import numpy as np
import pytest

from experiments.policy_dependency_sync.sparse_dependency import (
    omitted_dependency_tail,
    sparse_score_error_bound,
    validate_neighborhood,
)


def test_exact_sparse_quadratic_has_zero_omitted_gradient_tail():
    sensitivities = {1: 0.7, 2: 0.0, 3: 0.4, 4: 0.0}
    mismatches = {1: 0.3, 2: 1.2, 3: 0.6, 4: 0.9}
    assert omitted_dependency_tail(
        sensitivities, mismatches, {1, 3}
    ) == pytest.approx(0.0)


def test_dense_quadratic_omission_is_covered_by_l1_envelope():
    rng = np.random.default_rng(402)
    for _ in range(1_000):
        coefficients = rng.normal(size=7)
        displacement = rng.normal(size=7)
        neighbors = {0, 2, 5}
        sensitivities = {
            index: abs(float(value)) for index, value in enumerate(coefficients)
        }
        mismatches = {
            index: abs(float(value)) for index, value in enumerate(displacement)
        }
        actual_tail = sum(
            coefficients[index] * displacement[index]
            for index in range(7)
            if index not in neighbors
        )
        bound = omitted_dependency_tail(
            sensitivities, mismatches, neighbors
        )
        assert abs(actual_tail) <= bound + 1e-12


def test_sparse_score_bound_covers_alignment_and_curvature_error():
    rng = np.random.default_rng(991)
    maximum_step = 0.2
    smoothness = 1.7
    for _ in range(1_000):
        objective_gradient = rng.normal(size=4)
        local_gradient = rng.normal(size=4)
        omitted = rng.normal(size=4)
        omitted *= 0.3 / max(np.linalg.norm(omitted), 1e-12)
        step = float(rng.uniform(0.0, maximum_step))
        full_gradient = local_gradient + omitted
        local_score = (
            -step * objective_gradient @ local_gradient
            + 0.5 * smoothness * step**2 * np.linalg.norm(local_gradient) ** 2
        )
        full_score = (
            -step * objective_gradient @ full_gradient
            + 0.5 * smoothness * step**2 * np.linalg.norm(full_gradient) ** 2
        )
        bound = sparse_score_error_bound(
            maximum_step,
            float(np.linalg.norm(objective_gradient)),
            float(np.linalg.norm(local_gradient)),
            float(np.linalg.norm(omitted)),
            smoothness,
        )
        assert abs(full_score - local_score) <= bound + 1e-12


def test_neighborhood_validation_rejects_owner_and_out_of_range():
    validate_neighborhood(5, 0, {1, 3})
    with pytest.raises(ValueError):
        validate_neighborhood(5, 0, {0, 1})
    with pytest.raises(ValueError):
        validate_neighborhood(5, 0, {5})
