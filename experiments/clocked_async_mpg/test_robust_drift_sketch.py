from __future__ import annotations

import numpy as np
import pytest

from .coupled_actor_critic_drift import box_qp_objective
from .robust_drift_sketch import (
    choose_robust_drift_sketch,
    comparator_excess_bound,
)


def test_robust_sketch_upper_bounds_every_covered_true_drift() -> None:
    rng = np.random.default_rng(4401)
    for _ in range(300):
        basis = rng.normal(size=(2, 2))
        true_quadratic = basis.T @ basis + 0.2 * np.eye(2)
        true_linear = rng.normal(size=2)
        linear_error = rng.uniform(0.01, 0.08, size=2)
        quadratic_error = float(rng.uniform(0.01, 0.09))
        linear_perturbation = rng.uniform(-1.0, 1.0, size=2) * linear_error
        symmetric = rng.normal(size=(2, 2))
        symmetric = 0.5 * (symmetric + symmetric.T)
        norm = np.linalg.norm(symmetric, ord=2)
        symmetric *= quadratic_error / max(norm, 1e-15)
        estimated_linear = true_linear + linear_perturbation
        estimated_quadratic = true_quadratic + symmetric
        decision = choose_robust_drift_sketch(
            estimated_linear=estimated_linear,
            estimated_quadratic=estimated_quadratic,
            linear_absolute_error=linear_error,
            quadratic_operator_error=quadratic_error,
            upper=np.asarray([0.7, 0.8]),
        )
        true = box_qp_objective(
            decision.action, true_linear, true_quadratic
        )
        assert true <= decision.robust_upper_objective + 1e-11
        assert decision.robust_upper_objective <= 1e-11


def test_exact_sketch_reduces_to_exact_qp() -> None:
    linear = np.asarray([-1.0, -0.5])
    quadratic = np.asarray([[2.0, -0.3], [-0.3, 1.4]])
    decision = choose_robust_drift_sketch(
        estimated_linear=linear,
        estimated_quadratic=quadratic,
        linear_absolute_error=np.zeros(2),
        quadratic_operator_error=0.0,
        upper=np.ones(2),
    )
    assert decision.robust_upper_objective == pytest.approx(
        decision.estimated_objective
    )
    assert (decision.action > 0.0).all()


def test_comparator_excess_bound_covers_robust_minus_true_objective() -> None:
    true_linear = np.asarray([-0.8, -0.3])
    true_quadratic = np.asarray([[1.7, -0.2], [-0.2, 1.1]])
    linear_error = np.asarray([0.04, 0.07])
    quadratic_error = 0.09
    estimated_linear = true_linear - linear_error
    estimated_quadratic = true_quadratic - quadratic_error * np.eye(2)
    comparator = np.asarray([0.3, 0.5])
    robust = (
        (estimated_linear + linear_error) @ comparator
        + 0.5
        * comparator
        @ (estimated_quadratic + quadratic_error * np.eye(2))
        @ comparator
    )
    true = box_qp_objective(comparator, true_linear, true_quadratic)
    assert robust - true <= comparator_excess_bound(
        comparator=comparator,
        linear_absolute_error=linear_error,
        quadratic_operator_error=quadratic_error,
    ) + 1e-12


def test_uncovered_indefinite_estimate_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        choose_robust_drift_sketch(
            estimated_linear=np.zeros(2),
            estimated_quadratic=np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
            linear_absolute_error=np.zeros(2),
            quadratic_operator_error=0.2,
            upper=np.ones(2),
        )
