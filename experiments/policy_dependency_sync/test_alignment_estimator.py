from __future__ import annotations

import numpy as np
import pytest

from experiments.policy_dependency_sync.alignment_estimator import (
    GradientErrorBudget,
    geometric_mean_variance_factor,
    markov_mean_rms_bound,
    optimized_quadratic_score,
    optimized_score_error_bound,
    sparse_candidate_gradient,
    split_fully_charged_rollouts,
)


def test_geometric_factor_matches_covariance_double_sum() -> None:
    for count in (1, 2, 7, 32):
        for rho in (0.0, 0.4, 0.9):
            covariance_sum = sum(
                rho ** abs(left - right)
                for left in range(count)
                for right in range(count)
            )
            expected = covariance_sum / count
            assert geometric_mean_variance_factor(count, rho) == pytest.approx(
                expected
            )


def test_markov_rms_bound_matches_exact_scalar_ar1_mean_variance() -> None:
    count = 23
    rho = 0.8
    variance = 1.7
    covariance_sum = sum(
        variance * rho ** abs(left - right)
        for left in range(count)
        for right in range(count)
    )
    exact_rms = np.sqrt(covariance_sum / count**2)
    observed = markov_mean_rms_bound(count, rho, variance)
    assert observed == pytest.approx(exact_rms)


def test_candidate_error_budget_adds_every_declared_source_once() -> None:
    budget = GradientErrorBudget(
        base_rms=0.1,
        jvp_rms=0.2,
        displacement_norm=0.5,
        taylor_remainder=0.03,
        sparse_tail=0.04,
        off_policy_bias=0.02,
    )
    assert budget.candidate_rms == pytest.approx(0.29)


def test_sparse_candidate_gradient_is_sample_mean_jvp_contraction() -> None:
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    jvp = np.array(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 4.0]],
        ]
    )
    displacement = np.array([0.5, -0.25])
    expected = base.mean(axis=0) + jvp.mean(axis=0) @ displacement
    np.testing.assert_allclose(
        sparse_candidate_gradient(base, jvp, displacement), expected
    )


def test_optimized_score_error_bound_covers_deterministic_perturbations() -> None:
    rng = np.random.default_rng(91017)
    curvature = 1.7
    maximum_step = 0.4
    for _ in range(1000):
        current = rng.normal(size=4)
        candidate = rng.normal(size=4)
        current_error = rng.normal(size=4)
        candidate_error = rng.normal(size=4)
        current_radius = float(np.linalg.norm(current_error))
        candidate_radius = float(np.linalg.norm(candidate_error))
        true_score = optimized_quadratic_score(
            current, candidate, curvature, maximum_step
        )
        estimated_score = optimized_quadratic_score(
            current + current_error,
            candidate + candidate_error,
            curvature,
            maximum_step,
        )
        bound = optimized_score_error_bound(
            maximum_step=maximum_step,
            curvature=curvature,
            current_gradient_norm_bound=float(np.linalg.norm(current)),
            candidate_gradient_norm_bound=float(np.linalg.norm(candidate)),
            current_gradient_rms_error=current_radius,
            candidate_gradient_rms_error=candidate_radius,
        )
        assert abs(estimated_score - true_score) <= bound + 1e-12


def test_rollout_split_is_disjoint_and_fully_charged() -> None:
    control, update = split_fully_charged_rollouts(np.arange(12))
    assert control.tolist() == list(range(6))
    assert update.tolist() == list(range(6, 12))
    assert set(control).isdisjoint(update)
    assert len(control) + len(update) == 12


@pytest.mark.parametrize(
    "call",
    [
        lambda: geometric_mean_variance_factor(0, 0.2),
        lambda: geometric_mean_variance_factor(3, 1.0),
        lambda: markov_mean_rms_bound(3, 0.2, -1.0),
        lambda: split_fully_charged_rollouts(np.arange(3)),
    ],
)
def test_invalid_inputs_fail_closed(call) -> None:
    with pytest.raises(ValueError):
        call()
