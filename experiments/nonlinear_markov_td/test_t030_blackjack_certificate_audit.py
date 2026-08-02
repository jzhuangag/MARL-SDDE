import numpy as np

from experiments.nonlinear_markov_td.t030_blackjack_certificate_audit import (
    exact_euclidean_constants,
    optimistic_audit,
)


def test_exact_curvature_catalogue_is_positive_and_complete() -> None:
    result = exact_euclidean_constants()
    assert result["dimension"] == 280
    assert result["monotonicity"] > 0.0
    assert len(result["actions"]) == 36
    assert all(row["curvature"] > 0.0 for row in result["actions"])


def test_optimistic_closed_form_is_theorem4_zero_residual_minimum() -> None:
    result = optimistic_audit()
    mu = float(result["monotonicity"])
    curvature = float(result["minimum_curvature"])
    eta = float(result["optimistic_eta"])
    direct = 1.0 - eta * mu + 2.0 * curvature * eta * eta
    np.testing.assert_allclose(
        direct, result["optimistic_one_step_contraction"], atol=1e-15
    )
    assert not result["five_percent_nonvacuity_gate"]
    assert result["optimistic_maximum_improvement"] < 0.001
