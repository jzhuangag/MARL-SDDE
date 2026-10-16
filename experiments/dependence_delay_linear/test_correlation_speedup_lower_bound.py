"""Tests for the correlation-limited Gaussian minimax lower bound."""

import numpy as np

from correlation_speedup_lower_bound import (
    adaptive_budget_lower_bound,
    continuous_optimal_participation,
    effective_speedup,
    fisher_information,
    minimax_risk,
    observation_covariance,
    optimal_integer_participation,
)


def test_closed_form_fisher_information_matches_matrix_inverse():
    for q in (1, 2, 8, 32):
        covariance = observation_covariance(q, 0.35, 0.65)
        direct = np.ones(q) @ np.linalg.solve(
            covariance, np.ones(q)
        )
        assert np.isclose(
            direct,
            fisher_information(q, 0.35, 0.65),
            rtol=1e-12,
            atol=1e-12,
        )


def test_exact_risk_and_correlation_limited_speedup():
    q = 32
    rho = 0.4
    risk_one = minimax_risk(100, 1, rho, 1.0 - rho)
    risk_q = minimax_risk(100, q, rho, 1.0 - rho)
    assert np.isclose(risk_one / risk_q, effective_speedup(q, rho))
    assert effective_speedup(q, rho) <= 1.0 / rho
    assert effective_speedup(q, 0.0) == q


def test_continuous_and_integer_budget_optima_agree():
    candidate_agents = range(1, 33)
    overhead = 16.0
    common_variance = 0.2
    private_variance = 0.8
    continuous = continuous_optimal_participation(
        32, overhead, common_variance, private_variance
    )
    selected = optimal_integer_participation(
        candidate_agents,
        overhead,
        common_variance,
        private_variance,
    )
    assert continuous == 8.0
    assert selected["q"] == 8


def test_adaptive_budget_lower_bound_equals_best_fixed_efficiency():
    candidates = (1, 2, 4, 8, 16, 32)
    selected = optimal_integer_participation(
        candidates, 8.0, 0.5, 0.5
    )
    lower = adaptive_budget_lower_bound(
        1000.0, candidates, 8.0, 0.5, 0.5
    )
    assert np.isclose(
        lower,
        1.0 / (1000.0 * selected["information_per_cost"]),
    )
