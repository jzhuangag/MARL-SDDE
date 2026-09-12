from __future__ import annotations

import numpy as np
import pytest

from .freshness_sensing import (
    choose_budgeted_freshness_refresh,
    choose_freshness_refresh,
    cross_policy_bias_upper,
    fuse_gradient_estimates,
    optimal_fusion_certificate,
    smooth_potential_progress_lower_bound,
    update_risk_debt,
    update_resource_debts,
)


def test_fusion_weight_minimizes_quadratic_mse_bound() -> None:
    certificate = optimal_fusion_certificate(
        birth_variance=2.0,
        fresh_variance=1.0,
        birth_bias_upper=1.0,
    )
    assert certificate.fresh_weight == pytest.approx(0.75)
    assert certificate.refresh_mse_upper == pytest.approx(0.75)
    assert certificate.no_refresh_mse_upper == pytest.approx(3.0)
    assert certificate.refresh_value == pytest.approx(2.25)


def test_fusion_reduces_to_equal_average_under_equal_unbiased_noise() -> None:
    certificate = optimal_fusion_certificate(
        birth_variance=1.0,
        fresh_variance=1.0,
        birth_bias_upper=0.0,
    )
    fused = fuse_gradient_estimates(
        np.asarray([1.0, 3.0]),
        np.asarray([3.0, 1.0]),
        certificate,
    )
    assert certificate.fresh_weight == 0.5
    assert np.allclose(fused, np.asarray([2.0, 2.0]))


def test_cross_policy_drift_moves_weight_toward_fresh_estimator() -> None:
    no_drift = optimal_fusion_certificate(
        birth_variance=1.0,
        fresh_variance=1.0,
        birth_bias_upper=0.0,
    )
    bias = cross_policy_bias_upper(
        np.asarray([0.08, 0.0]), cross_gradient_lipschitz=2.0
    )
    drift = optimal_fusion_certificate(
        birth_variance=1.0,
        fresh_variance=1.0,
        birth_bias_upper=bias,
    )
    assert bias == pytest.approx(0.4)
    assert drift.fresh_weight > no_drift.fresh_weight


def test_lyapunov_threshold_refreshes_only_when_risk_debt_prices_value() -> None:
    certificate = optimal_fusion_certificate(
        birth_variance=1.0,
        fresh_variance=1.0,
        birth_bias_upper=1.0,
    )
    no_refresh = choose_freshness_refresh(
        certificate,
        risk_debt=0.1,
        refresh_cost=1.0,
        cost_tradeoff=1.0,
        mse_budget=0.5,
    )
    refresh = choose_freshness_refresh(
        certificate,
        risk_debt=2.0,
        refresh_cost=1.0,
        cost_tradeoff=1.0,
        mse_budget=0.5,
    )
    assert no_refresh.refresh is False
    assert refresh.refresh is True
    assert refresh.incurred_mse_upper < no_refresh.incurred_mse_upper


def test_smooth_progress_improves_when_estimation_mse_falls() -> None:
    poor = smooth_potential_progress_lower_bound(
        gradient_norm=2.0,
        mse_upper=1.0,
        learning_rate=0.05,
        smoothness=1.0,
    )
    good = smooth_potential_progress_lower_bound(
        gradient_norm=2.0,
        mse_upper=0.04,
        learning_rate=0.05,
        smoothness=1.0,
    )
    assert good > poor


def test_virtual_queue_certifies_sample_path_average_risk() -> None:
    budget = 0.4
    incurred = [0.8, 0.1, 0.5, 0.2]
    debt = 0.0
    for value in incurred:
        debt = update_risk_debt(
            debt, incurred_mse_upper=value, mse_budget=budget
        )
    assert sum(incurred) <= len(incurred) * budget + debt + 1e-15


def test_resource_queue_prices_refresh_and_certifies_cost() -> None:
    certificate = optimal_fusion_certificate(
        birth_variance=1.0,
        fresh_variance=1.0,
        birth_bias_upper=2.0,
    )
    decision = choose_budgeted_freshness_refresh(
        certificate,
        resource_debts=np.asarray([0.1, 0.2]),
        refresh_costs=np.asarray([1.0, 0.5]),
        average_budgets=np.asarray([0.25, 0.125]),
        risk_tradeoff=1.0,
    )
    assert decision.refresh is True
    assert decision.resource_price == pytest.approx(0.2)
    assert decision.resource_debts_after == pytest.approx((0.85, 0.575))


def test_hard_budget_can_veto_an_economically_valuable_refresh() -> None:
    certificate = optimal_fusion_certificate(
        birth_variance=1.0,
        fresh_variance=1.0,
        birth_bias_upper=2.0,
    )
    decision = choose_budgeted_freshness_refresh(
        certificate,
        resource_debts=np.zeros(1),
        refresh_costs=np.ones(1),
        average_budgets=np.asarray([0.25]),
        risk_tradeoff=10.0,
        hard_budget_feasible=False,
    )
    assert decision.refresh is False
    assert decision.incurred_costs == (0.0,)


def test_resource_queue_sample_path_certificate() -> None:
    average_budget = np.asarray([0.25, 0.5])
    debt = np.zeros(2)
    cumulative = np.zeros(2)
    for costs in (
        np.asarray([1.0, 0.5]),
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 0.5]),
        np.asarray([0.0, 0.0]),
    ):
        cumulative += costs
        debt = update_resource_debts(
            debt,
            incurred_costs=costs,
            average_budgets=average_budget,
        )
    assert np.all(cumulative <= 4 * average_budget + debt + 1e-15)
