"""Tests for the exact time-varying participation law in T-047."""

from __future__ import annotations

import numpy as np

from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    equicorrelated_ar_lag_covariances,
    exact_vector_risk,
)
from experiments.dependence_delay_linear.t047_scheduled_participation import (
    AffineRisk,
    exact_scheduled_vector_risk,
    hoeffding_interval,
    prefix_overlap_factor,
    robust_post_probe_choice,
    schedule_budget_summary,
    schedule_is_feasible,
    scheduled_risk_affine_coefficients,
)


def base_ar_lags(updates: int, variance: float, coefficient: float) -> np.ndarray:
    lags = np.arange(-(updates - 1), updates)
    return np.asarray(
        [[[variance * coefficient ** abs(int(lag))]] for lag in lags], dtype=float
    )


def test_prefix_overlap_factor_has_correct_endpoints() -> None:
    assert prefix_overlap_factor(2, 8, 0.0) == 1.0 / 8.0
    assert prefix_overlap_factor(2, 8, 1.0) == 1.0
    assert prefix_overlap_factor(4, 4, 0.0) == 1.0 / 4.0


def test_constant_schedule_matches_t037_stationary_identity() -> None:
    updates = 7
    q = 4
    rho = 0.35
    variance = np.asarray([[1.7]])
    standard = exact_vector_risk(
        initial_history=np.asarray([[2.0]]),
        drift=np.asarray([[0.6]]),
        step_size=0.12,
        delay=0,
        updates=updates,
        lag_covariances=equicorrelated_ar_lag_covariances(
            horizon=updates,
            single_agent_covariance=variance,
            q=q,
            rho=rho,
            markov_lambda=0.4,
        ),
    )
    scheduled = exact_scheduled_vector_risk(
        initial_history=np.asarray([[2.0]]),
        drift=np.asarray([[0.6]]),
        step_size=0.12,
        delay=0,
        q_schedule=[q] * updates,
        rho=rho,
        base_lag_covariances=base_ar_lags(updates, 1.7, 0.4),
    )
    assert np.allclose(scheduled["mean"], standard["mean"], atol=1e-13)
    assert np.allclose(scheduled["covariance"], standard["covariance"], atol=1e-13)
    assert np.isclose(scheduled["risk"], standard["risk"], atol=1e-13)


def test_perfect_common_factor_removes_schedule_dependence_at_fixed_horizon() -> None:
    arguments = {
        "initial_history": np.asarray([[1.0]]),
        "drift": np.asarray([[0.5]]),
        "step_size": 0.1,
        "delay": 0,
        "rho": 1.0,
        "base_lag_covariances": base_ar_lags(6, 2.0, 0.3),
    }
    one = exact_scheduled_vector_risk(q_schedule=[1] * 6, **arguments)
    varying = exact_scheduled_vector_risk(q_schedule=[1, 2, 8, 4, 2, 1], **arguments)
    assert np.isclose(one["risk"], varying["risk"], atol=1e-13)


def test_risk_is_affine_in_common_factor_correlation() -> None:
    arguments = {
        "initial_history": np.asarray([[1.5], [1.5]]),
        "drift": np.asarray([[0.7]]),
        "step_size": 0.08,
        "delay": 1,
        "q_schedule": [1, 2, 8, 4, 1],
        "base_lag_covariances": base_ar_lags(5, 1.2, 0.5),
    }
    coefficients = scheduled_risk_affine_coefficients(**arguments)
    direct = exact_scheduled_vector_risk(rho=0.37, **arguments)["risk"]
    assert np.isclose(coefficients.evaluate(0.37), direct, atol=1e-13)


def test_budget_accounting_charges_each_participation_action() -> None:
    used = schedule_budget_summary(
        [1, 4, 2],
        message_overhead=2.0,
        per_agent_message=3.0,
        stride=4,
        delay=3,
    )
    assert used == {"updates": 3, "message": 27.0, "environment": 15}
    assert schedule_is_feasible(
        [1, 4, 2],
        message_budget=27.0,
        environment_budget=15,
        message_overhead=2.0,
        per_agent_message=3.0,
        stride=4,
        delay=3,
    )
    assert not schedule_is_feasible(
        [1, 4, 2],
        message_budget=26.99,
        environment_budget=15,
        message_overhead=2.0,
        per_agent_message=3.0,
        stride=4,
        delay=3,
    )


def test_robust_choice_requires_uniform_post_probe_improvement() -> None:
    risks = {
        "fallback": AffineRisk(1.0, 0.0),
        "crossing": AffineRisk(0.7, 0.8),
        "safe": AffineRisk(0.8, 0.05),
    }
    assert (
        robust_post_probe_choice(
            risks,
            fallback="fallback",
            rho_lower=0.0,
            rho_upper=1.0,
            improvement_margin=0.1,
        )
        == "safe"
    )
    assert (
        robust_post_probe_choice(
            {"fallback": risks["fallback"], "crossing": risks["crossing"]},
            fallback="fallback",
            rho_lower=0.0,
            rho_upper=1.0,
            improvement_margin=0.0,
        )
        == "fallback"
    )


def test_fixed_sample_interval_shrinks_and_clips() -> None:
    broad = hoeffding_interval(0.95, trials=10, alpha=0.05)
    narrow = hoeffding_interval(0.95, trials=10_000, alpha=0.05)
    assert broad[1] == 1.0
    assert narrow[0] > broad[0]
    assert narrow[1] <= 1.0


def test_monte_carlo_common_private_ar_matches_exact_risk() -> None:
    rng = np.random.default_rng(20260803)
    repetitions = 120_000
    schedule = np.asarray([1, 3, 2, 4], dtype=int)
    maximum_agents = int(schedule.max())
    rho = 0.4
    coefficient = 0.35
    innovation_scale = np.sqrt(1.0 - coefficient**2)
    common = rng.normal(size=repetitions)
    private = rng.normal(size=(repetitions, maximum_agents))
    error = np.full(repetitions, 1.25)
    for q in schedule:
        aggregate = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * private[:, :q].mean(axis=1)
        error = error - 0.09 * 0.7 * error + 0.09 * aggregate
        common = coefficient * common + innovation_scale * rng.normal(size=repetitions)
        private = coefficient * private + innovation_scale * rng.normal(
            size=(repetitions, maximum_agents)
        )
    empirical = float(np.mean(error**2))
    exact = exact_scheduled_vector_risk(
        initial_history=np.asarray([[1.25]]),
        drift=np.asarray([[0.7]]),
        step_size=0.09,
        delay=0,
        q_schedule=schedule,
        rho=rho,
        base_lag_covariances=base_ar_lags(len(schedule), 1.0, coefficient),
    )["risk"]
    assert np.isclose(empirical, exact, rtol=0.012, atol=0.002)
