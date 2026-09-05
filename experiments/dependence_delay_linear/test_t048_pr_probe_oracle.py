"""Tests for the scheduled PR and probe-cost oracle results in T-048."""

from __future__ import annotations

import numpy as np

from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    equicorrelated_ar_lag_covariances,
)
from experiments.dependence_delay_linear.t044_pr_averaged_phase import (
    exact_pr_averaged_vector_risk,
)
from experiments.dependence_delay_linear.t047_scheduled_participation import (
    AffineRisk,
)
from experiments.dependence_delay_linear.t048_pr_probe_oracle import (
    exact_scheduled_pr_averaged_vector_risk,
    generic_post_probe_oracle_bound,
    mixing_corrected_icc_interval,
    probe_cost_oracle_certificate,
    robust_minimax_choice,
    scheduled_pr_risk_affine_coefficients,
)


def base_ar_lags(updates: int, variance: float, coefficient: float) -> np.ndarray:
    lags = np.arange(-(updates - 1), updates)
    return np.asarray(
        [[[variance * coefficient ** abs(int(lag))]] for lag in lags], dtype=float
    )


def test_constant_schedule_matches_t044_pr_identity() -> None:
    updates = 9
    q = 4
    rho = 0.3
    coefficient = 0.45
    standard = exact_pr_averaged_vector_risk(
        initial_history=np.asarray([[1.7], [1.7]]),
        drift=np.asarray([[0.6]]),
        step_size=0.1,
        delay=1,
        updates=updates,
        burn_in=3,
        lag_covariances=equicorrelated_ar_lag_covariances(
            horizon=updates,
            single_agent_covariance=np.asarray([[1.4]]),
            q=q,
            rho=rho,
            markov_lambda=coefficient,
        ),
    )
    scheduled = exact_scheduled_pr_averaged_vector_risk(
        initial_history=np.asarray([[1.7], [1.7]]),
        drift=np.asarray([[0.6]]),
        step_size=0.1,
        delay=1,
        q_schedule=[q] * updates,
        burn_in=3,
        rho=rho,
        base_lag_covariances=base_ar_lags(updates, 1.4, coefficient),
    )
    assert np.allclose(scheduled["mean"], standard["mean"], atol=1e-13)
    assert np.allclose(
        scheduled["covariance"], standard["covariance"], atol=1e-13
    )
    assert np.isclose(scheduled["risk"], standard["risk"], atol=1e-13)


def test_scheduled_pr_risk_is_affine_in_rho() -> None:
    arguments = {
        "initial_history": np.asarray([[1.0]]),
        "drift": np.asarray([[0.7]]),
        "step_size": 0.08,
        "delay": 0,
        "q_schedule": [1, 2, 8, 4, 2, 1],
        "burn_in": 2,
        "base_lag_covariances": base_ar_lags(6, 1.1, 0.35),
    }
    affine = scheduled_pr_risk_affine_coefficients(**arguments)
    exact = exact_scheduled_pr_averaged_vector_risk(rho=0.43, **arguments)
    assert np.isclose(affine.evaluate(0.43), exact["risk"], atol=1e-13)


def test_fast_affine_coefficients_match_both_direct_endpoints() -> None:
    rng = np.random.default_rng(4801)
    for delay in (0, 1, 3):
        for dimension in (1, 2, 3):
            updates = 6
            raw = rng.normal(size=(dimension, dimension))
            drift = 0.15 * np.eye(dimension) + 0.02 * (raw + raw.T)
            minimum = np.min(np.linalg.eigvalsh(drift))
            if minimum <= 0.05:
                drift += (0.06 - minimum) * np.eye(dimension)
            weight_raw = rng.normal(size=(dimension, dimension))
            weight = weight_raw.T @ weight_raw + np.eye(dimension)
            base = np.asarray(
                [
                    (0.3 ** abs(lag)) * np.eye(dimension)
                    for lag in range(-(updates - 1), updates)
                ]
            )
            arguments = {
                "initial_history": rng.normal(
                    size=(delay + 1, dimension)
                ),
                "drift": drift,
                "step_size": 0.04,
                "delay": delay,
                "q_schedule": [1, 3, 2, 8, 4, 1],
                "burn_in": 2,
                "base_lag_covariances": base,
                "risk_matrix": weight,
            }
            affine = scheduled_pr_risk_affine_coefficients(**arguments)
            at_zero = exact_scheduled_pr_averaged_vector_risk(
                rho=0.0, **arguments
            )["risk"]
            at_one = exact_scheduled_pr_averaged_vector_risk(
                rho=1.0, **arguments
            )["risk"]
            assert np.isclose(affine.intercept, at_zero, atol=2e-12)
            assert np.isclose(
                affine.intercept + affine.slope, at_one, atol=2e-12
            )


def test_monte_carlo_scheduled_pr_matches_exact_risk() -> None:
    rng = np.random.default_rng(20260803)
    repetitions = 100_000
    schedule = np.asarray([1, 3, 2, 4, 2], dtype=int)
    maximum_agents = int(schedule.max())
    rho = 0.35
    coefficient = 0.3
    common = rng.normal(size=repetitions)
    private = rng.normal(size=(repetitions, maximum_agents))
    error = np.full(repetitions, 1.2)
    iterates = []
    scale = np.sqrt(1.0 - coefficient**2)
    for q in schedule:
        noise = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * np.mean(
            private[:, :q], axis=1
        )
        error = error - 0.09 * 0.65 * error + 0.09 * noise
        iterates.append(error.copy())
        common = coefficient * common + scale * rng.normal(size=repetitions)
        private = coefficient * private + scale * rng.normal(
            size=(repetitions, maximum_agents)
        )
    empirical = float(np.mean(np.mean(iterates[2:], axis=0) ** 2))
    exact = exact_scheduled_pr_averaged_vector_risk(
        initial_history=np.asarray([[1.2]]),
        drift=np.asarray([[0.65]]),
        step_size=0.09,
        delay=0,
        q_schedule=schedule,
        burn_in=2,
        rho=rho,
        base_lag_covariances=base_ar_lags(len(schedule), 1.0, coefficient),
    )["risk"]
    assert np.isclose(empirical, exact, rtol=0.015, atol=0.0015)


def _bounded_icc_probes(
    *, rng: np.random.Generator, trials: int, agents: int, rho: float
) -> np.ndarray:
    common_regime = rng.random(trials) < rho
    common = rng.choice([-1.0, 1.0], size=trials)
    private = rng.choice([-1.0, 1.0], size=(trials, agents))
    centered = np.where(common_regime[:, None], common[:, None], private)
    return 0.1 + 0.8 * centered


def test_mixing_corrected_interval_handles_unknown_mean_and_contains_icc() -> None:
    rho = 0.4
    probes = _bounded_icc_probes(
        rng=np.random.default_rng(47), trials=20_000, agents=4, rho=rho
    )
    certificate = mixing_corrected_icc_interval(
        probes,
        alpha=0.05,
        beta_constant=0.0,
        beta_rate=0.0,
        stride=1,
        variance_lower=0.5,
    )
    assert certificate.informative
    assert certificate.lower <= rho <= certificate.upper
    assert certificate.upper - certificate.lower < 0.12


def test_insufficient_mixing_budget_forces_vacuous_interval() -> None:
    probes = np.zeros((100, 3))
    certificate = mixing_corrected_icc_interval(
        probes,
        alpha=0.05,
        beta_constant=1.0,
        beta_rate=0.9,
        stride=1,
    )
    assert not certificate.informative
    assert (certificate.lower, certificate.upper) == (0.0, 1.0)
    assert np.isinf(certificate.radius)


def test_registered_checkpoint_spending_gives_nested_union_bound_inputs() -> None:
    probes = _bounded_icc_probes(
        rng=np.random.default_rng(48), trials=12_000, agents=4, rho=0.6
    )
    alpha_spending = [0.025, 0.0125, 0.00625]
    certificates = [
        mixing_corrected_icc_interval(
            probes[:trials],
            alpha=alpha,
            beta_constant=0.0,
            beta_rate=0.0,
            stride=1,
            variance_lower=0.5,
        )
        for trials, alpha in zip((3_000, 6_000, 12_000), alpha_spending)
    ]
    assert sum(alpha_spending) < 0.05
    assert all(
        certificate.lower <= 0.6 <= certificate.upper
        for certificate in certificates
    )
    assert certificates[-1].upper - certificates[-1].lower < 0.25


def test_robust_minimax_choice_uses_both_interval_endpoints() -> None:
    risks = {
        "small": AffineRisk(0.7, 0.8),
        "middle": AffineRisk(0.85, 0.1),
        "large": AffineRisk(1.0, -0.4),
    }
    assert robust_minimax_choice(risks, rho_lower=0.0, rho_upper=1.0) == "middle"
    assert robust_minimax_choice(risks, rho_lower=0.0, rho_upper=0.1) == "small"


def test_oracle_certificate_includes_probe_cost_and_failure_event() -> None:
    full = {
        "fixed-small": AffineRisk(0.7, 0.6),
        "fixed-large": AffineRisk(1.0, -0.3),
    }
    post = {
        "adaptive": AffineRisk(0.8, 0.1),
    }
    certificate = probe_cost_oracle_certificate(
        full_budget_risks=full,
        post_probe_risks=post,
        selected="adaptive",
        rho_lower=0.2,
        rho_upper=0.8,
        failure_probability=0.05,
        risk_cap=2.0,
        baseline="fixed-small",
    )
    direct = max(
        post["adaptive"].evaluate(rho)
        - min(risk.evaluate(rho) for risk in full.values())
        for rho in (0.2, 0.8)
    )
    assert np.isclose(certificate.conditional_full_oracle_excess, direct)
    assert np.isclose(
        certificate.expected_full_oracle_excess,
        direct + 0.05 * max(2.0 - direct, 0.0),
    )
    assert certificate.conditional_baseline_difference is not None


def test_generic_robust_oracle_bound_uses_largest_slope() -> None:
    risks = {
        "a": AffineRisk(1.0, 0.2),
        "b": AffineRisk(0.9, -0.5),
    }
    assert np.isclose(generic_post_probe_oracle_bound(risks, interval_width=0.3), 0.15)


def test_invalid_probe_values_are_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        mixing_corrected_icc_interval(
            np.asarray([[0.0, 1.2], [0.1, 0.2]]),
            alpha=0.05,
            beta_constant=0.0,
            beta_rate=0.0,
            stride=1,
        )


def test_incompatible_empirical_moments_return_vacuous_certificate() -> None:
    certificate = mixing_corrected_icc_interval(
        np.zeros((20_000, 3)),
        alpha=0.05,
        beta_constant=0.0,
        beta_rate=0.0,
        stride=1,
        variance_lower=0.8,
    )
    assert not certificate.informative
    assert (certificate.lower, certificate.upper) == (0.0, 1.0)


def test_negative_conditional_difference_has_valid_failure_adjustment() -> None:
    certificate = probe_cost_oracle_certificate(
        full_budget_risks={"baseline": AffineRisk(1.0, 0.0)},
        post_probe_risks={"selected": AffineRisk(0.5, 0.0)},
        selected="selected",
        rho_lower=0.0,
        rho_upper=1.0,
        failure_probability=0.1,
        risk_cap=2.0,
        baseline="baseline",
    )
    assert np.isclose(certificate.conditional_full_oracle_excess, -0.5)
    assert np.isclose(certificate.expected_full_oracle_excess, -0.25)
    assert np.isclose(certificate.expected_baseline_difference, -0.25)
