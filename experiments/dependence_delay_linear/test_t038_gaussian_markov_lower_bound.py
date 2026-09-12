import numpy as np
import pytest

from experiments.dependence_delay_linear.t038_gaussian_markov_lower_bound import (
    ProbeAction,
    dense_fisher_information,
    fixed_sequence_minimax_risk,
    posterior_covariance,
    predictable_dual_budget_lower_bound,
    predictable_dual_budget_minimax_risk,
)


def test_kalman_covariance_matches_dense_irregular_fisher_information() -> None:
    actions = [ProbeAction(1, 1), ProbeAction(4, 3), ProbeAction(2, 2)]
    prior = 7.0
    information = dense_fisher_information(
        actions=actions,
        common_variance=0.8,
        private_variance=1.2,
        markov_lambda=0.7,
    )
    kalman = posterior_covariance(
        actions=actions,
        prior_variance=prior,
        common_variance=0.8,
        private_variance=1.2,
        markov_lambda=0.7,
    )
    assert kalman[0, 0] == pytest.approx(1.0 / (1.0 / prior + information))


def test_diffuse_prior_limit_is_fixed_sequence_minimax_risk() -> None:
    actions = [ProbeAction(4, 2)] * 6
    minimax = fixed_sequence_minimax_risk(
        actions=actions,
        common_variance=0.5,
        private_variance=0.5,
        markov_lambda=0.6,
    )
    posterior = posterior_covariance(
        actions=actions,
        prior_variance=1e12,
        common_variance=0.5,
        private_variance=0.5,
        markov_lambda=0.6,
    )[0, 0]
    assert posterior == pytest.approx(minimax, rel=2e-5)


def test_same_time_agent_gain_saturates_under_common_noise() -> None:
    q1 = fixed_sequence_minimax_risk(
        actions=[ProbeAction(1, 1)],
        common_variance=1.0,
        private_variance=1.0,
        markov_lambda=0.8,
    )
    q64 = fixed_sequence_minimax_risk(
        actions=[ProbeAction(64, 1)],
        common_variance=1.0,
        private_variance=1.0,
        markov_lambda=0.8,
    )
    assert q1 == pytest.approx(2.0)
    assert q64 == pytest.approx(1.0 + 1.0 / 64.0)
    assert q1 / q64 < 2.0


def test_stride_can_increase_information_for_persistent_common_factor() -> None:
    dense = dense_fisher_information(
        actions=[ProbeAction(4, 1)] * 5,
        common_variance=1.0,
        private_variance=0.2,
        markov_lambda=0.95,
    )
    spaced = dense_fisher_information(
        actions=[ProbeAction(4, 4)] * 5,
        common_variance=1.0,
        private_variance=0.2,
        markov_lambda=0.95,
    )
    assert spaced > dense


def test_predictable_lower_bound_respects_both_budgets_and_delay() -> None:
    result = predictable_dual_budget_lower_bound(
        action_catalogue=[ProbeAction(1, 1), ProbeAction(4, 2)],
        message_budget=12,
        environment_budget=7,
        overhead=1,
        delay=3,
        prior_variance=100.0,
        common_variance=0.7,
        private_variance=0.3,
        markov_lambda=0.8,
    )
    sequence = result["best_action_sequence"]
    assert sum(1 + action.q for action in sequence) <= 12
    assert sum(action.stride for action in sequence) + 3 <= 7
    assert result["environment_available_after_delay"] == 4


def test_variable_action_optimum_dominates_each_feasible_fixed_path() -> None:
    catalogue = [ProbeAction(1, 1), ProbeAction(3, 2)]
    result = predictable_dual_budget_lower_bound(
        action_catalogue=catalogue,
        message_budget=12,
        environment_budget=8,
        overhead=1,
        delay=1,
        prior_variance=50.0,
        common_variance=0.9,
        private_variance=0.1,
        markov_lambda=0.9,
    )
    for action in catalogue:
        count = min(12 // (1 + action.q), 7 // action.stride)
        if count:
            fixed = posterior_covariance(
                actions=[action] * count,
                prior_variance=50.0,
                common_variance=0.9,
                private_variance=0.1,
                markov_lambda=0.9,
            )[0, 0]
            assert result["bayes_risk_lower_bound"] <= fixed + 1e-12


def test_dense_information_is_zero_without_observations() -> None:
    assert dense_fisher_information(
        actions=[],
        common_variance=1.0,
        private_variance=1.0,
        markov_lambda=0.5,
    ) == 0.0


def test_exact_minimax_design_equals_gls_risk_of_selected_sequence() -> None:
    result = predictable_dual_budget_minimax_risk(
        action_catalogue=[ProbeAction(1, 1), ProbeAction(3, 2)],
        message_budget=10,
        environment_budget=7,
        overhead=1,
        delay=1,
        common_variance=0.6,
        private_variance=0.4,
        markov_lambda=0.75,
    )
    selected_risk = fixed_sequence_minimax_risk(
        actions=result["best_action_sequence"],
        common_variance=0.6,
        private_variance=0.4,
        markov_lambda=0.75,
    )
    assert result["minimax_risk"] == pytest.approx(selected_risk)
