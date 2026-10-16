"""A bounded RL-interface qualification, not a performance experiment."""

import itertools

import numpy as np
import pytest

from rl_collaboration_interface_audit import (
    delayed_consequence_counterexample, dirichlet_operator, history_mismatch,
    markov_td_risk_gramian, nonreversible_counterexample,
    oracle_line_transfer, oracle_policy_cost_and_advantage,
    report, residual_ranking_counterexample, return_moments,
    td_label_counterexample, value_oracle,
)


def enumerate_returns(p, reward, gamma, horizon, initial=0):
    values, probabilities = [], []
    for tail in itertools.product(range(len(p)), repeat=horizon-1):
        states = (initial,) + tail
        prob, value = 1., 0.
        for t, state in enumerate(states):
            value += gamma**t*reward[state]
            if t:
                prob *= p[states[t-1], state]
        values.append(value)
        probabilities.append(prob)
    return np.array(values), np.array(probabilities)


def test_td_label_bias_changes_contrast_sign():
    r = td_label_counterexample()
    assert r["executed_value_risk_contrast"] < 0
    assert r["incorrect_unbiased_label_prediction"] > 0
    assert abs(r["identity_residual_after_correction"]) < 1e-12
    np.testing.assert_allclose(r["missing_bootstrap_bias_term"], -115.2)


@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_finite_return_moments_by_exhaustive_paths(horizon):
    p = np.array([[.7, .3], [.2, .8]])
    reward, gamma = np.array([0., 1.]), .9
    mean, second = return_moments(p, reward, gamma, horizon)
    for state in range(2):
        returns, probability = enumerate_returns(p, reward, gamma, horizon, state)
        np.testing.assert_allclose(probability.sum(), 1)
        np.testing.assert_allclose(probability@returns, mean[state], atol=1e-14)
        np.testing.assert_allclose(probability@(returns**2), second[state], atol=1e-14)


def test_bounded_mc_label_contrast_and_subgaussian_mgf():
    p = np.array([[.7, .3], [.2, .8]])
    gamma, h = .9, 5
    values, probability = enumerate_returns(p, np.array([0., 1.]), gamma, h)
    theta = probability @ values
    z, local, a = 2., .3, .8
    x, s = a*z+(1-a)*values, a*local+(1-a)*values
    true_contrast = probability@((x-theta)**2-(s-theta)**2)
    loss_contrast = a*a*(probability@((z-values)**2-(local-values)**2))
    np.testing.assert_allclose(true_contrast, loss_contrast, atol=1e-14)
    interval_width = (1-gamma**h)/(1-gamma)
    for beta in [-3., -.1, 0., .7, 3.]:
        mgf = probability @ np.exp(beta*(values-theta))
        assert mgf <= np.exp(beta*beta*interval_width**2/8) + 1e-12


@pytest.mark.parametrize("horizon", [1, 8, 32])
def test_truncation_bias_is_not_sampling_error(horizon):
    p = np.array([[.7, .3], [.2, .8]])
    reward, gamma = np.array([0., 1.]), .9
    mean, _ = return_moments(p, reward, gamma, horizon)
    truth = value_oracle(p, reward, gamma)
    tail = -gamma**horizon*np.linalg.matrix_power(p, horizon)@truth
    np.testing.assert_allclose(mean-truth, tail, atol=1e-13)
    assert np.max(np.abs(tail)) <= gamma**horizon/(1-gamma)


@pytest.mark.parametrize("horizon", [1, 2, 32, 128])
def test_feedback_changes_the_counterfactual_comparator(horizon):
    r = history_mismatch(horizon)
    np.testing.assert_allclose(r["gap"], r["closed_form_gap"], atol=1e-13)
    if horizon > 1:
        assert r["gap"] > 0


def test_dirichlet_representation_has_a_reversibility_boundary():
    p = np.array([[.7, .3], [.2, .8]])
    symmetric, mu = dirichlet_operator(p, .9)
    np.testing.assert_allclose(symmetric, np.diag(mu)@(np.eye(2)-.9*p), atol=1e-14)
    nonrev = nonreversible_counterexample()
    a = np.array(nonrev["A"])
    np.testing.assert_allclose(nonrev["symmetric_dirichlet"], (a+a.T)/2, atol=1e-14)
    assert nonrev["skew_operator_norm"] > 0
    assert np.linalg.norm(nonrev["gradient_at_true_value"]) > 0
    assert nonrev["value_error_of_symmetric_surrogate_minimizer"] > 0


def test_mean_bellman_residual_is_not_value_error_ordering():
    r = residual_ranking_counterexample()
    assert r["value_mse"][0] > r["value_mse"][1]
    assert r["squared_mean_Bellman_residual"][0] < r["squared_mean_Bellman_residual"][1]


def test_qualification_does_not_authorize_efficacy():
    r = report()
    assert not r["decision"]["new_efficacy_pilot_authorized"]
    assert r["decision"]["reference_retained_as_baseline"]


def test_markov_jump_risk_metric_against_exhaustive_paths():
    p = np.array([[.9, .1], [.4, .6]])
    gamma, step, horizon = .9, .5, 4
    q = markov_td_risk_gramian(p, gamma, step, horizon)
    eye = np.eye(2)
    for initial in range(2):
        exact = np.zeros((2, 2))
        for tail in itertools.product(range(2), repeat=horizon-1):
            states = (initial,) + tail
            probability, propagator, costs = 1., eye.copy(), eye.copy()
            for t in range(horizon-1):
                s, j = states[t], states[t+1]
                probability *= p[s, j]
                b = eye-step*np.outer(eye[s], eye[s]-gamma*eye[j])
                propagator = b @ propagator
                costs += propagator.T @ propagator
            exact += probability*costs
        np.testing.assert_allclose(q[initial], exact, atol=1e-13)


def test_immediate_improvement_can_increase_future_td_risk():
    r = delayed_consequence_counterexample()
    np.testing.assert_allclose(r["immediate_squared_error_change"], -.36)
    assert r["expected_cumulative_squared_error_change"] > 0
    assert r["oracle_future_risk_QP_advantage"] < r["future_advantage_of_immediate_error_QP"]
    assert 0 < r["oracle_future_risk_QP_beta"] < 1


def test_oracle_quadratic_line_solution():
    q = np.diag([1., 4.])
    e, donor = np.array([1., 0.]), np.array([0., .8])
    beta, advantage = oracle_line_transfer(q, e, donor)
    d = donor-e
    assert abs(d@q@(e+beta*d)) < 1e-14
    assert advantage <= 0
    for alternative in [0., .1, .5, 1.]:
        x = e+alternative*d
        assert advantage <= x@q@x-e@q@e + 1e-14


@pytest.mark.parametrize("state", [0, 1])
def test_counterfactual_performance_difference_by_full_tree(state):
    p = np.array([[.9, .1], [.4, .6]])
    gamma, step, horizon = .9, .5, 5
    e, donor = np.array([-1., .3]), np.array([.3, .8])
    baseline = e@markov_td_risk_gramian(p, gamma, step, horizon)[state]@e
    policy, advantages = oracle_policy_cost_and_advantage(p, gamma, step, e, donor, state, horizon)
    np.testing.assert_allclose(policy-baseline, advantages, atol=1e-12)
    assert policy <= baseline + 1e-12
