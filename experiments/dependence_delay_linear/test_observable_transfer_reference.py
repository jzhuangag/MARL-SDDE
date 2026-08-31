"""Exact estimator/controller qualification. No Monte Carlo or formal seeds."""

import inspect

import numpy as np
import pytest

from observable_transfer_reference import (
    ProbeAccumulator, ProbeSpec, confidence_terms, cost_contract, encode_probe,
    exact_cost_contrast, exact_probe_mean, projection_leakage_witness,
    project_transfer_simplex, qp_controller, qualification_report, robust_qp, scalar_controller,
)
from rl_collaboration_interface_audit import return_moments, value_oracle


def small_spec(horizon=3, length=3):
    return ProbeSpec(np.array([.2, -.1]), np.array([[.4, -.3], [-.2, .5]]),
                     0, .6, .4, horizon, length, .5)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_random_time_identity_by_full_enumeration(horizon):
    p, r = np.array([[.8, .2], [.3, .7]]), np.array([0., .5])
    spec = small_spec(horizon)
    G, g, probability = exact_probe_mean(spec, p, r)
    finite_target, _ = return_moments(p, r, spec.discount, spec.return_length)
    np.testing.assert_allclose(probability, 1., atol=1e-13)
    assert np.linalg.eigvalsh(G).min() >= -1e-12
    for beta in [np.zeros(2), np.array([1., 0.]), np.array([.2, .6])]:
        truth = exact_cost_contrast(spec, p, r, beta, finite_target)
        np.testing.assert_allclose(beta@G@beta+2*g@beta, truth, atol=1e-12)


@pytest.mark.parametrize("length", [1, 2, 4])
def test_infinite_value_bias_is_explicitly_bounded(length):
    p, r = np.array([[.8, .2], [.3, .7]]), np.array([0., .5])
    spec = small_spec(3, length)
    G, g, _ = exact_probe_mean(spec, p, r)
    target, beta = value_oracle(p, r, spec.discount), np.array([.2, .6])
    true = exact_cost_contrast(spec, p, r, beta, target)
    upper_bias = 2*confidence_terms(spec, 128, .01)["return_bias_radius"]*sum(beta)
    assert abs(true-(beta@G@beta+2*g@beta)) <= upper_bias+1e-12


def test_streaming_estimator_requires_no_model_or_true_value():
    assert set(inspect.signature(encode_probe).parameters) == {
        "spec", "training_transitions", "evaluation_state", "return_transitions"}
    spec = small_spec()
    sample = encode_probe(spec, [(0, 0., 1)], 1, [(1, .5, 1)]*3)
    accumulator = ProbeAccumulator(spec)
    accumulator.add(0, sample)
    np.testing.assert_allclose(accumulator.means()[0], sample["G"])
    assert (accumulator.training, accumulator.returns, accumulator.resets) == (1, 3, 2)
    with pytest.raises(ValueError, match="unique"):
        accumulator.add(0, sample)
    with pytest.raises(ValueError, match="context"):
        accumulator.add(1, {**sample, "context_hash": "changed"})


def test_self_transition_sensitivity_uses_preupdate_values():
    spec = ProbeSpec(np.array([1.]), np.array([[-.5]]), 0, .9, .5, 3, 2, 1.)
    sample = encode_probe(spec, [(0, 0., 0)], 0, [(0, 0., 0)]*2)
    np.testing.assert_allclose(sample["G"], [[3*(.475**2)]])
    np.testing.assert_allclose(sample["g"], [3*(-.475)*.95])


@pytest.mark.parametrize("bad", [
    {"training_transitions": [(1, 0., 0)]},
    {"training_transitions": [(0, 2., 1)]},
    {"training_transitions": [(0, 0., 0)]*3},
    {"return_transitions": [(0, 0., 1)]},
    {"return_transitions": [(1, 0., 1)]*3},
    {"evaluation_state": 5},
])
def test_invalid_observation_contract_rejected(bad):
    args = {"training_transitions": [], "evaluation_state": 0,
            "return_transitions": [(0, 0., 0)]*3}
    args.update(bad)
    with pytest.raises(ValueError):
        encode_probe(small_spec(), **args)


def test_uniform_confidence_lift_preserves_convexity_and_zero_action():
    spec = small_spec()
    G = np.array([[1., .2], [.2, .5]])
    g = np.array([-.1, .2])
    terms = confidence_terms(spec, 256, .01)
    upper_G, upper_g = robust_qp(G, g, terms)
    assert np.linalg.eigvalsh(upper_G).min() >= 0
    for beta in [np.zeros(2), np.array([.3, .2]), np.array([0., 1.])]:
        manual = (beta@G@beta+2*g@beta+terms["gram_radius"]*sum(beta)**2
                  +2*(terms["linear_radius"]+terms["return_bias_radius"])*sum(beta))
        np.testing.assert_allclose(beta@upper_G@beta+2*upper_g@beta, manual)
    assert np.zeros(2)@upper_G@np.zeros(2)+2*upper_g@np.zeros(2) == 0


def test_scalar_controller_solves_the_robust_not_raw_quadratic():
    terms = {"gram_radius": .2, "linear_radius": .1, "return_bias_radius": .05}
    result = scalar_controller(1., -.5, terms)
    np.testing.assert_allclose(result["beta"], .35/1.2)
    assert result["upper_advantage"] < 0
    assert scalar_controller(1., -.1, terms)["beta"] == 0


def test_cost_charges_full_return_and_horizon_dependence():
    spec = ProbeSpec(np.array([1.]), np.array([[-.5]]), 0, .9, .5, 64, 32, 1.)
    cost = cost_contract(spec, 128)
    assert cost["expected_probe_transitions"] == 8128
    assert cost["worst_probe_transitions"] == 12160
    assert cost["expected_forward_TD_steps"] == 4032
    assert cost["reset_requests"] == 256


def test_projection_and_joint_controller_have_bounded_optimization_error():
    np.testing.assert_allclose(project_transfer_simplex([-.3, .1, .2]), [0., .1, .2])
    np.testing.assert_allclose(project_transfer_simplex([2., 1., -1.]), [1., 0., 0.])
    G, g = np.diag([1., 2.]), np.array([-.2, -.6])
    zero = {"gram_radius": 0., "linear_radius": 0., "return_bias_radius": 0.}
    out = qp_controller(G, g, zero, iterations=20)
    optimum = np.array([.2, .3])
    best = optimum@G@optimum+2*g@optimum
    assert out["upper_advantage"] >= best-1e-13
    assert out["upper_advantage"]-best <= out["optimization_gap_bound"]+1e-13
    assert np.min(out["beta"]) >= 0 and sum(out["beta"]) <= 1
    np.testing.assert_allclose(qp_controller(np.zeros((2, 2)), [-.2, .3], zero)["beta"], [1., 0.])


def test_projecting_directions_does_not_close_the_dynamics():
    result = projection_leakage_witness()
    np.testing.assert_allclose(result["missed_leakage"], .081, atol=1e-13)
    np.testing.assert_allclose(result["missed_leakage"], result["exact_missed_leakage"])


def test_qualification_rejects_compression_claim_without_global_impossibility():
    decision = qualification_report()["decision"]
    assert decision["cheap_long_horizon_final_candidate"].startswith("reject")
    assert not decision["general_impossibility_claim"]
    assert not decision["new_efficacy_pilot_authorized"]
