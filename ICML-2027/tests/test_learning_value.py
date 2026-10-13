import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments"))
from learning_value import (action_table, calibration_cost, choose, covers,
                            fit_calibration, forecast, observe_calibration,
                            risk_coefficients, selection_bound, simulate_errors,
                            true_risk)


def test_exact_white_noise_formula():
    step, n = .17, 37
    bias, noise, _ = risk_coefficients(step, 0, 0, n)
    assert bias == pytest.approx((1-step)**(2*n))
    assert noise == pytest.approx(step/(2-step)*(1-(1-step)**(2*n)))


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_lifted_covariance_matches_direct_iteration(delay):
    step, rate, n = .11, .83, 83
    d = delay+2
    transition = np.zeros((d, d))
    transition[0, 0] = 1
    transition[0, delay] -= step
    transition[0, -1] = -step
    if delay:
        transition[1:delay+1, :delay] = np.eye(delay)
    transition[-1, -1] = rate
    covariance = np.zeros((d, d)); covariance[-1, -1] = 1
    forcing = np.zeros((d, d)); forcing[-1, -1] = 1-rate**2
    mean = np.ones(d); mean[-1] = 0
    for _ in range(n):
        mean = transition@mean
        covariance = transition@covariance@transition.T+forcing
    bias, noise, _ = risk_coefficients(step, rate, delay, n)
    assert bias == pytest.approx(mean[0]**2, abs=1e-12)
    assert noise == pytest.approx(covariance[0, 0], abs=1e-12)


def test_exact_resource_accounting_and_complete_catalogue():
    pm, pe, pt = calibration_cost(8, 32, 4)
    assert (pm, pe, pt) == (396, 264, 33)
    table = action_table(qs=[1, 2, 4, 8], spacings=[1, 4], steps=[.01, .1], persistence=.8,
                         delay=2, message_budget=5000, actor_budget=4000, overhead=4,
                         probe_messages=pm, probe_actors=pe, probe_ticks=pt)
    assert {a.q for a in table} == {1, 2, 4, 8}
    assert all(a.messages <= 5000 and a.actor_transitions <= 4000 for a in table)
    for a in table:
        assert a.actor_transitions == pe+a.updates*a.q*a.spacing


def test_whitening_contrasts_and_exact_confidence_interface():
    values = observe_calibration(np.random.default_rng(1729), target=1.2, rho=.4,
                                persistence=.9, workers=8, observations=4096)
    certificate = fit_calibration(values, persistence=.9, delta=.001)
    assert covers(certificate, target=1.2, rho=.4)
    assert certificate.variance(8)[0] == certificate.average_variance
    assert certificate.variance(1)[0] >= certificate.variance(8)[0]


def test_shared_singular_case_is_supported():
    values = observe_calibration(np.random.default_rng(19), target=.7, rho=1.,
                                persistence=.6, workers=8, observations=512)
    certificate = fit_calibration(values, persistence=.6, delta=.001)
    assert certificate.private_variance < 1e-27
    assert covers(certificate, target=.7, rho=1.)


def test_coverage_transfers_to_all_actions_and_oracle_regret():
    values = observe_calibration(np.random.default_rng(32), target=1., rho=.5,
                                persistence=.6, workers=8, observations=4096)
    certificate = fit_calibration(values, persistence=.6, delta=.001)
    table = action_table(qs=[1,2,4,8], spacings=[1,4], steps=[.003,.01,.03], persistence=.6,
                         delay=2, message_budget=3000, actor_budget=3000, overhead=4)
    assert covers(certificate, target=1., rho=.5)
    for a in table:
        _, lower, upper = forecast(a, certificate)
        assert lower-1e-12 <= true_risk(a, target=1., rho=.5) <= upper+1e-12
    selected = choose(table, certificate)
    regret = true_risk(selected, target=1., rho=.5)-min(true_risk(a,target=1.,rho=.5) for a in table)
    assert regret <= selection_bound(table, certificate, selected)+1e-12


def test_unstable_action_rejected():
    with pytest.raises(ValueError, match="unstable"):
        risk_coefficients(.9, .5, 2, 10)


def test_simulator_matches_exact_population_risk():
    action = action_table(qs=[2], spacings=[1], steps=[.1], persistence=.8,
                          delay=2, message_budget=250, actor_budget=100, overhead=3)[0]
    errors = simulate_errors(np.random.default_rng(405), action, target=1., rho=.5,
                             persistence=.8, delay=2, replicates=30000,
                             checkpoints=np.array([action.updates]))[-1]
    observed = np.square(errors).mean()
    standard_error = np.square(errors).std(ddof=1)/np.sqrt(len(errors))
    assert abs(observed-true_risk(action,target=1.,rho=.5)) <= 5*standard_error
