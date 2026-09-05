import numpy as np
import pytest

from experiments.dependence_delay_linear.t035_scalar_phase_theorem import (
    exact_scalar_risk,
)
from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    covariance_impulse_upper_bound,
    delayed_vector_companion,
    dual_budget_updates,
    equicorrelated_ar_lag_covariances,
    exact_vector_risk,
)


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_vector_formula_reduces_to_t035_scalar(delay: int) -> None:
    updates = 17
    lags = equicorrelated_ar_lag_covariances(
        horizon=updates,
        single_agent_covariance=np.array([[1.7]]),
        q=8,
        rho=0.3,
        markov_lambda=0.6,
    )
    vector = exact_vector_risk(
        initial_history=np.full((delay + 1, 1), 1.2),
        drift=np.array([[0.8]]),
        step_size=0.11,
        delay=delay,
        updates=updates,
        lag_covariances=lags,
    )
    scalar = exact_scalar_risk(
        initial_error=1.2,
        mu=0.8,
        step_size=0.11,
        delay=delay,
        updates=updates,
        single_variance=1.7,
        q=8,
        rho=0.3,
        markov_lambda=0.6,
    )
    assert vector["risk"] == pytest.approx(scalar["risk"], abs=1e-11)


def test_vector_iid_no_delay_matches_coordinate_closed_forms() -> None:
    drift = np.diag([0.5, 1.1])
    covariance = np.diag([1.2, 0.7])
    updates = 12
    step = 0.15
    lags = np.zeros((2 * updates - 1, 2, 2))
    lags[updates - 1] = covariance
    result = exact_vector_risk(
        initial_history=np.array([[2.0, -1.0]]),
        drift=drift,
        step_size=step,
        delay=0,
        updates=updates,
        lag_covariances=lags,
    )
    expected = 0.0
    for initial, mu, variance in zip([2.0, -1.0], [0.5, 1.1], [1.2, 0.7]):
        contraction = 1.0 - step * mu
        expected += contraction ** (2 * updates) * initial**2
        expected += (
            step**2
            * variance
            * (1.0 - contraction ** (2 * updates))
            / (1.0 - contraction**2)
        )
    assert result["risk"] == pytest.approx(expected)


def test_equicorrelated_lags_recover_effective_agent_factor() -> None:
    covariance = np.array([[2.0, 0.3], [0.3, 1.0]])
    q = 10
    rho = 0.4
    lags = equicorrelated_ar_lag_covariances(
        horizon=4,
        single_agent_covariance=covariance,
        q=q,
        rho=rho,
        markov_lambda=0.5,
    )
    np.testing.assert_allclose(
        lags[3], (rho + (1.0 - rho) / q) * covariance
    )
    np.testing.assert_allclose(lags[4], 0.5 * lags[3])


def test_trace_norm_bound_dominates_exact_noise_risk() -> None:
    updates = 10
    drift = np.array([[0.9, 0.15], [0.15, 0.7]])
    base = np.array([[1.0, 0.2], [0.2, 0.8]])
    lags = equicorrelated_ar_lag_covariances(
        horizon=updates,
        single_agent_covariance=base,
        q=4,
        rho=0.25,
        markov_lambda=0.7,
    )
    result = exact_vector_risk(
        initial_history=np.tile(np.array([[1.0, -0.5]]), (2, 1)),
        drift=drift,
        step_size=0.08,
        delay=1,
        updates=updates,
        lag_covariances=lags,
        risk_matrix=np.diag([2.0, 0.5]),
    )
    bound = covariance_impulse_upper_bound(
        drift=drift,
        step_size=0.08,
        delay=1,
        updates=updates,
        lag_covariances=lags,
        risk_matrix=np.diag([2.0, 0.5]),
    )
    assert 0.0 <= result["noise_risk"] <= bound + 1e-12


def test_dual_budget_charges_delay_only_to_environment() -> None:
    assert dual_budget_updates(
        message_budget=100,
        environment_budget=83,
        message_cost=7,
        stride=4,
        delay=3,
    ) == 14
    assert dual_budget_updates(
        message_budget=100,
        environment_budget=2,
        message_cost=7,
        stride=4,
        delay=3,
    ) == 0


def test_zero_updates_preserves_vector_bias() -> None:
    result = exact_vector_risk(
        initial_history=np.array([[1.0, -2.0], [1.0, -2.0]]),
        drift=np.eye(2),
        step_size=0.1,
        delay=1,
        updates=0,
        lag_covariances=None,
        risk_matrix=np.diag([2.0, 3.0]),
    )
    assert result["risk"] == pytest.approx(14.0)
    np.testing.assert_allclose(result["covariance"], np.zeros((2, 2)))


def test_delay_companion_has_expected_block_shift() -> None:
    drift = np.diag([1.0, 2.0])
    companion = delayed_vector_companion(drift, 0.1, 2)
    np.testing.assert_allclose(companion[2:, :-2], np.eye(4))
    np.testing.assert_allclose(companion[:2, 4:6], -0.1 * drift)
