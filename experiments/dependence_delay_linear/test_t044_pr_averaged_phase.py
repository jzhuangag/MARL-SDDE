import numpy as np
import pytest

from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    equicorrelated_ar_lag_covariances,
    exact_vector_risk,
)
from experiments.dependence_delay_linear.t044_pr_averaged_phase import (
    discrete_pr_proxy_optimum,
    exact_pr_averaged_scalar_risk,
    exact_pr_averaged_vector_risk,
    pr_message_proxy,
)


def test_one_update_average_equals_terminal_risk() -> None:
    lags = equicorrelated_ar_lag_covariances(
        horizon=1,
        single_agent_covariance=np.array([[1.7]]),
        q=4,
        rho=0.3,
        markov_lambda=0.6,
    )
    averaged = exact_pr_averaged_vector_risk(
        initial_history=np.array([[1.2]]),
        drift=np.array([[0.8]]),
        step_size=0.1,
        delay=0,
        updates=1,
        burn_in=0,
        lag_covariances=lags,
    )
    terminal = exact_vector_risk(
        initial_history=np.array([[1.2]]),
        drift=np.array([[0.8]]),
        step_size=0.1,
        delay=0,
        updates=1,
        lag_covariances=lags,
    )
    assert averaged["risk"] == pytest.approx(terminal["risk"], abs=1e-12)


def test_deterministic_average_matches_direct_recursion() -> None:
    updates = 7
    drift = np.array([[0.7, 0.1], [0.0, 0.5]])
    initial = np.array([1.0, -0.4])
    lags = np.zeros((2 * updates - 1, 2, 2))
    result = exact_pr_averaged_vector_risk(
        initial_history=initial[None, :],
        drift=drift,
        step_size=0.12,
        delay=0,
        updates=updates,
        burn_in=2,
        lag_covariances=lags,
    )
    iterate = initial.copy()
    values = []
    for _ in range(updates):
        iterate = iterate - 0.12 * drift @ iterate
        values.append(iterate.copy())
    expected = np.mean(values[2:], axis=0)
    np.testing.assert_allclose(result["mean"], expected, atol=1e-12)
    assert result["noise_risk"] == 0.0


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_scalar_iid_noise_matches_explicit_impulse_coefficients(delay: int) -> None:
    updates = 9
    variance = 1.4
    lags = np.zeros((2 * updates - 1, 1, 1))
    lags[updates - 1, 0, 0] = variance
    result = exact_pr_averaged_vector_risk(
        initial_history=np.full((delay + 1, 1), 0.0),
        drift=np.array([[0.9]]),
        step_size=0.08,
        delay=delay,
        updates=updates,
        burn_in=1,
        lag_covariances=lags,
    )
    coefficients = []
    for noise_time in range(updates):
        history = [0.0] * (delay + 1)
        values = []
        for time in range(updates):
            next_value = history[-1] - 0.08 * 0.9 * history[0]
            if time == noise_time:
                next_value += 0.08
            history = history[1:] + [next_value]
            values.append(next_value)
        coefficients.append(float(np.mean(values[1:])))
    expected = variance * float(np.sum(np.square(coefficients)))
    assert result["noise_risk"] == pytest.approx(expected, abs=1e-12)


def test_equal_horizon_rho_one_is_independent_of_q() -> None:
    risks = []
    for q in (1, 4, 16):
        lags = equicorrelated_ar_lag_covariances(
            horizon=12,
            single_agent_covariance=np.array([[2.0]]),
            q=q,
            rho=1.0,
            markov_lambda=0.7,
        )
        risks.append(
            exact_pr_averaged_vector_risk(
                initial_history=np.array([[0.0]]),
                drift=np.array([[1.0]]),
                step_size=0.05,
                delay=0,
                updates=12,
                burn_in=3,
                lag_covariances=lags,
            )["risk"]
        )
    np.testing.assert_allclose(risks, risks[0], atol=1e-12)


def test_asymptotic_proxy_has_participation_transition() -> None:
    catalogue = [1, 4, 16]
    assert discrete_pr_proxy_optimum(catalogue=catalogue, rho=0.0, overhead=2.0) == 16
    assert discrete_pr_proxy_optimum(catalogue=catalogue, rho=1.0, overhead=2.0) == 1
    assert pr_message_proxy(q=16, rho=0.0, overhead=2.0) < pr_message_proxy(
        q=1, rho=0.0, overhead=2.0
    )
    assert pr_message_proxy(q=16, rho=1.0, overhead=2.0) > pr_message_proxy(
        q=1, rho=1.0, overhead=2.0
    )


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_fast_scalar_pr_risk_matches_vector_identity(delay: int) -> None:
    updates = 11
    q = 4
    rho = 0.35
    markov_lambda = 0.72
    single_variance = 1.8
    lags = equicorrelated_ar_lag_covariances(
        horizon=updates,
        single_agent_covariance=np.array([[single_variance]]),
        q=q,
        rho=rho,
        markov_lambda=markov_lambda,
    )
    vector = exact_pr_averaged_vector_risk(
        initial_history=np.full((delay + 1, 1), 0.7),
        drift=np.array([[0.9]]),
        step_size=0.06,
        delay=delay,
        updates=updates,
        burn_in=4,
        lag_covariances=lags,
    )
    scalar = exact_pr_averaged_scalar_risk(
        initial_error=0.7,
        mu=0.9,
        step_size=0.06,
        delay=delay,
        updates=updates,
        burn_in=4,
        single_variance=single_variance,
        q=q,
        rho=rho,
        markov_lambda=markov_lambda,
    )
    assert scalar["risk"] == pytest.approx(vector["risk"], abs=1e-12)
    assert scalar["noise_variance"] == pytest.approx(vector["noise_risk"], abs=1e-12)
