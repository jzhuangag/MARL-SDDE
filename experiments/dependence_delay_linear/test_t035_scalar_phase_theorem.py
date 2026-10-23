import math

import numpy as np
import pytest

from experiments.dependence_delay_linear.t035_scalar_phase_theorem import (
    aggregate_variance,
    continuous_proxy_optimum,
    effective_agents,
    exact_scalar_risk,
    noise_dominated_proxy,
    update_count,
)


def test_effective_agents_recovers_independence_and_saturation() -> None:
    assert effective_agents(8, 0.0) == pytest.approx(8.0)
    assert effective_agents(8, 1.0) == pytest.approx(1.0)
    assert effective_agents(8, 0.5) == pytest.approx(16.0 / 9.0)


def test_aggregate_variance_identity() -> None:
    assert aggregate_variance(2.0, 4, 0.25) == pytest.approx(
        2.0 * (0.25 + 0.75 / 4.0)
    )


def test_exact_iid_no_delay_closed_form() -> None:
    initial = 2.0
    mu = 0.7
    step = 0.2
    updates = 13
    variance = aggregate_variance(1.3, 4, 0.2)
    result = exact_scalar_risk(
        initial_error=initial,
        mu=mu,
        step_size=step,
        delay=0,
        updates=updates,
        single_variance=1.3,
        q=4,
        rho=0.2,
        markov_lambda=0.0,
    )
    contraction = 1.0 - step * mu
    expected_mean_squared = contraction ** (2 * updates) * initial**2
    expected_variance = (
        step**2
        * variance
        * (1.0 - contraction ** (2 * updates))
        / (1.0 - contraction**2)
    )
    assert result["mean_squared"] == pytest.approx(expected_mean_squared)
    assert result["noise_variance"] == pytest.approx(expected_variance)


def test_independent_equal_horizon_is_speedup() -> None:
    common = dict(
        initial_error=1.0,
        mu=1.0,
        step_size=0.1,
        delay=0,
        updates=100,
        single_variance=2.0,
        rho=0.0,
        markov_lambda=0.5,
    )
    q1 = exact_scalar_risk(q=1, **common)["risk"]
    q8 = exact_scalar_risk(q=8, **common)["risk"]
    assert q8 < q1


def test_perfect_correlation_equal_horizon_saturates() -> None:
    common = dict(
        initial_error=1.0,
        mu=1.0,
        step_size=0.1,
        delay=1,
        updates=50,
        single_variance=2.0,
        rho=1.0,
        markov_lambda=0.5,
    )
    assert exact_scalar_risk(q=1, **common)["risk"] == pytest.approx(
        exact_scalar_risk(q=16, **common)["risk"]
    )


def test_finite_budget_can_reverse_independent_variance_gain() -> None:
    budget = 20.0
    common = dict(
        initial_error=10.0,
        mu=1.0,
        step_size=0.1,
        delay=0,
        single_variance=1.0,
        rho=0.0,
        markov_lambda=0.0,
    )
    q1 = exact_scalar_risk(q=1, updates=update_count(budget, 1.0), **common)
    q8 = exact_scalar_risk(q=8, updates=update_count(budget, 8.0), **common)
    assert q8["noise_variance"] < q1["noise_variance"]
    assert q8["risk"] > q1["risk"]


def test_continuous_proxy_optimum() -> None:
    rho = 0.2
    overhead = 8.0
    per_agent = 1.0
    optimum = continuous_proxy_optimum(rho, overhead, per_agent)
    assert optimum == pytest.approx(math.sqrt(32.0))
    epsilon = 1e-4
    assert noise_dominated_proxy(optimum, rho, overhead, per_agent) <= min(
        noise_dominated_proxy(optimum - epsilon, rho, overhead, per_agent),
        noise_dominated_proxy(optimum + epsilon, rho, overhead, per_agent),
    ) + 1e-10
    assert np.isinf(continuous_proxy_optimum(0.0, overhead, per_agent))


def test_zero_updates_preserves_initial_error() -> None:
    result = exact_scalar_risk(
        initial_error=3.0,
        mu=1.0,
        step_size=0.1,
        delay=2,
        updates=0,
        single_variance=1.0,
        q=4,
        rho=0.5,
        markov_lambda=0.8,
    )
    assert result["risk"] == pytest.approx(9.0)
    assert result["noise_variance"] == 0.0
