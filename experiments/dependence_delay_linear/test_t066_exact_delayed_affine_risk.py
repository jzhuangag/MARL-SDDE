import numpy as np
import pytest

from experiments.dependence_delay_linear.t066_exact_delayed_affine_risk import (
    ScheduledAction,
    correlation_factor,
    exact_learning_horizon,
    propagate_schedule,
    terminal_mean_square_risk,
)


def test_environment_horizon_divides_by_participation():
    assert exact_learning_horizon(
        participation=4,
        message_overhead=8,
        message_budget=10_000,
        environment_budget=100,
    ) == 25
    assert exact_learning_horizon(
        participation=4,
        message_overhead=8,
        message_budget=10_000,
        environment_budget=100,
        sensor_environment_cost=20,
        reserved_delay_updates=3,
    ) == 17


def test_common_private_correlation_factor_endpoints():
    assert correlation_factor(8, 0.0) == pytest.approx(1 / 8)
    assert correlation_factor(8, 1.0) == 1.0


def test_zero_noise_matches_deterministic_scalar_recursion():
    state = propagate_schedule(
        drift=np.asarray([[0.5]]),
        base_noise_covariance=np.zeros((1, 1)),
        initial_error=np.asarray([2.0]),
        delay=0,
        rho=0.3,
        schedule=[ScheduledAction(participation=4, gain=0.1, updates=10)],
    )
    expected = 2.0 * (1.0 - 0.05) ** 10
    assert state.mean[0] == pytest.approx(expected)
    assert terminal_mean_square_risk(state, 1) == pytest.approx(expected**2)


@pytest.mark.parametrize("delay", [0, 2])
def test_exact_moments_match_monte_carlo(delay):
    drift = np.asarray([[0.4]])
    noise_variance = np.asarray([[0.7]])
    initial = np.asarray([1.2])
    rho = 0.3
    action = ScheduledAction(participation=4, gain=0.05, updates=30)
    exact = propagate_schedule(
        drift=drift,
        base_noise_covariance=noise_variance,
        initial_error=initial,
        delay=delay,
        rho=rho,
        schedule=[action],
    )
    rng = np.random.default_rng(66001 + delay)
    samples = []
    scale = np.sqrt(correlation_factor(action.participation, rho) * noise_variance[0, 0])
    for _ in range(60_000):
        history = [float(initial[0])] * (delay + 1)
        for _ in range(action.updates):
            next_error = history[0] - action.gain * drift[0, 0] * history[delay]
            next_error += action.gain * scale * rng.normal()
            history = [next_error, *history[:-1]]
        samples.append(history[0])
    samples = np.asarray(samples)
    assert np.mean(samples) == pytest.approx(exact.mean[0], abs=0.002)
    assert np.mean(samples**2) == pytest.approx(
        terminal_mean_square_risk(exact, 1), rel=0.012
    )


def test_schedule_supports_online_action_changes():
    state = propagate_schedule(
        drift=np.diag([0.5, 1.0]),
        base_noise_covariance=np.eye(2),
        initial_error=np.ones(2),
        delay=1,
        rho=0.5,
        schedule=[
            ScheduledAction(participation=2, gain=0.02, updates=10),
            ScheduledAction(participation=8, gain=0.01, updates=20),
        ],
    )
    assert state.updates == 30
    assert terminal_mean_square_risk(state, 2) > 0.0
