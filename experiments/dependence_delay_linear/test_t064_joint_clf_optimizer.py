import numpy as np
import pytest

from experiments.dependence_delay_linear.t064_joint_clf_optimizer import (
    JointDriftParameters,
    brute_force_integer_joint_action,
    exact_integer_joint_action,
    joint_drift_score,
    optimal_gain_for_participation,
)


def parameters(**overrides) -> JointDriftParameters:
    values = {
        "contraction": 1.2,
        "state_signal": 2.0,
        "delay_curvature": 0.3,
        "noise_coefficient": 1.0,
        "noise_scale": 4.0,
        "rho_upper": 0.2,
        "message_price": 0.001,
        "environment_price": 0.002,
        "overhead": 8.0,
        "eta_min": 0.0,
        "eta_max": 0.5,
    }
    values.update(overrides)
    return JointDriftParameters(**values)


def test_fixed_q_gain_is_exact_quadratic_minimizer():
    config = parameters()
    gain = optimal_gain_for_participation(7.0, config)
    grid = np.linspace(config.eta_min, config.eta_max, 10_001)
    scores = [joint_drift_score(7.0, float(eta), config) for eta in grid]
    assert joint_drift_score(7.0, gain, config) <= min(scores) + 1e-8


def test_continuous_rounding_matches_exhaustive_integer_solution():
    rng = np.random.default_rng(64001)
    for _ in range(1000):
        q_max = int(rng.integers(2, 129))
        config = parameters(
            contraction=float(rng.uniform(0.05, 3.0)),
            state_signal=float(rng.uniform(0.0, 10.0)),
            delay_curvature=float(rng.uniform(0.0, 2.0)),
            noise_coefficient=float(rng.uniform(0.0, 3.0)),
            noise_scale=float(rng.uniform(0.0, 10.0)),
            rho_upper=float(rng.uniform(0.0, 1.0)),
            message_price=float(rng.uniform(0.0, 0.1)),
            environment_price=float(rng.uniform(0.0, 0.1)),
            overhead=float(rng.uniform(0.0, 64.0)),
            eta_max=float(rng.uniform(1e-4, 1.0)),
        )
        exact = exact_integer_joint_action(q_min=1, q_max=q_max, parameters=config)
        exhaustive = brute_force_integer_joint_action(
            q_min=1, q_max=q_max, parameters=config
        )
        assert exact.participation == exhaustive.participation
        assert exact.gain == pytest.approx(exhaustive.gain, abs=1e-12)
        assert exact.drift_score == pytest.approx(exhaustive.drift_score, abs=1e-12)
        assert len(exact.integer_candidates) <= 2


def test_high_common_correlation_removes_value_of_extra_participation():
    low = exact_integer_joint_action(
        q_min=1,
        q_max=64,
        parameters=parameters(
            rho_upper=0.0,
            message_price=1e-5,
            environment_price=1e-5,
        ),
    )
    high = exact_integer_joint_action(
        q_min=1,
        q_max=64,
        parameters=parameters(
            rho_upper=1.0,
            message_price=1e-5,
            environment_price=1e-5,
        ),
    )
    assert low.participation > 1
    assert high.participation == 1


def test_more_noise_reduces_the_safe_drift_minimizing_gain():
    quiet = optimal_gain_for_participation(4.0, parameters(noise_scale=0.1))
    noisy = optimal_gain_for_participation(4.0, parameters(noise_scale=20.0))
    assert noisy < quiet


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="rho_upper"):
        optimal_gain_for_participation(1.0, parameters(rho_upper=1.1))
    with pytest.raises(ValueError, match="positive"):
        optimal_gain_for_participation(0.0, parameters())
