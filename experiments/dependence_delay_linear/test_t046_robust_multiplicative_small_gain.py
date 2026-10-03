import numpy as np
import pytest

from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    delayed_vector_companion,
)
from experiments.dependence_delay_linear.t046_robust_multiplicative_small_gain import (
    finite_horizon_impulse_gain,
    multiplicative_risk_envelope,
    pathwise_perturbation_audit,
    phase_order_certified,
    robust_small_gain,
)


def test_scalar_geometric_impulse_gain() -> None:
    companion = np.array([[0.8]])
    expected = sum(0.8**index for index in range(7))
    assert finite_horizon_impulse_gain(companion, 7) == pytest.approx(expected)


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_pathwise_small_gain_bound_holds_for_delayed_lift(delay: int) -> None:
    companion = delayed_vector_companion(
        np.array([[0.7]]), step_size=0.03, delay=delay
    )
    horizon = 12
    rng = np.random.RandomState(1240 + delay)
    inputs = rng.normal(scale=0.01, size=(horizon, delay + 1))
    perturbations = np.zeros((horizon, delay + 1, delay + 1))
    for time in range(horizon):
        perturbations[time, 0, delay] = rng.uniform(-2e-4, 2e-4)
    audit = pathwise_perturbation_audit(
        companion=companion,
        additive_inputs=inputs,
        multiplicative_updates=perturbations,
        initial_state=np.full(delay + 1, 0.2),
    )
    assert audit["certified"]
    assert audit["difference_path_sup"] <= audit["difference_bound"] + 1e-12


def test_uncertified_gain_is_not_reported_as_finite() -> None:
    result = robust_small_gain(
        step_size=0.2,
        multiplicative_deviation_bound=2.0,
        impulse_gain=3.0,
    )
    assert result["certified"] is False
    assert np.isinf(result["path_amplification"])


def test_risk_envelope_and_phase_certificate() -> None:
    envelope = multiplicative_risk_envelope(
        additive_terminal_risk=1.0,
        additive_lifted_second_moment_sum=2.0,
        relative_path_perturbation=0.05,
    )
    assert envelope["lower"] < 1.0 < envelope["upper"]
    assert phase_order_certified(preferred_upper=0.8, comparator_lower=0.9)
    assert not phase_order_certified(preferred_upper=0.95, comparator_lower=0.9)


def test_zero_multiplicative_deviation_recovers_additive_risk() -> None:
    gain = robust_small_gain(
        step_size=0.1,
        multiplicative_deviation_bound=0.0,
        impulse_gain=100.0,
    )
    envelope = multiplicative_risk_envelope(
        additive_terminal_risk=2.5,
        additive_lifted_second_moment_sum=9.0,
        relative_path_perturbation=gain["relative_path_perturbation"],
    )
    assert envelope["lower"] == pytest.approx(2.5)
    assert envelope["upper"] == pytest.approx(2.5)
