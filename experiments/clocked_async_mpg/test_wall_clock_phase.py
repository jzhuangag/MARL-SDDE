from __future__ import annotations

import numpy as np
import pytest

from .wall_clock_phase import (
    certified_wall_clock_coefficients,
    essential_agent_clock_lower_bound,
    essential_agent_periodic_service_clock_lower_bound,
    expected_maximum_exponential,
    gaussian_sign_packet_lower_bound,
    robust_stale_direction_progress,
    stochastic_essential_agent_clock_lower_bound,
    stochastic_essential_agent_periodic_service_lower_bound,
    symmetric_interaction_phase,
)


def test_equal_rate_barrier_time_is_harmonic_number() -> None:
    for agents in (1, 2, 5, 9):
        rate = 1.7
        expected = sum(1.0/index for index in range(1, agents+1))/rate
        assert expected_maximum_exponential(
            np.full(agents, rate)
        ) == pytest.approx(expected, abs=1e-12)


def test_zero_delay_symmetric_ratio_matches_closed_form() -> None:
    agents, diagonal, cross = 6, 2.0, 0.3
    result = symmetric_interaction_phase(
        agents, diagonal, cross, completion_rate=1.4, maximum_event_delay=0
    )
    harmonic = sum(1.0/index for index in range(1, agents+1))
    global_smoothness = diagonal+(agents-1)*cross
    expected_ratio = harmonic*global_smoothness/diagonal
    assert result["coefficient_ratio"] == pytest.approx(expected_ratio)


def test_certified_async_advantage_decreases_with_event_delay() -> None:
    ratios = [
        float(
            symmetric_interaction_phase(
                agents=8,
                diagonal_smoothness=1.5,
                cross_smoothness=0.2,
                completion_rate=1.0,
                maximum_event_delay=delay,
                history_inflation=1.4,
            )["coefficient_ratio"]
        )
        for delay in (0, 2, 5, 10, 20)
    ]
    assert all(left > right for left, right in zip(ratios, ratios[1:]))
    assert ratios[0] > 1.0
    assert ratios[-1] < 1.0


def test_heterogeneous_rate_coefficients_are_finite() -> None:
    matrix = np.asarray([[2.0, 0.3, 0.1], [0.3, 1.4, 0.2], [0.1, 0.2, 1.8]])
    result = certified_wall_clock_coefficients(
        matrix,
        np.asarray([0.4, 1.0, 2.5]),
        maximum_event_delay=3,
        synchronous_smoothness=float(np.max(np.linalg.eigvalsh(matrix))),
        history_inflation=1.2,
    )
    assert float(result["asynchronous_coefficient"]) > 0.0
    assert float(result["synchronous_coefficient"]) > 0.0
    assert np.sum(np.asarray(result["mark_probabilities"])) == pytest.approx(1.0)


def test_wall_clock_helpers_reject_invalid_rates() -> None:
    with pytest.raises(ValueError):
        expected_maximum_exponential(np.asarray([1.0, 0.0]))


def test_robust_stale_progress_matches_dense_step_search() -> None:
    signal, uncertainty, smoothness, cap = 2.4, 0.7, 1.8, 0.9
    exact = robust_stale_direction_progress(
        signal, uncertainty, smoothness, cap
    )
    steps = np.linspace(0.0, cap, 200_001)
    values = (
        steps*signal*(signal-uncertainty)
        -0.5*smoothness*steps**2*signal**2
    )
    assert exact["certified_progress"] == pytest.approx(
        float(np.max(values)), abs=1e-10
    )
    assert exact["step"] == pytest.approx(float(steps[np.argmax(values)]), abs=1e-5)


def test_robust_stale_progress_has_exact_identifiability_boundary() -> None:
    for uncertainty in (1.0, 1.2, 10.0):
        result = robust_stale_direction_progress(1.0, uncertainty, 2.0)
        assert result == {
            "certified_progress": 0.0,
            "step": 0.0,
            "unconstrained_step": 0.0,
        }
    below = robust_stale_direction_progress(1.0, 0.2, 2.0)
    assert below["certified_progress"] == pytest.approx((1.0-0.2)**2/4.0)


def test_essential_agent_clock_lower_bound_tracks_slow_required_block() -> None:
    lower = essential_agent_clock_lower_bound(
        np.asarray([3, 8, 2]), np.asarray([2.0, 0.5, 5.0])
    )
    assert lower == pytest.approx(16.0)


def test_periodic_service_lower_bound_matches_poisson_rate_reparameterization() -> None:
    packets = np.asarray([3, 8, 2])
    periods = np.asarray([0.5, 2.0, 0.2])
    assert essential_agent_periodic_service_clock_lower_bound(
        packets, periods
    ) == pytest.approx(
        essential_agent_clock_lower_bound(packets, 1.0/periods)
    )


def test_periodic_service_lower_bound_tracks_slow_essential_block() -> None:
    lower = essential_agent_periodic_service_clock_lower_bound(
        np.asarray([5, 3, 8]), np.asarray([0.2, 5.0, 0.3])
    )
    assert lower == pytest.approx(15.0)


def test_phase_lower_bound_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        robust_stale_direction_progress(-1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        robust_stale_direction_progress(1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        essential_agent_clock_lower_bound(
            np.asarray([1.5, 2.0]), np.asarray([1.0, 1.0])
        )
    with pytest.raises(ValueError):
        essential_agent_clock_lower_bound(
            np.asarray([1.0]), np.asarray([0.0])
        )
    with pytest.raises(ValueError):
        essential_agent_periodic_service_clock_lower_bound(
            np.asarray([1.0, 2.0]), np.asarray([1.0])
        )
    with pytest.raises(ValueError):
        essential_agent_periodic_service_clock_lower_bound(
            np.asarray([1.0]), np.asarray([np.inf])
        )


def test_gaussian_sign_packet_bound_matches_binary_change_of_measure() -> None:
    signal, noise, error = 0.4, 1.7, 0.05
    expected_binary_kl = (
        (1.0-error)*np.log((1.0-error)/error)
        +error*np.log(error/(1.0-error))
    )
    expected = expected_binary_kl/(2.0*signal**2/noise**2)
    assert gaussian_sign_packet_lower_bound(signal, noise, error) == pytest.approx(
        expected
    )


def test_stochastic_clock_lower_bound_is_set_by_hardest_essential_agent() -> None:
    signals = np.asarray([0.5, 0.2, 0.8])
    noise = np.asarray([1.0, 1.4, 0.7])
    rates = np.asarray([1.0, 0.3, 2.0])
    packet_bounds = np.asarray(
        [
            gaussian_sign_packet_lower_bound(signal, scale, 0.1)
            for signal, scale in zip(signals, noise, strict=True)
        ]
    )
    lower = stochastic_essential_agent_clock_lower_bound(
        signals, noise, rates, 0.1
    )
    assert lower == pytest.approx(float(np.max(packet_bounds/rates)))
    assert int(np.argmax(packet_bounds/rates)) == 1


def test_periodic_stochastic_clock_lower_bound_matches_rate_reparameterization() -> None:
    signals = np.asarray([0.5, 0.2, 0.8])
    noise = np.asarray([1.0, 1.4, 0.7])
    periods = np.asarray([1.0, 1.0/0.3, 0.5])
    error = 0.1
    assert stochastic_essential_agent_periodic_service_lower_bound(
        signals, noise, periods, error
    ) == pytest.approx(
        stochastic_essential_agent_clock_lower_bound(
            signals, noise, 1.0/periods, error
        )
    )


def test_stochastic_clock_lower_bound_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        gaussian_sign_packet_lower_bound(1.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        stochastic_essential_agent_clock_lower_bound(
            np.asarray([1.0]), np.asarray([1.0, 2.0]), np.asarray([1.0]), 0.1
        )
    with pytest.raises(ValueError):
        stochastic_essential_agent_periodic_service_lower_bound(
            np.asarray([1.0]), np.asarray([1.0]), np.asarray([0.0]), 0.1
        )
