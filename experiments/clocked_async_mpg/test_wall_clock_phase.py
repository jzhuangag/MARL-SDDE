from __future__ import annotations

import numpy as np
import pytest

from .wall_clock_phase import (
    certified_wall_clock_coefficients,
    expected_maximum_exponential,
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
