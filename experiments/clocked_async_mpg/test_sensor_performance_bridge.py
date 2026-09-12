from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from .sensor_performance_bridge import (
    anytime_energy_log_envelope,
    certified_contraction_after_mismatches,
    certified_schedule_log_cost,
    schedule_excess_and_mismatch_bound,
)


def test_schedule_cost_selects_the_requested_multipliers() -> None:
    table = np.asarray([[-0.1, -0.3], [0.2, -0.4], [-0.2, -0.1]])
    assert certified_schedule_log_cost(table, [0, 1, 0]) == pytest.approx(-0.7)


def test_mismatch_bound_is_exhaustive_for_binary_schedules() -> None:
    table = np.asarray([[-0.2, -0.5], [0.3, -0.1], [-0.4, -0.35], [0.1, -0.2]])
    schedules = tuple(itertools.product((0, 1), repeat=table.shape[0]))
    for actions in schedules:
        for comparator in schedules:
            result = schedule_excess_and_mismatch_bound(table, actions, comparator)
            exact = certified_schedule_log_cost(table, actions) - certified_schedule_log_cost(
                table, comparator
            )
            assert result["exact_log_cost_excess"] == pytest.approx(exact)
            assert exact <= result["mismatch_penalty_bound"] + 1e-12
            assert result["action_mismatches"] == sum(
                left != right for left, right in zip(actions, comparator)
            )


def test_identical_schedules_have_zero_excess_and_penalty() -> None:
    table = np.asarray([[0.1, -0.2], [-0.3, -0.1]])
    result = schedule_excess_and_mismatch_bound(table, [1, 0], [1, 0])
    assert result == {
        "action_mismatches": 0,
        "exact_log_cost_excess": 0.0,
        "mismatch_penalty_bound": 0.0,
    }


def test_ville_log_envelope_has_the_exact_confidence_penalty() -> None:
    cumulative = np.asarray([0.0, -0.1, -0.25, -0.4])
    envelope = anytime_energy_log_envelope(
        initial_energy=2.0,
        cumulative_log_multipliers=cumulative,
        failure_probability=0.05,
    )
    assert envelope == pytest.approx(math.log(2.0) + cumulative + math.log(20.0))


def test_mismatch_penalty_reduces_the_comparator_contraction_margin() -> None:
    assert certified_contraction_after_mismatches(
        comparator_log_cost=-12.0,
        mismatch_penalty_bound=2.0,
        horizon=100,
    ) == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("table", "actions"),
    [([[0.1]], [0]), ([[0.1, math.inf]], [0]), ([[0.1, 0.2]], [2])],
)
def test_invalid_schedule_inputs_are_rejected(table: list[list[float]], actions: list[int]) -> None:
    with pytest.raises(ValueError):
        certified_schedule_log_cost(table, actions)
