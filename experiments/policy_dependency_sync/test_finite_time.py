from __future__ import annotations

import itertools

import pytest

from experiments.policy_dependency_sync.finite_time import (
    average_cost_upper,
    choose_scaled_drift_plus_penalty,
    deterministic_queue_cap,
    finite_time_stationarity_upper,
    queue_squared_drift_upper,
    selected_index_regret_bound,
)


def test_scaled_selector_and_index_regret_bound() -> None:
    true = {"null": 0.0, "edge-a": -1.0, "edge-b": -0.4}
    estimated = {"null": 0.1, "edge-a": -0.6, "edge-b": -0.7}
    costs = {"null": 0.0, "edge-a": 1.0, "edge-b": 0.3}
    choice = choose_scaled_drift_plus_penalty(estimated, costs, 0.4, 3.0)
    assert choice.action == "edge-b"
    regret, bound = selected_index_regret_bound(
        true, estimated, costs, 0.4, 3.0
    )
    assert regret >= 0.0
    assert regret <= bound + 1e-12


def test_two_v_sup_error_bound_exhaustively_on_small_grid() -> None:
    actions = ("0", "1", "2")
    costs = {"0": 0.0, "1": 0.5, "2": 1.0}
    values = (-1.0, -0.2, 0.4)
    errors = (-0.3, 0.0, 0.3)
    for true_values, perturbations in itertools.product(
        itertools.product(values, repeat=3),
        itertools.product(errors, repeat=3),
    ):
        true = dict(zip(actions, true_values, strict=True))
        estimated = {
            action: true[action] + error
            for action, error in zip(actions, perturbations, strict=True)
        }
        regret, bound = selected_index_regret_bound(
            true, estimated, costs, queue=0.7, learning_weight=2.5
        )
        assert regret <= bound + 1e-12


def test_queue_squared_drift_is_an_upper_bound() -> None:
    for queue in (0.0, 0.2, 3.0):
        for cost in (0.0, 0.5, 2.0):
            for budget in (0.0, 0.4, 1.5):
                next_queue = max(queue + cost - budget, 0.0)
                exact = 0.5 * (next_queue**2 - queue**2)
                assert exact <= queue_squared_drift_upper(
                    queue, cost, budget
                ) + 1e-12


def test_finite_time_bound_matches_displayed_formula() -> None:
    observed = finite_time_stationarity_upper(
        initial_learning_energy=4.0,
        initial_queue=2.0,
        horizon=100,
        learning_weight=10.0,
        descent_coefficient=0.25,
        queue_drift_constant=0.5,
        comparator_remainder_sum=3.0,
        score_error_sum=2.0,
    )
    expected = (4.0 + 0.5 * 4.0 / 10.0) / 25.0
    expected += 0.5 / 2.5
    expected += (3.0 + 4.0) / 25.0
    assert observed == pytest.approx(expected)


def test_queue_cap_and_average_cost_are_explicit() -> None:
    cap = deterministic_queue_cap(
        learning_weight=5.0,
        estimated_score_bound=2.0,
        minimum_positive_cost=0.5,
        maximum_cost=1.0,
    )
    assert cap == pytest.approx(41.0)
    assert average_cost_upper(0.2, 0.0, cap, 100) == pytest.approx(0.61)


@pytest.mark.parametrize(
    "call",
    [
        lambda: choose_scaled_drift_plus_penalty({}, {}, 0.0, 1.0),
        lambda: deterministic_queue_cap(1.0, 1.0, 0.0, 1.0),
        lambda: finite_time_stationarity_upper(0, 0, 0, 1, 1, 0, 0, 0),
        lambda: average_cost_upper(0.0, 0.0, 0.0, 0),
    ],
)
def test_invalid_inputs_fail_closed(call) -> None:
    with pytest.raises(ValueError):
        call()
