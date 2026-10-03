from __future__ import annotations

import itertools

import numpy as np
import pytest

from .controlled_kernel_bridge import (
    categorical_total_variation,
    finite_horizon_oscillation_charge,
    induced_state_kernel,
    maximum_row_total_variation,
    trajectory_marginal_tv_bound,
)


def _enumerated_state_law(
    initial: np.ndarray, kernel: np.ndarray, horizon: int
) -> np.ndarray:
    law = np.asarray(initial, dtype=float)
    for _ in range(horizon):
        law = law @ kernel
    return law


def test_data_processing_from_policy_to_state_kernel() -> None:
    policy_zero = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    policy_one = np.asarray([[0.4, 0.6], [0.6, 0.4]])
    controlled = np.asarray(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.6, 0.4], [0.1, 0.9]],
        ]
    )
    kernel_zero = induced_state_kernel(policy_zero, controlled)
    kernel_one = induced_state_kernel(policy_one, controlled)
    policy_tv = max(
        categorical_total_variation(left, right)
        for left, right in zip(policy_zero, policy_one, strict=True)
    )
    assert maximum_row_total_variation(kernel_zero, kernel_one) <= policy_tv + 1e-12


@pytest.mark.parametrize("horizon", range(7))
def test_coupling_bound_dominates_exact_two_state_marginal_tv(horizon: int) -> None:
    left = np.asarray([[0.92, 0.08], [0.25, 0.75]])
    right = np.asarray([[0.75, 0.25], [0.35, 0.65]])
    initial = np.asarray([1.0, 0.0])
    one_step = maximum_row_total_variation(left, right)
    exact = categorical_total_variation(
        _enumerated_state_law(initial, left, horizon),
        _enumerated_state_law(initial, right, horizon),
    )
    assert exact <= trajectory_marginal_tv_bound(one_step, horizon) + 1e-12


def test_score_expectation_shift_is_bounded_by_oscillation_charge() -> None:
    left = np.asarray([[0.85, 0.15], [0.20, 0.80]])
    right = np.asarray([[0.70, 0.30], [0.35, 0.65]])
    initial = np.asarray([0.4, 0.6])
    score = np.asarray([-0.4, 0.9])
    horizon = 6
    left_expectation = 0.0
    right_expectation = 0.0
    for step in range(horizon):
        left_expectation += float(_enumerated_state_law(initial, left, step) @ score)
        right_expectation += float(_enumerated_state_law(initial, right, step) @ score)
    charge = finite_horizon_oscillation_charge(
        [float(np.ptp(score))] * horizon,
        maximum_row_total_variation(left, right),
    )
    assert abs(left_expectation - right_expectation) <= charge + 1e-12


def test_action_score_charges_policy_shift_at_launch_step() -> None:
    charge = finite_horizon_oscillation_charge(
        [2.0, 2.0], 0.25, score_uses_action=True
    )
    assert charge == pytest.approx(2.0 * 0.25 + 2.0 * (1.0 - 0.75**2))


@pytest.mark.parametrize(
    "one_step,horizon,expected",
    [(0.0, 9, 0.0), (1.0, 1, 1.0), (0.2, 0, 0.0), (0.2, 2, 0.36)],
)
def test_closed_form_marginal_bound(
    one_step: float, horizon: int, expected: float
) -> None:
    assert trajectory_marginal_tv_bound(one_step, horizon) == pytest.approx(expected)


@pytest.mark.parametrize(
    "policy,transition",
    [
        (np.asarray([[1.0, 0.0]]), np.ones((1, 1, 1))),
        (np.asarray([[0.7, 0.7]]), np.ones((1, 2, 1))),
    ],
)
def test_invalid_induced_kernel_inputs_fail_closed(
    policy: np.ndarray, transition: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        induced_state_kernel(policy, transition)


def test_all_binary_two_step_kernels_obey_the_bound() -> None:
    probabilities = (0.0, 0.25, 0.5, 0.75, 1.0)
    initial = np.asarray([0.35, 0.65])
    for entries in itertools.product(probabilities, repeat=4):
        left = np.asarray(
            [[entries[0], 1.0 - entries[0]], [entries[1], 1.0 - entries[1]]]
        )
        right = np.asarray(
            [[entries[2], 1.0 - entries[2]], [entries[3], 1.0 - entries[3]]]
        )
        step_tv = maximum_row_total_variation(left, right)
        exact = categorical_total_variation(initial @ left @ left, initial @ right @ right)
        assert exact <= trajectory_marginal_tv_bound(step_tv, 2) + 1e-12
