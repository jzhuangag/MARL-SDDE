from __future__ import annotations

import numpy as np
import pytest

from .signed_packet_control import (
    choose_one_edge_signed,
    favorable_phase_margin,
    launch_to_receipt_error_bound,
    paired_stationarity_upper,
    signed_selection_regret_bound,
    smooth_packet_drift_score,
)


def test_smooth_packet_score_matches_quadratic_drift_at_launch() -> None:
    gradient = np.asarray([0.7, -0.2])
    mean = np.asarray([0.4, -0.5])
    variance = 0.13
    score = smooth_packet_drift_score(
        gradient,
        mean,
        float(mean @ mean) + variance,
        smoothness=1.3,
        step=0.08,
    )
    expected = -0.08 * float(gradient @ mean)
    expected += 0.5 * 1.3 * 0.08**2 * (float(mean @ mean) + variance)
    assert score == pytest.approx(expected)


def test_launch_to_receipt_bound_dominates_gradient_motion_error() -> None:
    launch = np.asarray([0.2, -0.7, 0.4])
    receipt = np.asarray([0.3, -0.55, 0.32])
    mean = np.asarray([-0.4, 0.1, 0.25])
    step = 0.06
    actual = step * abs(float((receipt - launch) @ mean))
    bound = launch_to_receipt_error_bound(
        step,
        float(np.linalg.norm(receipt - launch)),
        float(np.linalg.norm(mean)),
    )
    assert actual <= bound + 1e-15


def test_one_edge_rule_is_exact_and_has_stable_ties() -> None:
    drift = {None: -0.1, (1, 0): -0.18, (2, 0): -0.22}
    cost = {None: 0.0, (1, 0): 1.0, (2, 0): 2.0}
    choice = choose_one_edge_signed(drift, cost, queue=0.3, learning_weight=5.0)
    indices = {action: 5.0 * drift[action] + 0.3 * cost[action] for action in drift}
    assert choice.action == min(indices, key=lambda item: (indices[item], repr(item)))
    assert choice.index == pytest.approx(indices[choice.action])


def test_noisy_selection_regret_is_bounded_by_twice_uniform_error() -> None:
    true = {None: -0.1, "a": -0.3, "b": -0.2}
    estimated = {None: -0.08, "a": -0.24, "b": -0.27}
    cost = {None: 0.0, "a": 1.0, "b": 1.0}
    regret, bound = signed_selection_regret_bound(
        true, estimated, cost, queue=0.1, learning_weight=4.0
    )
    assert 0.0 <= regret <= bound + 1e-15


def test_favorable_margin_is_equivalent_to_declared_sufficient_bound() -> None:
    margin = favorable_phase_margin(
        gradient_norm=2.0,
        packet_bias_bound=0.1,
        packet_variance=0.05,
        receipt_gradient_motion_bound=0.05,
        smoothness=1.0,
        step=0.05,
        descent_fraction=0.25,
    )
    assert margin > 0.0
    assert favorable_phase_margin(0.1, 0.5, 1.0, 0.5, 1.0, 0.1, 0.25) < 0.0


def test_paired_stationarity_bound_decreases_at_root_rate() -> None:
    values = []
    for launches in (100, 400, 1600):
        root = launches**0.5
        values.append(
            paired_stationarity_upper(
                initial_suboptimality=1.0,
                initial_queue=0.0,
                launches=launches,
                learning_weight=root,
                descent_mass=1.0 / root,
                queue_drift_constant=0.0,
                comparator_remainder_sum=0.0,
                score_error_sum=0.0,
            )
        )
    assert values == pytest.approx([0.1, 0.05, 0.025])


def test_stochastic_joint_budget_scaling_is_cube_root() -> None:
    values = []
    budget_remainders = []
    for launches in (125, 1000, 8000):
        step = launches ** (-1.0 / 3.0)
        weight = launches ** (2.0 / 3.0)
        values.append(
            paired_stationarity_upper(
                initial_suboptimality=1.0,
                initial_queue=0.0,
                launches=launches,
                learning_weight=weight,
                descent_mass=step,
                queue_drift_constant=1.0,
                comparator_remainder_sum=launches * step * step,
                score_error_sum=0.0,
            )
        )
        budget_remainders.append(weight / launches)
    assert np.all(np.asarray(values[1:]) < np.asarray(values[:-1]))
    assert budget_remainders == pytest.approx([0.2, 0.1, 0.05])
