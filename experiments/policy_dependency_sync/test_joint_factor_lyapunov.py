from __future__ import annotations

import numpy as np
import pytest

from .joint_factor_lyapunov import (
    choose_joint_factor_action,
    joint_queue_cap,
    projected_quadratic_minimizer,
)


def test_projected_quadratic_minimizer_matches_dense_grid() -> None:
    alpha, value = projected_quadratic_minimizer(
        linear_gain=3.0, quadratic_curvature=4.0, maximum_weight=1.0
    )
    grid = np.linspace(0.0, 1.0, 10001)
    objectives = -3.0 * grid + 2.0 * grid**2
    assert alpha == pytest.approx(0.75)
    assert value == pytest.approx(float(np.min(objectives)))


def test_joint_choice_solves_each_weight_then_selects_refresh() -> None:
    choice = choose_joint_factor_action(
        alignment_lower_by_action={"null": 1.0, "edge": 2.0},
        reset_benefit_by_action={"null": 0.0, "edge": 0.3},
        communication_cost_by_action={"null": 0.0, "edge": 1.0},
        communication_queue=0.1,
        learning_weight=2.0,
        gradient_norm_upper=1.0,
        learning_smoothness=1.0,
        receipt_motion_upper=0.0,
        receipt_cache_linear_upper=0.0,
        outgoing_cache_weight=0.0,
        maximum_packet_weight=1.0,
    )
    assert choice.action == "edge"
    assert choice.packet_weight == 1.0
    assert choice.index == pytest.approx(-3.2)


def test_uncertainty_and_delay_reduce_packet_weight_without_a_heuristic_rule() -> None:
    common = dict(
        reset_benefit_by_action={"null": 0.0},
        communication_cost_by_action={"null": 0.0},
        communication_queue=0.0,
        learning_weight=1.0,
        gradient_norm_upper=2.0,
        learning_smoothness=1.0,
        receipt_cache_linear_upper=0.0,
        outgoing_cache_weight=0.0,
        maximum_packet_weight=1.0,
    )
    fresh = choose_joint_factor_action(
        alignment_lower_by_action={"null": 4.0},
        receipt_motion_upper=0.0,
        **common,
    )
    delayed = choose_joint_factor_action(
        alignment_lower_by_action={"null": 4.0},
        receipt_motion_upper=1.0,
        **common,
    )
    assert fresh.packet_weight == 1.0
    assert delayed.packet_weight == 0.5


def test_large_queue_selects_zero_cost_null_even_with_refresh_benefit() -> None:
    choice = choose_joint_factor_action(
        alignment_lower_by_action={"null": 0.0, "edge": 0.0},
        reset_benefit_by_action={"null": 0.0, "edge": 2.0},
        communication_cost_by_action={"null": 0.0, "edge": 1.0},
        communication_queue=3.0,
        learning_weight=1.0,
        gradient_norm_upper=1.0,
        learning_smoothness=1.0,
        receipt_motion_upper=0.0,
        receipt_cache_linear_upper=0.0,
        outgoing_cache_weight=0.0,
        maximum_packet_weight=1.0,
    )
    assert choice.action == "null"
    assert choice.packet_weight == 0.0
    assert choice.index == 0.0


def test_joint_queue_cap_contains_last_possible_positive_increment() -> None:
    cap = joint_queue_cap(
        maximum_packet_gain=2.0,
        minimum_quadratic_curvature=1.0,
        maximum_reset_benefit=3.0,
        minimum_positive_cost=1.0,
        queue_step=0.5,
        maximum_cost=2.0,
        average_budget=0.5,
    )
    assert cap == pytest.approx(5.75)
