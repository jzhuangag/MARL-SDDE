from __future__ import annotations

import numpy as np
import pytest

from .paired_freshness_control import (
    choose_launch_cache_action,
    choose_receipt_packet_weight,
    communication_queue_update,
)


def test_launch_rule_matches_direct_index_and_respects_prefix_feasibility() -> None:
    choice = choose_launch_cache_action(
        utility_upper_by_action={"null": 1.0, "left": 1.4, "right": 1.6},
        communication_cost_by_action={"null": 0.0, "left": 1.0, "right": 1.0},
        communication_queue=0.2,
        utility_weight=2.0,
        feasible_by_action={"null": True, "left": True, "right": False},
    )
    assert choice.action == "left"
    assert choice.index == pytest.approx(-2.6)


def test_large_queue_prefers_zero_cost_launch() -> None:
    choice = choose_launch_cache_action(
        utility_upper_by_action={"null": 1.0, "edge": 1.2},
        communication_cost_by_action={"null": 0.0, "edge": 1.0},
        communication_queue=1.0,
        utility_weight=1.0,
    )
    assert choice.action == "null"


def test_receipt_root_matches_dense_smoothness_minimizer() -> None:
    choice = choose_receipt_packet_weight(
        observed_launch_alignment=1.4,
        packet_norm_upper=2.0,
        reference_error_upper=0.1,
        learning_smoothness=1.5,
        launch_to_receipt_motion_upper=0.2,
        maximum_packet_weight=0.5,
    )
    grid = np.linspace(0.0, 0.5, 100001)
    values = (
        -choice.certified_alignment * grid
        + 0.5 * choice.quadratic_curvature * grid**2
    )
    assert choice.packet_weight == pytest.approx(
        float(grid[int(np.argmin(values))]), abs=5.1e-6
    )
    assert choice.drift_upper == pytest.approx(float(np.min(values)), abs=1e-10)


def test_receipt_motion_can_certifiably_reject_packet() -> None:
    choice = choose_receipt_packet_weight(
        observed_launch_alignment=0.2,
        packet_norm_upper=1.0,
        reference_error_upper=0.05,
        learning_smoothness=2.0,
        launch_to_receipt_motion_upper=0.2,
        maximum_packet_weight=1.0,
    )
    assert choice.certified_alignment < 0.0
    assert choice.packet_weight == 0.0
    assert choice.drift_upper == 0.0


def test_queue_update_has_exact_reflection_and_scaling() -> None:
    assert communication_queue_update(
        queue=0.1, queue_step=0.5, realized_cost=0.0, average_budget=0.4
    ) == 0.0
    assert communication_queue_update(
        queue=0.1, queue_step=0.5, realized_cost=1.0, average_budget=0.4
    ) == pytest.approx(0.4)
