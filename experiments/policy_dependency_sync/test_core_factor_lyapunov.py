from __future__ import annotations

import numpy as np
import pytest

from .core_factor_lyapunov import (
    choose_core_factor_action,
    choose_core_factor_action_variable_bounds,
    core_queue_cap,
)


def test_core_choice_matches_topology_free_dense_grid() -> None:
    choice = choose_core_factor_action_variable_bounds(
        alignment_lower_by_action={"null": 0.08, "edge": 0.22},
        communication_cost_by_action={"null": 0.0, "edge": 1.0},
        gradient_norm_upper_by_action={"null": 0.2, "edge": 0.7},
        communication_queue=0.03,
        learning_weight=2.0,
        learning_smoothness=1.3,
        receipt_motion_upper=0.01,
        maximum_packet_weight=0.5,
    )
    dense = []
    for action, alignment, cost, gradient_bound in (
        ("null", 0.08, 0.0, 0.2),
        ("edge", 0.22, 1.0, 0.7),
    ):
        gain = 2.0 * (alignment - 1.3 * gradient_bound * 0.01)
        curvature = 2.0 * 1.3 * gradient_bound**2
        for alpha in np.linspace(0.0, 0.5, 100001):
            dense.append(
                (-gain * alpha + 0.5 * curvature * alpha**2 + 0.03 * cost, action, alpha)
            )
    expected = min(dense)
    assert choice.action == expected[1]
    assert choice.packet_weight == pytest.approx(expected[2], abs=5.1e-6)
    assert choice.index == pytest.approx(expected[0], abs=1e-10)
    assert choice.reset_benefit == 0.0


def test_core_choice_depends_on_alignment_not_cache_topology() -> None:
    choice = choose_core_factor_action(
        alignment_lower_by_action={"null": 0.05, "edge": 0.8},
        communication_cost_by_action={"null": 0.0, "edge": 0.1},
        communication_queue=0.01,
        learning_weight=1.0,
        gradient_norm_upper=1.0,
        learning_smoothness=1.0,
        receipt_motion_upper=0.0,
        maximum_packet_weight=0.5,
    )
    assert choice.action == "edge"
    assert choice.packet_weight == pytest.approx(0.5)


def test_core_queue_cap_excludes_cache_reset_advantage() -> None:
    cap = core_queue_cap(
        maximum_packet_gain=2.0,
        minimum_quadratic_curvature=1.0,
        minimum_positive_cost=1.0,
        queue_step=0.5,
        maximum_cost=2.0,
        average_budget=0.5,
    )
    assert cap == pytest.approx(2.75)
