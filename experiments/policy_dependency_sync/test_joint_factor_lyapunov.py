from __future__ import annotations

import numpy as np
import pytest

from .joint_factor_lyapunov import (
    choose_joint_factor_action,
    choose_joint_factor_action_variable_bounds,
    joint_queue_cap,
    projected_quadratic_minimizer,
)


def test_action_specific_gradient_bounds_match_dense_grid() -> None:
    choice = choose_joint_factor_action_variable_bounds(
        alignment_lower_by_action={"null": 0.08, "edge": 0.14},
        reset_benefit_by_action={"null": 0.0, "edge": 0.02},
        communication_cost_by_action={"null": 0.0, "edge": 1.0},
        gradient_norm_upper_by_action={"null": 0.2, "edge": 0.7},
        communication_queue=0.03,
        learning_weight=2.0,
        learning_smoothness=1.3,
        receipt_motion_upper=0.01,
        receipt_cache_linear_upper=0.02,
        outgoing_cache_weight=0.4,
        maximum_packet_weight=0.5,
    )
    dense = []
    for action, alignment, reset, cost, gradient_bound in (
        ("null", 0.08, 0.0, 0.0, 0.2),
        ("edge", 0.14, 0.02, 1.0, 0.7),
    ):
        gain = 2.0 * (alignment - 1.3 * gradient_bound * 0.01) - gradient_bound * 0.02
        curvature = gradient_bound**2 * (2.0 * 1.3 + 0.4)
        for alpha in np.linspace(0.0, 0.5, 100001):
            dense.append(
                (
                    -gain * alpha
                    + 0.5 * curvature * alpha**2
                    - reset
                    + 0.03 * cost,
                    action,
                    alpha,
                )
            )
    expected = min(dense)
    assert choice.action == expected[1]
    assert choice.packet_weight == pytest.approx(expected[2], abs=5.1e-6)
    assert choice.index == pytest.approx(expected[0], abs=1e-10)


def test_exact_index_tie_prefers_zero_cost_null_action() -> None:
    choice = choose_joint_factor_action_variable_bounds(
        alignment_lower_by_action={"edge": float("-inf"), "null": 0.0},
        reset_benefit_by_action={"edge": 0.0, "null": 0.0},
        communication_cost_by_action={"edge": 1.0, "null": 0.0},
        gradient_norm_upper_by_action={"edge": 1.0, "null": 0.1},
        communication_queue=0.0,
        learning_weight=1.0,
        learning_smoothness=1.0,
        receipt_motion_upper=0.0,
        receipt_cache_linear_upper=0.0,
        outgoing_cache_weight=0.0,
        maximum_packet_weight=0.1,
    )
    assert choice.action == "null"
    assert choice.packet_weight == 0.0


def test_plugin_dynamic_oracle_gap_is_at_most_twice_uniform_error() -> None:
    rng = np.random.default_rng(20491)
    actions = ("a", "b", "null")
    learning_weight = 3.0
    smoothness = 1.2
    motion = 0.03
    cache_linear = 0.04
    outgoing_weight = 0.3
    alpha_max = 0.2
    queue = 0.07
    gradient_bounds = {"a": 0.8, "b": 0.5, "null": 0.1}
    resets = {"a": 0.02, "b": 0.01, "null": 0.0}
    costs = {"a": 1.0, "b": 1.0, "null": 0.0}
    epsilon = 0.025

    def true_index(action: str, alpha: float, alignment: dict[str, float]) -> float:
        gradient_bound = gradient_bounds[action]
        gain = (
            learning_weight
            * (alignment[action] - smoothness * gradient_bound * motion)
            - gradient_bound * cache_linear
        )
        curvature = gradient_bound**2 * (
            learning_weight * smoothness + outgoing_weight
        )
        return (
            -gain * alpha
            + 0.5 * curvature * alpha**2
            - resets[action]
            + queue * costs[action]
        )

    for _ in range(200):
        truth = {action: float(rng.uniform(-0.1, 0.25)) for action in actions}
        estimate = {
            action: truth[action] + float(rng.uniform(-epsilon, epsilon))
            for action in actions
        }
        keyword = dict(
            reset_benefit_by_action=resets,
            communication_cost_by_action=costs,
            gradient_norm_upper_by_action=gradient_bounds,
            communication_queue=queue,
            learning_weight=learning_weight,
            learning_smoothness=smoothness,
            receipt_motion_upper=motion,
            receipt_cache_linear_upper=cache_linear,
            outgoing_cache_weight=outgoing_weight,
            maximum_packet_weight=alpha_max,
        )
        selected = choose_joint_factor_action_variable_bounds(
            alignment_lower_by_action=estimate, **keyword
        )
        oracle = choose_joint_factor_action_variable_bounds(
            alignment_lower_by_action=truth, **keyword
        )
        gap = true_index(selected.action, selected.packet_weight, truth) - oracle.index
        assert gap <= 2.0 * learning_weight * alpha_max * epsilon + 1e-12


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
