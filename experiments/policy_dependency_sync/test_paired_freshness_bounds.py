from __future__ import annotations

import pytest

from .paired_freshness_bounds import (
    paired_finite_time_rhs,
    pathwise_average_cost_upper,
    selected_utility_lagrangian_regret_upper,
    utility_queue_cap,
)


def test_optimistic_action_regret_pays_only_selected_radius() -> None:
    assert selected_utility_lagrangian_regret_upper(
        utility_weight=3.0, selected_confidence_radius=0.2
    ) == pytest.approx(1.2)


def test_queue_cap_is_rejection_threshold_plus_one_increment() -> None:
    assert utility_queue_cap(
        utility_weight=2.0,
        utility_range_upper=3.0,
        minimum_positive_cost=0.5,
        queue_step=0.2,
        maximum_cost=1.0,
        average_budget=0.25,
        maximum_reset_benefit=0.5,
    ) == pytest.approx(13.15)


def test_pathwise_average_cost_uses_the_unscaled_queue() -> None:
    assert pathwise_average_cost_upper(
        average_budget=0.25,
        terminal_queue=2.0,
        queue_step=0.5,
        launches=100,
    ) == pytest.approx(0.29)


def test_paired_rhs_keeps_every_nonnegative_remainder() -> None:
    assert paired_finite_time_rhs(
        initial_objective_gap=2.0,
        initial_queue=1.0,
        queue_step=0.5,
        utility_weight=3.0,
        selected_radius_sum=0.2,
        queue_remainder_sum=0.4,
        receipt_remainder_sum=0.3,
        topology_motion_positive_sum=0.1,
        initial_cache_energy=0.2,
    ) == pytest.approx(5.2)
