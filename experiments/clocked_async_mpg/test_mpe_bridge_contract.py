from __future__ import annotations

import pytest

from .mpe_bridge_contract import (
    METHODS,
    async_completed_packets,
    barrier_schedule,
    charge_completed_packets,
    packet_scale,
    service_bases,
    trajectory_seed,
)


def test_public_task_shapes_and_profiles_are_explicit() -> None:
    assert service_bases("simple_spread_v2", "balanced") == (1.0, 1.0, 1.0)
    assert service_bases("simple_spread_v2", "heterogeneous") == (1.0, 1.55, 4.0)
    assert service_bases("simple_reference_v2", "heterogeneous") == (1.0, 4.0)


def test_common_random_number_seed_does_not_depend_on_method() -> None:
    seeds = {
        trajectory_seed(
            7103,
            task="simple_spread_v2",
            owner=1,
            owner_packet_index=4,
            replicate=replicate,
        )
        for _method in METHODS
        for replicate in (0, 1)
    }
    assert len(seeds) == 2


def test_async_opportunities_follow_physical_service_clocks() -> None:
    assert async_completed_packets((1.0, 1.55, 4.0), 8.0) == (8, 5, 2)


def test_frozen_barrier_charges_unfinished_tail_work() -> None:
    schedule = barrier_schedule((1.0, 1.55, 4.0), 8.0, 25)
    assert schedule["rounds"] == 2
    assert schedule["completed_by_owner"] == (8, 4, 2)
    assert schedule["cancelled_environment_steps_by_owner"] == (0, 28, 0)


def test_packet_scales_are_outcome_independent() -> None:
    assert packet_scale("raw_async", event_delay=5, offdiag_scale=0.4) == 1.0
    assert packet_scale("delay_scaled_async", event_delay=4, offdiag_scale=0.4) == 0.2
    assert packet_scale("offdiag_async", event_delay=4, offdiag_scale=0.4) == 0.4
    with pytest.raises(ValueError):
        packet_scale("frozen_barrier", event_delay=0, offdiag_scale=0.4)


def test_two_trajectory_packet_charge_counts_all_actor_transitions() -> None:
    charge = charge_completed_packets((8, 5, 2), 25)
    assert charge.completed_packets == 15
    assert charge.environment_steps == 750
    assert charge.actor_transitions == 2250
