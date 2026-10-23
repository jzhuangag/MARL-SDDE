from __future__ import annotations

import numpy as np
import pytest

from .factored_markov_packet import (
    actual_conditional_bias,
    average_offset_occupancy,
    certificate_dominates_actual_bias,
    conditional_owner_row,
    donor_order,
    global_hessian,
    horizon_certificate,
    prospective_offsets,
)


def test_average_occupancy_is_probability_and_h1_is_launch_state() -> None:
    occupancy = average_offset_occupancy(5, 2, 0.4, 1)
    assert occupancy.sum() == pytest.approx(1.0)
    assert occupancy[2] == 1.0


def test_stationary_average_of_conditional_rows_is_global_row() -> None:
    agents = 6
    rows = [
        conditional_owner_row(2, agents, 0.4, 0.9, start, 0.63, 5)
        for start in range(agents - 1)
    ]
    assert np.mean(rows, axis=0) == pytest.approx(
        global_hessian(agents, 0.4, 0.9)[2]
    )


def test_slow_motion_has_smaller_prospective_cone() -> None:
    slow = prospective_offsets(9, 2, 0.1, 8, 0.9)
    fast = prospective_offsets(9, 2, 0.8, 8, 0.9)
    assert len(slow) < len(fast)


@pytest.mark.parametrize("owner", range(6))
@pytest.mark.parametrize("start_offset", range(5))
@pytest.mark.parametrize("move_probability", [0.1, 0.45, 0.8])
@pytest.mark.parametrize("horizon", [2, 4, 8])
def test_certificate_dominates_exact_birth_bias(
    owner: int,
    start_offset: int,
    move_probability: float,
    horizon: int,
) -> None:
    agents = 6
    theta = np.asarray([1.1, -0.8, 0.5, -1.3, 0.7, -0.2])
    target = np.asarray([0.4, -0.2, 0.8, -0.5, 0.1, -0.7])
    cache = theta + np.asarray([0.2, -0.1, 0.3, -0.25, 0.15, -0.05])
    item, row = horizon_certificate(
        owner=owner,
        current_parameter=theta,
        target=target,
        cache_for_owner=cache,
        start_offset=start_offset,
        move_probability=move_probability,
        horizon=horizon,
        cone_coverage=0.9,
        strong_convexity=0.4,
        coupling=0.9,
        innovation_variance=0.5,
        temporal_correlation=0.6,
        step_cap=0.05,
    )
    edges = tuple(item.stale_radius_by_edge)
    for mask in range(1 << len(edges)):
        selected = tuple(edge for index, edge in enumerate(edges) if mask & (1 << index))
        bias = actual_conditional_bias(
            owner,
            theta,
            target,
            cache,
            row,
            selected,
            0.4,
            0.9,
        )
        assert certificate_dominates_actual_bias(item, bias, selected)


def test_donor_order_excludes_owner_exactly_once() -> None:
    assert donor_order(3, 6) == (4, 5, 0, 1, 2)

