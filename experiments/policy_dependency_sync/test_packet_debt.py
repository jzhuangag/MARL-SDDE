from __future__ import annotations

from itertools import combinations
from random import Random

import pytest

from .packet_debt import (
    HorizonCertificate,
    bias_square_upper,
    choose_horizon_and_graph,
    direct_full_cap_drift_upper,
    full_cap_receipt_bound,
    packet_debt,
    packet_history_energy,
    remaining_packet_coefficients,
)


def certificate(
    horizon: int = 4,
    max_edges: int | None = None,
) -> HorizonCertificate:
    return HorizonCertificate(
        horizon=horizon,
        base_bias_components=(0.1, 0.2),
        gradient_variance=1.5 / horizon,
        smoothness=2.0,
        step_cap=0.1,
        stale_radius_by_edge={(0, 1): 0.5, (2, 1): 0.3},
        fresh_radius_by_edge={(0, 1): 0.1, (2, 1): 0.2},
        cost_by_edge={(0, 1): 0.4, (2, 1): 0.1},
        max_edges=max_edges,
    )


def test_bias_square_is_valid_cauchy_bound() -> None:
    item = certificate()
    square = bias_square_upper(item, ((0, 1),))
    scalar_sum = 0.1 + 0.2 + 0.1 + 0.3
    assert square >= scalar_sum * scalar_sum


def test_packet_energy_coefficients() -> None:
    assert packet_history_energy(0.1, 2.0, 3.0, 4.0) == pytest.approx(0.31)


def test_launch_rule_matches_brute_force() -> None:
    items = [certificate(4), certificate(8)]
    queue = 0.3
    weight = 2.0
    chosen = choose_horizon_and_graph(items, queue, weight)
    brute = []
    for item in items:
        edges = tuple(item.stale_radius_by_edge)
        for size in range(len(edges) + 1):
            for selected in combinations(edges, size):
                square, debt = packet_debt(item, selected)
                cost = sum(item.cost_by_edge[edge] for edge in selected)
                brute.append((weight * debt + queue * cost, item.horizon, selected, square))
    best = min(brute, key=lambda row: (row[0], row[1], tuple(map(repr, row[2]))))
    assert chosen.drift_plus_penalty_index == pytest.approx(best[0])
    assert chosen.horizon == best[1]
    assert chosen.refreshed_edges == best[2]
    assert chosen.bias_square_upper == pytest.approx(best[3])


def test_launch_rule_matches_random_brute_force_instances() -> None:
    random = Random(918273)
    for trial in range(25):
        items = []
        for horizon in (2, 5, 9):
            edges = tuple((edge, trial) for edge in range(3))
            items.append(
                HorizonCertificate(
                    horizon=horizon,
                    base_bias_components=(random.random(), random.random()),
                    gradient_variance=random.random(),
                    smoothness=0.5 + random.random(),
                    step_cap=0.01 + 0.1 * random.random(),
                    stale_radius_by_edge={edge: random.random() for edge in edges},
                    fresh_radius_by_edge={edge: random.random() for edge in edges},
                    cost_by_edge={edge: 0.1 + random.random() for edge in edges},
                    max_edges=trial % 4,
                )
            )
        queue = 2.0 * random.random()
        weight = 0.5 + random.random()
        chosen = choose_horizon_and_graph(items, queue, weight)
        brute = []
        for item in items:
            edges = tuple(item.stale_radius_by_edge)
            cap = len(edges) if item.max_edges is None else item.max_edges
            for size in range(min(cap, len(edges)) + 1):
                for selected in combinations(edges, size):
                    square, debt = packet_debt(item, selected)
                    cost = sum(item.cost_by_edge[edge] for edge in selected)
                    brute.append(
                        (weight * debt + queue * cost, item.horizon, selected, square)
                    )
        best = min(
            brute,
            key=lambda row: (row[0], row[1], tuple(map(repr, row[2]))),
        )
        assert chosen.drift_plus_penalty_index == pytest.approx(best[0])
        assert chosen.horizon == best[1]
        assert chosen.refreshed_edges == best[2]
        assert chosen.bias_square_upper == pytest.approx(best[3])


def test_cardinality_cap_takes_best_net_edge() -> None:
    chosen = choose_horizon_and_graph([certificate(max_edges=1)], 0.0, 1.0)
    assert chosen.refreshed_edges == ((0, 1),)


def test_large_queue_selects_no_refresh() -> None:
    chosen = choose_horizon_and_graph([certificate()], 1_000.0, 1.0)
    assert chosen.refreshed_edges == ()
    assert chosen.communication_cost == 0.0


def test_remaining_packet_coefficients_match_direct_expansion() -> None:
    caps = [0.1, 0.2]
    multipliers = [3.0, 2.0]
    sensitivities = [0.5, 0.25]
    distances = [0.4, 0.7]
    linear, curvature = remaining_packet_coefficients(
        caps, multipliers, sensitivities, distances
    )
    movement = 0.03
    before = sum(
        1.25 * cap * multiplier * sensitivity**2 * distance**2
        for cap, multiplier, sensitivity, distance in zip(
            caps, multipliers, sensitivities, distances, strict=True
        )
    )
    after = sum(
        1.25 * cap * multiplier * sensitivity**2 * (distance + movement) ** 2
        for cap, multiplier, sensitivity, distance in zip(
            caps, multipliers, sensitivities, distances, strict=True
        )
    )
    assert after - before == pytest.approx(
        linear * movement + 0.5 * curvature * movement**2
    )


@pytest.mark.parametrize("gradient", [-2.0, -0.3, 0.0, 0.7, 2.0])
@pytest.mark.parametrize("bias", [-0.4, -0.1, 0.0, 0.2, 0.4])
@pytest.mark.parametrize("noise_std", [0.0, 0.2, 0.8])
def test_receipt_stationarity_bound_dominates_direct_bound(
    gradient: float,
    bias: float,
    noise_std: float,
) -> None:
    smoothness = 2.0
    cap = 0.1
    linear = 0.04
    curvature = 0.5
    direct = direct_full_cap_drift_upper(
        gradient,
        bias,
        noise_std,
        smoothness,
        cap,
        linear,
        curvature,
    )
    theorem = full_cap_receipt_bound(
        abs(gradient),
        abs(bias),
        noise_std,
        smoothness,
        cap,
        linear,
        curvature,
    )
    assert direct <= theorem.drift_upper + 1e-12


def test_receipt_bound_rejects_unstable_caps() -> None:
    with pytest.raises(ValueError, match=r"L \* step_cap"):
        full_cap_receipt_bound(1.0, 0.1, 0.2, 3.0, 0.1)
    with pytest.raises(ValueError, match=r"kappa \* step_cap"):
        full_cap_receipt_bound(1.0, 0.1, 0.2, 1.0, 0.1, 0.0, 2.0)
