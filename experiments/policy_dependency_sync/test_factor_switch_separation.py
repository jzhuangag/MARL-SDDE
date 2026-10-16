from __future__ import annotations

import pytest

from .factor_switch_separation import (
    binary_markov_paths,
    separation_table,
    simulate_path,
)


def test_binary_markov_path_mass_is_one() -> None:
    assert sum(probability for _, probability in binary_markov_paths(5, 0.83)) == pytest.approx(1.0)


def test_negative_alignment_is_rejected_by_joint_packet_weight() -> None:
    terminal, cumulative, cost = simulate_path(
        (0,),
        scheduler=lambda path, event: 1,
        initial_parameter=1.0,
        packet_magnitude=0.25,
        maximum_packet_weight=0.5,
    )
    assert terminal == pytest.approx(0.5)
    assert cumulative == pytest.approx(0.5)
    assert cost == 1.0


def test_matching_edge_makes_exact_quadratic_progress() -> None:
    terminal, _, cost = simulate_path(
        (0,),
        scheduler=lambda path, event: 0,
        initial_parameter=1.0,
        packet_magnitude=0.25,
        maximum_packet_weight=0.5,
    )
    assert terminal == pytest.approx(0.5 * 0.875**2)
    assert cost == 1.0


def test_joint_signed_strictly_beats_strong_equal_cost_baseline() -> None:
    result = separation_table(horizon=4, persistence=0.9)
    proposed = result["rows"]["joint_signed"]
    baseline = result["rows"][result["best_equal_cost_baseline"]]
    assert proposed["expected_refresh_cost"] == pytest.approx(4.0)
    assert baseline["expected_refresh_cost"] == pytest.approx(4.0)
    assert result["terminal_potential_improvement"] >= 0.10
    assert result["terminal_potential_ratio"] < 1.0
