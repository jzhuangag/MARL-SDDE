from __future__ import annotations

import numpy as np
import pytest

from .controlled_sensing_upper_bound import (
    belief_state_probabilities,
    best_periodic_fixed_cost,
    exact_phase_cost,
    phase_log_multiplier_table,
    solve_perfect_observation_bound,
)


def test_belief_states_converge_to_the_stationary_fraction() -> None:
    beliefs = belief_state_probabilities(
        persistence=0.8, rotation_fraction=0.3, maximum_age=100
    )
    assert beliefs[0] == pytest.approx(0.3)
    assert beliefs[100] == pytest.approx(0.3, abs=1e-9)
    assert beliefs[200] == pytest.approx(0.3, abs=1e-9)


def test_zero_budget_matches_never_optimism() -> None:
    table = phase_log_multiplier_table(0.5, 0.5)
    bound = solve_perfect_observation_bound(
        table,
        persistence=0.9,
        rotation_fraction=0.4,
        optimism_budget=0.0,
        maximum_age=32,
    )
    never = float(0.6 * table[0, 0] + 0.4 * table[1, 0])
    assert bound.optimal_log_cost == pytest.approx(never, abs=1e-10)
    assert bound.call_rate == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("rotation_fraction", (0.25, 0.5, 0.75))
def test_perfect_paid_sensing_lies_between_exact_phase_and_fixed(
    rotation_fraction: float,
) -> None:
    table = phase_log_multiplier_table(0.5, 0.3)
    sensing = solve_perfect_observation_bound(
        table,
        persistence=0.9,
        rotation_fraction=rotation_fraction,
        optimism_budget=0.25,
        maximum_age=64,
    )
    fixed, _ = best_periodic_fixed_cost(
        table,
        rotation_fraction=rotation_fraction,
        optimism_budget=0.25,
    )
    exact, _ = exact_phase_cost(
        table,
        rotation_fraction=rotation_fraction,
        optimism_budget=0.25,
    )
    assert exact <= sensing.optimal_log_cost + 1e-10
    assert sensing.optimal_log_cost <= fixed + 1e-10
    assert sensing.call_rate <= 0.25 + 1e-10
    assert sensing.flow_residual <= 1e-9
    assert sensing.normalization_residual <= 1e-9
    assert sensing.calibration_residual <= 1e-3


def test_age_truncation_is_numerically_stable_on_a_persistent_case() -> None:
    table = phase_log_multiplier_table(0.8, 0.1)
    values = [
        solve_perfect_observation_bound(
            table,
            persistence=0.95,
            rotation_fraction=0.5,
            optimism_budget=0.25,
            maximum_age=age,
        ).optimal_log_cost
        for age in (64, 128, 256)
    ]
    assert abs(values[2] - values[1]) <= abs(values[1] - values[0]) + 1e-12
    assert abs(values[2] - values[1]) < 1e-7


def test_stationary_potential_never_uses_harmful_optimism() -> None:
    table = phase_log_multiplier_table(0.5, 0.5)
    bound = solve_perfect_observation_bound(
        table,
        persistence=0.95,
        rotation_fraction=0.0,
        optimism_budget=0.5,
        maximum_age=32,
    )
    assert bound.call_rate == pytest.approx(0.0, abs=1e-12)
    assert bound.optimal_log_cost == pytest.approx(table[0, 0], abs=1e-10)
