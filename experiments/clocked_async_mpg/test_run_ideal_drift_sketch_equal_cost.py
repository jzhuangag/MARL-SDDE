from __future__ import annotations

import numpy as np
import pytest

from .run_coupled_actor_critic_headroom import _initial_state, _next_owner, frozen_scenarios
from .run_ideal_drift_sketch_equal_cost import (
    _ideal_sketch_coefficients,
    _true_split_coefficients,
    load_config,
    simulate_ideal_sketch,
    validate,
)


def test_frozen_configuration_is_exhaustive_and_unique() -> None:
    payload = validate()
    assert payload["outcome_free"] is True
    assert payload["source_scenario_hash_matches"] is True
    assert payload["seed_count"] == 16
    assert payload["seeds_unique"] is True
    assert payload["split_total"] == 256
    assert payload["split_exhaustive"] is True


def test_deterministic_initial_state_gives_exact_sketch_coefficients() -> None:
    scenario = frozen_scenarios()[0]
    state = _initial_state(scenario)
    owner, _ = _next_owner(state)
    multiplier = 256.0 / 230.0
    true_linear, true_quadratic = _true_split_coefficients(
        scenario, state, owner, multiplier
    )
    estimated_linear, estimated_quadratic = _ideal_sketch_coefficients(
        scenario,
        state,
        owner,
        s1=13,
        s2=13,
        variance_multiplier=multiplier,
        rng=np.random.default_rng(77),
    )
    assert estimated_linear == pytest.approx(true_linear, abs=1e-12)
    assert estimated_quadratic == pytest.approx(true_quadratic, abs=1e-12)


def test_sampled_quadratic_is_psd_and_simulation_is_reproducible() -> None:
    scenario = frozen_scenarios()[37]
    first = simulate_ideal_sketch(scenario, 905001)
    second = simulate_ideal_sketch(scenario, 905001)
    assert first == second
    assert first["minimum_sampled_qp_eigenvalue"] >= -1e-9
    assert np.isfinite(first["normalized_auc"])
    assert first["variance_multiplier"] == pytest.approx(256.0 / 230.0)


def test_zero_target_full_and_diagonal_sketch_reduce_exactly() -> None:
    controls = [
        scenario for scenario in frozen_scenarios() if scenario.population == "zero_target"
    ]
    for scenario in controls[:3]:
        full = simulate_ideal_sketch(scenario, 905003)
        diagonal = simulate_ideal_sketch(scenario, 905003, diagonalize=True)
        assert full["normalized_auc"] == pytest.approx(
            diagonal["normalized_auc"], abs=1e-12
        )
        assert full["normalized_terminal"] == pytest.approx(
            diagonal["normalized_terminal"], abs=1e-12
        )


def test_stop_gates_are_not_weakened_from_successor_design() -> None:
    gates = load_config()["gates"]
    assert gates["I3"]["maximum"] == pytest.approx(0.90)
    assert gates["I4"]["maximum"] == pytest.approx(0.97)
    assert gates["I5"]["minimum"] == pytest.approx(0.60)
    assert gates["I6"]["minimum"] == pytest.approx(0.60)
    assert gates["I7"]["minimum"] == pytest.approx(0.70)
    assert gates["I7"]["per_cell_regret_fraction_maximum"] == pytest.approx(0.25)
