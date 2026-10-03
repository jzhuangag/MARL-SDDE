from __future__ import annotations

import numpy as np
import pytest

from .run_coupled_actor_critic_headroom import (
    ALPHA_CAP,
    BETA_CAP,
    _advance,
    _drift_qp,
    _initial_state,
    _next_owner,
    _risk,
    frozen_scenarios,
    simulate,
    validate,
)


def test_frozen_population_counts_and_hash() -> None:
    payload = validate()
    assert payload["outcome_free"] is True
    assert payload["counts"] == {
        "primary": 128,
        "zero_target": 16,
        "zero_interaction": 16,
    }
    assert len(payload["scenario_hash"]) == 64


def test_every_frozen_qp_is_psd_at_initial_event() -> None:
    for scenario in frozen_scenarios():
        state = _initial_state(scenario)
        owner, _ = _next_owner(state)
        _, quadratic, _ = _drift_qp(scenario, state, owner)
        assert np.linalg.eigvalsh(quadratic)[0] >= -1e-10


def test_zero_target_reduces_exactly_to_diagonal_online() -> None:
    controls = [
        scenario for scenario in frozen_scenarios() if scenario.population == "zero_target"
    ]
    for scenario in controls:
        coupled = simulate(scenario, "coupled")
        diagonal = simulate(scenario, "diagonal_online")
        for field in ("normalized_auc", "normalized_terminal", "mean_alpha", "mean_beta"):
            assert coupled[field] == pytest.approx(diagonal[field], abs=1e-12)
        assert coupled["max_abs_cross_curvature"] == pytest.approx(0.0, abs=1e-12)


def test_one_event_drift_matches_exact_moment_propagation() -> None:
    scenario = frozen_scenarios()[37]
    state = _initial_state(scenario)
    owner, _ = _next_owner(state)
    linear, quadratic, _ = _drift_qp(scenario, state, owner)
    alpha, beta = 0.23, 0.61
    predicted = float(
        linear @ np.asarray([alpha, beta])
        + 0.5 * np.asarray([alpha, beta]) @ quadratic @ np.asarray([alpha, beta])
    )
    exact = _risk(scenario, _advance(scenario, state, owner, alpha, beta)) - _risk(
        scenario, state
    )
    assert predicted == pytest.approx(exact, abs=1e-11)


def test_dynamic_actions_remain_in_frozen_box() -> None:
    for scenario in frozen_scenarios()[::17]:
        for method in ("coupled", "diagonal_online"):
            result = simulate(scenario, method)
            assert 0.0 <= float(result["mean_alpha"]) <= ALPHA_CAP
            assert 0.0 <= float(result["mean_beta"]) <= BETA_CAP
            assert np.isfinite(float(result["normalized_auc"]))

