from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .parallel_commit_headroom import (
    _run_async,
    evaluate_scenario,
    make_scenario,
    potential_gap,
    run,
    validate_config,
)


CONFIG = Path(__file__).with_name("parallel_commit_headroom_config.json")


def test_scenario_is_positive_definite_and_deterministic() -> None:
    first = make_scenario(
        seed=7,
        agents=4,
        interaction_strength=0.75,
        anisotropy=5.0,
        service_profile="two_tier",
    )
    second = make_scenario(
        seed=7,
        agents=4,
        interaction_strength=0.75,
        anisotropy=5.0,
        service_profile="two_tier",
    )
    assert np.min(np.linalg.eigvalsh(first.hessian)) > 0.0
    np.testing.assert_array_equal(first.initial, second.initial)
    assert potential_gap(first.optimum, first) == 0.0


def test_equal_service_methods_receive_equal_proposal_opportunities() -> None:
    scenario = make_scenario(
        seed=11,
        agents=4,
        interaction_strength=0.25,
        anisotropy=1.0,
        service_profile="balanced",
    )
    oracle = _run_async(scenario, method="bound_oracle", horizon=12, step_cap=0.2)
    causal = _run_async(
        scenario,
        method="causal_lyapunov",
        horizon=12,
        step_cap=0.2,
    )
    fixed = _run_async(
        scenario,
        method="fixed_async",
        horizon=12,
        step_cap=0.2,
        fixed_scale=0.5,
    )
    assert oracle["completed_proposals"] == causal["completed_proposals"]
    assert oracle["completed_proposals"] == fixed["completed_proposals"]
    assert len(oracle["gap_curve"]) == 13


def test_scenario_evaluation_includes_strong_oracle_baselines() -> None:
    scenario = make_scenario(
        seed=13,
        agents=4,
        interaction_strength=0.75,
        anisotropy=5.0,
        service_profile="skewed",
    )
    row = evaluate_scenario(
        scenario,
        horizon=20,
        step_cap=0.25,
        fixed_scale_grid=[0.2, 0.5, 1.0],
        tradeoff=4.0,
        risk_budget=0.02,
    )
    assert row["best_fixed_async"]["selected_scale"] in {0.2, 0.5, 1.0}
    assert sorted(row["best_sequential"]["selected_order"]) == list(
        range(scenario.agents)
    )
    assert row["strong_static_gap_auc"] == min(
        row["best_fixed_async"]["gap_auc"],
        row["best_sequential"]["gap_auc"],
    )


def test_development_grid_is_complete_and_reproducible_on_small_slice() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validation = validate_config(config)
    assert validation["scenario_count"] == 192
    small = dict(config)
    small["development_seeds"] = [19031]
    small["agent_counts"] = [4]
    small["interaction_strengths"] = [0.25]
    small["anisotropies"] = [1.0]
    small["service_profiles"] = ["balanced"]
    first = run(small)
    second = run(small)
    assert first == second
    assert first["summary"]["scenario_count"] == 1
