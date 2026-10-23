"""Static and algebraic tests for the outcome-free T-032 design."""

from __future__ import annotations

import math

import numpy as np

from experiments.dependence_delay_linear.t032_fresh_diversity_scan import (
    CONFIG,
    budget_limits,
    exact_full_risk,
    expected_cell_count,
    freshness_subset,
    greedy_joint_subset,
    innovation_autocovariance,
    profiles,
    usable_horizon,
)


def test_expected_cell_count_is_frozen() -> None:
    assert expected_cell_count() == 576


def test_layouts_preserve_delay_histogram() -> None:
    expected = sorted(np.resize(np.asarray(CONFIG["delay_profile"]), 32).tolist())
    for layout in CONFIG["layouts"]:
        observed = sorted(profile.delay for profile in profiles(32, layout, 1))
        assert observed == expected


def test_zero_delay_scale_is_homogeneous_in_delay() -> None:
    for layout in CONFIG["layouts"]:
        assert {profile.delay for profile in profiles(16, layout, 0)} == {0}


def test_single_agent_marginal_variance_is_invariant() -> None:
    for layout in CONFIG["layouts"]:
        profile = profiles(16, layout, 1)[7]
        for rho in CONFIG["rho_values"]:
            gamma = innovation_autocovariance([profile], rho, 0.8, 16)
            assert math.isclose(gamma[0], 1.0, rel_tol=0.0, abs_tol=1e-12)
            assert np.allclose(gamma, np.power(0.8, np.arange(16)), atol=1e-12)


def test_independent_equal_size_subsets_with_same_delays_match() -> None:
    agent_profiles = profiles(16, "balanced", 1)
    first = [agent_profiles[index] for index in (0, 1, 4, 5)]
    second = [agent_profiles[index] for index in (8, 9, 12, 13)]
    first_risk = exact_full_risk(first, rho=0.0, markov_lambda=0.8, horizon=80)
    second_risk = exact_full_risk(second, rho=0.0, markov_lambda=0.8, horizon=80)
    assert first_risk == second_risk


def test_alpha_zero_keeps_initial_error(monkeypatch) -> None:
    original = CONFIG["alpha"]
    monkeypatch.setitem(CONFIG, "alpha", 0.0)
    try:
        selected = profiles(16, "balanced", 1)[:4]
        risk = exact_full_risk(selected, rho=0.9, markov_lambda=0.8, horizon=30)
        assert risk["auc"] == CONFIG["initial_error"] ** 2
        assert risk["terminal"] == CONFIG["initial_error"] ** 2
    finally:
        CONFIG["alpha"] = original


def test_message_and_environment_horizons_match_with_fixed_m() -> None:
    agent_profiles = profiles(16, "balanced", 1)
    subset = [agent_profiles[index] for index in freshness_subset(agent_profiles, 4)]
    for ray in ("message", "environment"):
        horizon, _ = usable_horizon(subset, budget_limits(16, 4, 64, ray))
        assert horizon == 64


def test_wall_budget_penalizes_stale_subset() -> None:
    agent_profiles = profiles(16, "clustered", 1)
    fresh = [profile for profile in agent_profiles if profile.delay == 0][:4]
    stale = [profile for profile in agent_profiles if profile.delay == 6][:4]
    budgets = budget_limits(16, 4, 64, "wall")
    fresh_horizon, _ = usable_horizon(fresh, budgets)
    stale_horizon, _ = usable_horizon(stale, budgets)
    assert fresh_horizon == 64
    assert stale_horizon == 9


def test_diversity_greedy_spreads_highly_correlated_agents() -> None:
    agent_profiles = profiles(16, "balanced", 0)
    subset = greedy_joint_subset(agent_profiles, m=4, rho=0.9, kappa=0.0)
    groups = {agent_profiles[index].group for index in subset}
    assert len(groups) == 4


def test_freshness_ignores_groups_by_definition() -> None:
    agent_profiles = profiles(16, "clustered", 1)
    subset = freshness_subset(agent_profiles, 4)
    assert {agent_profiles[index].delay for index in subset} == {0}

