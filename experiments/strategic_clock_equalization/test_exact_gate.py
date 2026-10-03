from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.strategic_clock_equalization.exact_gate import (
    Scenario,
    _simulate,
    availability_path,
    exact_logit_gradient,
    expected_reward,
    noise_path,
    scenarios_from_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "strategic_clock_oracle_gate_manifest_20260906.json"


def test_exact_gradient_matches_central_difference() -> None:
    reward = np.asarray(((0.4, -0.2), (-0.2, 2.0)))
    logits = np.asarray((-0.3, 0.7))
    analytic = exact_logit_gradient(logits, reward)
    epsilon = 1e-6
    numeric = []
    for index in range(2):
        offset = np.zeros(2)
        offset[index] = epsilon
        numeric.append(
            (expected_reward(logits + offset, reward) - expected_reward(logits - offset, reward))
            / (2.0 * epsilon)
        )
    assert analytic == pytest.approx(numeric, abs=1e-9)


def test_reversal_has_equal_counts_and_opposite_seed_orientation() -> None:
    left = availability_path("equal_count_early_burst_reversal", 400, 0.8, 91001)
    right = availability_path("equal_count_early_burst_reversal", 400, 0.8, 91002)
    assert np.bincount(left, minlength=2).tolist() == [200, 200]
    assert np.array_equal(left, 1 - right)


def test_paths_are_reproducible_and_scenario_specific() -> None:
    scenario = Scenario(((0.4, 0.0), (0.0, 1.0)), (0.35, 0.55), "symmetric_markov_bursts", 0.95, 0.05)
    assert np.array_equal(noise_path(91001, scenario, 32), noise_path(91001, scenario, 32))
    changed = Scenario(((0.7, 0.0), (0.0, 1.5)), (0.35, 0.55), "symmetric_markov_bursts", 0.95, 0.05)
    assert not np.array_equal(noise_path(91001, scenario, 32), noise_path(91001, changed, 32))


def test_scenario_manifest_is_finite_and_unique() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scenarios = scenarios_from_manifest(manifest)
    assert len(scenarios) == len({scenario.key for scenario in scenarios})
    assert any(s.population == "primary" for s in scenarios)
    assert any(s.population == "single_basin_control" for s in scenarios)
    assert any(s.population == "uncoupled_control" for s in scenarios)


def test_all_causal_methods_obey_mass_cap() -> None:
    scenario = Scenario(((0.4, 0.0), (0.0, 1.0)), (0.35, 0.55), "iid_stationary_80_20", 0.8, 0.05)
    for method in (
        "raw_async",
        "barrier_fresh_sequential",
        "true_stationary_rate_inverse",
        "ewma_rate_inverse",
        "count_only_debt",
        "lyapunov_strategic_clock",
    ):
        result = _simulate(
            scenario,
            91001,
            method,
            horizon=400,
            base_step=0.05,
            maximum_mass=0.2,
        )
        assert np.isfinite(result["auc"])
        # Barrier can apply two capped blocks at the same event; its recorded
        # per-event total mass is therefore allowed to be twice the block cap.
        bound = 0.4 if method == "barrier_fresh_sequential" else 0.2
        assert result["maximum_mass"] <= bound + 1e-12


def test_lyapunov_action_is_not_hardcoded_count_debt() -> None:
    scenario = Scenario(((0.4, -0.2), (-0.2, 2.0)), (0.35, 0.55), "equal_count_early_burst_reversal", 0.8, 0.0)
    result = _simulate(
        scenario,
        91001,
        "lyapunov_strategic_clock",
        horizon=400,
        base_step=0.05,
        maximum_mass=0.2,
    )
    assert result["performance_active_fraction"] > 0.0
    assert result["mass_difference_fraction"] > 0.0
