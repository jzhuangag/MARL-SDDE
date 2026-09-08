from __future__ import annotations

import json
from dataclasses import asdict

from .multiwalker_budget_oracle_amendment import ExactOracleRow
from .multiwalker_oracle_confirmation import (
    BUDGET_RATES,
    CONFIRMATION_SEEDS,
    DRIFT_SCALES,
    analyze_confirmation,
)
from .multiwalker_oracle_headroom_dev import POLICIES, ScenarioResult


def _baseline(seed: int, drift: float, budget: float, policy: str) -> ScenarioResult:
    value = 0.0 if policy == "no_refresh" else 0.5
    return ScenarioResult(
        seed=seed,
        drift_scale=drift,
        budget_rate=budget,
        policy=policy,
        selected_h_return=value,
        no_action_h_return=0.0,
        captured_immediate_gain=value,
        spent_edges=0,
        optional_policy_bytes=0,
        positive_choices=0,
        nonnull_choices=0,
        launches=40,
        replay_failures=0,
    )


def _exact(seed: int, drift: float, budget: float) -> ExactOracleRow:
    return ExactOracleRow(
        seed=seed,
        drift_scale=drift,
        budget_rate=budget,
        exact_oracle_h_return=1.0,
        exact_oracle_spent_edges=0,
        strong_online_policy="age",
        strong_online_h_return=0.5,
        no_refresh_h_return=0.0,
        oracle_gain_over_no_refresh=1.0,
        oracle_headroom_over_strong=0.5,
        oracle_recovery_gap=0.5,
        normalized_absolute_headroom=0.5,
        optimizer_success=True,
        optimizer_status=0,
        optimizer_mip_gap=0.0,
        selected_owner_count=5,
        maximum_prefix_excess=0,
        no_refresh_replay_error=0.0,
    )


def test_confirmation_seed_registry_is_disjoint_from_development() -> None:
    assert CONFIRMATION_SEEDS == tuple(range(96100, 96108))
    assert set(CONFIRMATION_SEEDS).isdisjoint(range(95700, 95708))


def test_confirmation_analyzer_enforces_frozen_complete_population() -> None:
    baseline = [
        _baseline(seed, drift, budget, policy)
        for seed in CONFIRMATION_SEEDS
        for drift in DRIFT_SCALES
        for budget in BUDGET_RATES
        for policy in POLICIES
    ]
    exact = [
        _exact(seed, drift, budget)
        for seed in CONFIRMATION_SEEDS
        for drift in DRIFT_SCALES
        for budget in BUDGET_RATES
    ]
    summary = analyze_confirmation(baseline, exact)
    assert summary["all_gates_pass"]
    assert summary["baseline_rows"] == 352
    assert summary["exact_rows"] == 32
    assert summary["active_direction_rate"] == 1.0


def test_confirmation_records_have_canonical_json_types() -> None:
    payload = asdict(_exact(CONFIRMATION_SEEDS[0], 0.04, 0.5))
    assert json.loads(json.dumps(payload, allow_nan=False))["optimizer_success"] is True
