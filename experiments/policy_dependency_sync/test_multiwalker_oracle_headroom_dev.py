from __future__ import annotations

import math

from .multiwalker_oracle_headroom_dev import POLICIES, analyze, run_scenario


def test_multiwalker_counterfactual_headroom_is_reproducible_and_charged() -> None:
    first = run_scenario(
        seed=95900,
        drift_scale=0.04,
        budget_rate=0.5,
        policy="oracle_h8",
        walkers=3,
        events=6,
        horizon=2,
    )
    second = run_scenario(
        seed=95900,
        drift_scale=0.04,
        budget_rate=0.5,
        policy="oracle_h8",
        walkers=3,
        events=6,
        horizon=2,
    )
    assert first == second
    assert first.replay_failures == 0
    assert first.spent_edges <= math.floor(first.budget_rate * first.launches)
    assert first.optional_policy_bytes >= 0


def test_multiwalker_headroom_analyzer_uses_strong_online_envelope() -> None:
    rows = [
        run_scenario(
            seed=95901,
            drift_scale=0.04,
            budget_rate=0.5,
            policy=policy,
            walkers=3,
            events=5,
            horizon=2,
        )
        for policy in POLICIES
    ]
    result = analyze(rows)
    assert result["rows"] == len(POLICIES)
    assert result["cells"] == 1
    assert result["gates"]["D1_complete_finite"]
    assert result["gates"]["D2_exact_replay"]
    assert result["gates"]["D3_prefix_budget"]
    assert result["cell_metrics"][0]["strong_online_policy"] != "oracle_h8"
