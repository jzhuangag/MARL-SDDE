from __future__ import annotations

from .audit_pursuit_cached_rollout import run_audit


def test_short_actual_pursuit_cached_rollout_smoke() -> None:
    result = run_audit(
        seeds=(92999,),
        cycles=4,
        n_pursuers=8,
        n_evaders=30,
        observation_range=7,
        max_neighbors=4,
    )
    assert result["profile_steps"] == 4
    assert result["reward_values_read"] is False
    assert result["gates"]["B3_refreshed_action_matches_current"]
