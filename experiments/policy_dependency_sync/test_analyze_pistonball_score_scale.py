from __future__ import annotations

import json
from pathlib import Path

import pytest

from .analyze_pistonball_score_scale import analyze


def test_score_scale_audit_reconstructs_index_and_ignores_return(tmp_path: Path) -> None:
    record = {
        "scheduler": "signed_lyapunov",
        "seed": 79004,
        "development_only": True,
        "finite": True,
        "budget_feasible": True,
        "remaining_packets_after_drain": 0,
        "spent_refresh_units": 1,
        "allowed_refresh_units": 2,
        "optional_policy_bytes": 12,
        "runtime_seconds": 3.0,
        "evaluations": [{"return": -999.0}],
        "config": {
            "launches": 2048,
            "n_agents": 20,
            "rollout_horizon": 4,
            "actor_receipt_step": 0.01,
            "critic_step": 0.0003,
            "budget_rate": 0.5,
            "budget_enforcement": "terminal",
            "queue_step": 1.0,
            "maximum_delay": 8,
            "lyapunov_weight": 100000.0,
            "cache_debt_weight": 100000.0,
        },
        "launch_trace": [
            {
                "score_diagnostic": {
                    "selected_donor": 1,
                    "null_index": 10.0,
                    "best_edge_index": 8.5,
                    "best_edge_learning_index_delta": -2.0,
                    "best_edge_cache_reset_benefit": 0.5,
                    "best_edge_queue_price": 1.0,
                }
            },
            {
                "score_diagnostic": {
                    "selected_donor": None,
                    "null_index": 10.0,
                    "best_edge_index": 12.0,
                    "best_edge_learning_index_delta": 1.0,
                    "best_edge_cache_reset_benefit": 0.0,
                    "best_edge_queue_price": 1.0,
                }
            },
        ],
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    result = analyze(source)
    assert result["outcome_blind_calibration"]
    assert result["selected_fraction_under_legacy_scale"] == 0.5
    assert result["signed_favorable_fraction"] == 0.5
    assert result["raw_learning_drift_delta"]["median"] == pytest.approx(-5e-6)
    assert result["descriptive_return_path"] == [{"return": -999.0}]
