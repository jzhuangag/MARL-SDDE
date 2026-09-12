from __future__ import annotations

import json
from pathlib import Path

from .analyze_pistonball_smooth_interface import analyze


def test_smooth_interface_requires_learning_and_active_signed_score(
    tmp_path: Path,
) -> None:
    common_config = {
        "network_activation": "silu",
        "n_agents": 20,
        "rollout_horizon": 4,
        "actor_receipt_step": 0.01,
        "critic_step": 0.0003,
    }
    fresh = {
        "scheduler": "complete_burst",
        "seed": 79005,
        "development_only": True,
        "finite": True,
        "budget_feasible": True,
        "remaining_packets_after_drain": 0,
        "received_packets": 4089,
        "launched_gradient_packets": 4089,
        "launched_actor_transitions": 4096 * 4 * 20,
        "mean_packet_gradient_norm": 0.2,
        "actor_parameter_drift_l2": [0.1] * 20,
        "spent_refresh_units": 10,
        "optional_policy_bytes": 20,
        "runtime_seconds": 2.0,
        "evaluations": [{"return": -5.0}, {"return": -2.0}],
        "config": {
            **common_config,
            "launches": 4096,
            "budget_rate": 19.0,
            "budget_enforcement": "prefix",
            "maximum_delay": 0,
        },
    }
    scale = {
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
        "evaluations": [],
        "config": {
            **common_config,
            "launches": 2048,
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
                    "null_index": 4.0,
                    "best_edge_index": 2.0,
                    "best_edge_learning_index_delta": -1.0,
                    "best_edge_cache_reset_benefit": 1.0,
                    "best_edge_queue_price": 0.0,
                }
            },
            {
                "score_diagnostic": {
                    "selected_donor": None,
                    "null_index": 4.0,
                    "best_edge_index": 6.0,
                    "best_edge_learning_index_delta": 1.0,
                    "best_edge_cache_reset_benefit": 0.0,
                    "best_edge_queue_price": 1.0,
                }
            },
        ],
    }
    fresh_path = tmp_path / "fresh.json"
    scale_path = tmp_path / "scale.json"
    fresh_path.write_text(json.dumps(fresh), encoding="utf-8")
    scale_path.write_text(json.dumps(scale), encoding="utf-8")
    result = analyze(fresh_path, scale_path)
    assert result["qualified"]
    assert result["learner_gate"]["passed"]
    assert result["score_gate"]["passed"]
