from __future__ import annotations

import json
from pathlib import Path

from .analyze_pistonball_exact_score_audit import analyze


def _record(*, scheduler: str, runtime: float) -> dict[str, object]:
    return {
        "scheduler": scheduler,
        "seed": 79008,
        "development_only": True,
        "finite": True,
        "budget_feasible": True,
        "remaining_packets_after_drain": 0,
        "received_packets": 1000,
        "launched_gradient_packets": 1000,
        "launched_actor_transitions": 1024 * 4 * 20,
        "spent_refresh_units": 2 if scheduler == "exact_lyapunov" else 0,
        "allowed_refresh_units": 512,
        "runtime_seconds": runtime,
        "evaluations": [
            {"actor_transitions": 0, "return": -8.0},
            {"actor_transitions": 1024 * 4 * 20, "return": -3.0},
        ],
        "config": {
            "network_activation": "silu",
            "launches": 1024,
            "n_agents": 20,
            "rollout_horizon": 4,
            "max_cycles": 125,
            "batch_size": 32,
            "evaluations": 2,
            "evaluation_episodes": 1,
            "cone_shell": 6,
            "maximum_delay": 8,
            "actor_receipt_step": 0.01,
            "critic_step": 0.0003,
            "budget_rate": 0.5,
            "budget_enforcement": "terminal",
            "queue_step": 1.0,
            "lyapunov_weight": 1.0,
            "cache_debt_weight": 1.0,
            "random_drop": True,
            "random_rotate": True,
        },
        "launch_trace": [],
    }


def test_exact_audit_is_outcome_blind_and_reconstructs_sparse_index(
    tmp_path: Path,
) -> None:
    exact = _record(scheduler="exact_lyapunov", runtime=2.5)
    exact["launch_trace"] = [
        {
            "score_diagnostic": {
                "selected_donor": 2,
                "null_index": 4.0,
                "best_edge_index": 2.5,
                "best_edge_learning_index_delta": -1.0,
                "best_edge_cache_reset_benefit": 1.0,
                "best_edge_queue_price": 0.5,
                "candidate_count": 7,
                "vjp_calls": 8,
            }
        },
        {
            "score_diagnostic": {
                "selected_donor": None,
                "null_index": 4.0,
                "best_edge_index": 5.0,
                "best_edge_learning_index_delta": 0.5,
                "best_edge_cache_reset_benefit": 0.5,
                "best_edge_queue_price": 1.0,
                "candidate_count": 4,
                "vjp_calls": 5,
            }
        },
    ]
    control = _record(scheduler="no_refresh", runtime=1.0)
    exact_path = tmp_path / "exact.json"
    control_path = tmp_path / "control.json"
    exact_path.write_text(json.dumps(exact), encoding="utf-8")
    control_path.write_text(json.dumps(control), encoding="utf-8")

    result = analyze(exact_path, control_path)

    assert result["passed"]
    assert result["outcome_blind_calibration"]
    assert result["recommended_learning_weight"] == 4.0 / 3.0
    assert result["recommended_cache_debt_weight"] == 4.0 / 3.0
    assert result["runtime_ratio_to_no_refresh"] == 2.5
    assert result["descriptive_returns_not_used_by_gates"]["exact"][1][
        "return"
    ] == -3.0


def test_exact_audit_fails_closed_when_signed_component_is_zero(
    tmp_path: Path,
) -> None:
    exact = _record(scheduler="exact_lyapunov", runtime=1.0)
    exact["launch_trace"] = [
        {
            "score_diagnostic": {
                "selected_donor": None,
                "null_index": 4.0,
                "best_edge_index": 3.0,
                "best_edge_learning_index_delta": 0.0,
                "best_edge_cache_reset_benefit": 1.0,
                "best_edge_queue_price": 0.0,
                "candidate_count": 2,
                "vjp_calls": 3,
            }
        }
    ]
    control = _record(scheduler="no_refresh", runtime=1.0)
    exact_path = tmp_path / "zero.json"
    control_path = tmp_path / "control.json"
    exact_path.write_text(json.dumps(exact), encoding="utf-8")
    control_path.write_text(json.dumps(control), encoding="utf-8")

    result = analyze(exact_path, control_path)

    assert not result["passed"]
    assert not result["gates"]["E1_nonzero_signed_component"]
    assert result["recommended_learning_weight"] is None
    assert result["decision"] == "stop the Pistonball policy-cache mainline"
