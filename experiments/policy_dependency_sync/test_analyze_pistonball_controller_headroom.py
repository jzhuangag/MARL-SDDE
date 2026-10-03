from __future__ import annotations

import json
from pathlib import Path

from .analyze_pistonball_controller_headroom import (
    ALL_CODES,
    SCHEDULER_BY_CODE,
    SEEDS,
    analyze,
)


def test_headroom_analyzer_requires_broad_equal_resource_gain(
    tmp_path: Path,
) -> None:
    terminal = {
        "signed": 7.0,
        "signed_only": 6.5,
        "cache": 5.0,
        "mismatch_terminal": 4.5,
        "complete_terminal": 4.0,
        "age_prefix": 3.5,
        "mismatch_prefix": 4.25,
        "round_prefix": 3.0,
        "static_prefix": 2.5,
        "no_refresh": 0.0,
        "full_refresh": 10.0,
    }
    paths: list[Path] = []
    for seed in SEEDS:
        for code in ALL_CODES:
            equal_resource = code != "full_refresh"
            enforcement = (
                "prefix"
                if code.endswith("_prefix")
                or code in {"no_refresh", "full_refresh"}
                else "terminal"
            )
            spent = 0 if code == "no_refresh" else (500 if equal_resource else 77000)
            record = {
                "headroom_code": code,
                "scheduler": SCHEDULER_BY_CODE[code],
                "seed": seed,
                "development_only": True,
                "finite": True,
                "budget_feasible": True,
                "remaining_packets_after_drain": 0,
                "received_packets": 4089,
                "launched_gradient_packets": 4089,
                "launched_actor_transitions": 4096 * 4 * 20,
                "allowed_refresh_units": 2048 if equal_resource else 77824,
                "spent_refresh_units": spent,
                "optional_policy_bytes": spent * 4,
                "runtime_seconds": 15.0 if code == "signed" else 10.0,
                "evaluations": [
                    {"actor_transitions": 0, "return": 0.0},
                    {"actor_transitions": 327680, "return": terminal[code]},
                ],
                "config": {
                    "launches": 4096,
                    "n_agents": 20,
                    "rollout_horizon": 4,
                    "network_activation": "silu",
                    "actor_receipt_step": 0.01,
                    "critic_step": 0.0003,
                    "maximum_delay": 8,
                    "cone_shell": 20 if code == "full_refresh" else 6,
                    "budget_rate": 0.5 if equal_resource else 19.0,
                    "budget_enforcement": enforcement,
                    "queue_step": 1.0,
                    "lyapunov_weight": 100000000.0,
                    "cache_debt_weight": 83650.1230871728,
                },
            }
            path = tmp_path / f"{seed}_{code}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            paths.append(path)
    result = analyze(paths)
    assert result["passed"]
    assert all(result["gates"].values())
    assert result["best_strong_terminal_code"] == "cache"
    assert result["mean_freshness_headroom"] == 10.0
    assert result["mean_signed_minus_best_strong"] == 2.0
