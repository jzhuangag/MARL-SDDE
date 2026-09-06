from __future__ import annotations

import json
from pathlib import Path

from .analyze_pistonball_development import EXPECTED_SCHEDULERS, analyze


def test_development_analyzer_uses_higher_return_and_auc(tmp_path: Path) -> None:
    paths = []
    terminal = {
        "signed_lyapunov": 4.0,
        "signed_only": 2.0,
        "mismatch": 3.0,
        "complete_burst": 2.5,
        "no_refresh": 1.0,
    }
    for scheduler in sorted(EXPECTED_SCHEDULERS):
        record = {
            "scheduler": scheduler,
            "seed": 7,
            "allowed_refresh_units": 5,
            "launched_actor_transitions": 100,
            "spent_refresh_units": 4,
            "optional_policy_bytes": 40,
            "runtime_seconds": 2.0,
            "finite": True,
            "budget_feasible": True,
            "remaining_packets_after_drain": 0,
            "evaluations": [
                {"actor_transitions": 0, "return": 0.0},
                {"actor_transitions": 100, "return": terminal[scheduler]},
            ],
        }
        path = tmp_path / f"{scheduler}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
    result = analyze(paths)
    assert result["main_minus_best_terminal"] == 1.0
    assert result["main_minus_best_auc"] == 0.5
    assert result["main_learning_delta"] == 4.0
    assert not result["confirmatory"]
