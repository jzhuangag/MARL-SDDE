from __future__ import annotations

import json
from pathlib import Path

from .analyze_pistonball_backbone_qualification import EXPECTED_CONFIGS, analyze


def test_backbone_analyzer_applies_frozen_learning_delta(tmp_path: Path) -> None:
    paths = []
    for index, code in enumerate(sorted(EXPECTED_CONFIGS)):
        terminal = 3.0 if code == "fresh_s3" else 1.0
        record = {
            "qualification_code": code,
            "scheduler": "complete_burst" if code.startswith("fresh_") else "no_refresh",
            "seed": 5,
            "config": {"critic_step": 0.001, "actor_receipt_step": 0.003},
            "evaluations": [
                {"actor_transitions": 0, "return": 0.0},
                {"actor_transitions": 100, "return": terminal},
            ],
            "launched_actor_transitions": 100,
            "launched_gradient_packets": 9,
            "received_packets": 9,
            "remaining_packets_after_drain": 0,
            "finite": True,
            "budget_feasible": True,
            "deterministic_algorithms": True,
            "mean_td_loss": 0.5,
            "mean_packet_gradient_norm": 0.2,
            "actor_parameter_drift_l2": [0.1, 0.2],
            "spent_refresh_units": index,
            "optional_policy_bytes": 10 * index,
            "runtime_seconds": 1.0,
        }
        path = tmp_path / f"{code}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
    result = analyze(paths)
    assert result["qualified"]
    assert result["best_fresh_code"] == "fresh_s3"
    assert result["minimum_learning_delta"] == 2.0
