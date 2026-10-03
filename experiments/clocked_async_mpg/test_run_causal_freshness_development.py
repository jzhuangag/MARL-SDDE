from __future__ import annotations

import json
from pathlib import Path

from .run_causal_freshness_development import run


def test_small_causal_development_scan_writes_complete_summary(tmp_path: Path) -> None:
    config = {
        "horizon": 24,
        "seeds": [1],
        "high_prevalences": [0.25],
        "persistence_values": [0.8],
        "high_risk_multipliers": [1.0, 8.0],
        "fresh_variance_ratios": [1.0],
        "refresh_fractions": [0.25],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    summary = run(path, tmp_path / "out")
    assert summary["row_count"] == 10
    assert len(summary["by_tradeoff"]) == 5
    assert summary["gpu_authorized"] is False
