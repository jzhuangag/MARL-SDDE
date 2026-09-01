from __future__ import annotations

import json
from pathlib import Path

from .run_freshness_headroom_scan import _validate_config, run


def _small_config(path: Path) -> Path:
    payload = {
        "experiment_id": "unit",
        "horizon": 32,
        "seeds": [1, 2],
        "high_prevalences": [0.25],
        "persistence_values": [0.0, 0.8],
        "high_risk_multipliers": [1.0, 8.0],
        "fresh_variance_ratios": [1.0],
        "refresh_fractions": [0.25],
        "gates": {
            "stationary_max_improvement": 1e-10,
            "dynamic_max_geometric_ratio": 1.0,
            "cell_improvement_threshold": 0.0,
            "minimum_directional_fraction": 0.0,
            "high_contrast_max_geometric_ratio": 1.0,
            "persistence_max_geometric_ratio": 1.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_small_scan_is_byte_reproducible(tmp_path: Path) -> None:
    config = _small_config(tmp_path / "config.json")
    first = tmp_path / "first"
    second = tmp_path / "second"
    left = run(config, first)
    right = run(config, second)
    assert left == right
    assert (first / "rows.csv").read_bytes() == (second / "rows.csv").read_bytes()
    assert (first / "summary.json").read_bytes() == (second / "summary.json").read_bytes()


def test_frozen_config_schema_validates() -> None:
    path = Path(__file__).with_name("freshness_headroom_scan_config.json")
    _validate_config(json.loads(path.read_text(encoding="utf-8")))
