from __future__ import annotations

import json
from pathlib import Path

from .run_causal_freshness_confirmation import _validate, run


def test_frozen_confirmation_config_validates() -> None:
    path = Path(__file__).with_name("causal_freshness_confirmation_config.json")
    _validate(json.loads(path.read_text(encoding="utf-8")))


def test_small_confirmation_is_reproducible(tmp_path: Path) -> None:
    frozen = json.loads(
        Path(__file__).with_name("causal_freshness_confirmation_config.json").read_text(
            encoding="utf-8"
        )
    )
    frozen.update(
        {
            "horizon": 24,
            "seeds": list(range(92001, 92065)),
            "high_prevalences": [0.25],
            "persistence_values": [0.8],
            "high_risk_multipliers": [1.0, 8.0],
            "fresh_variance_ratios": [1.0],
            "refresh_fractions": [0.25],
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    first = run(path, tmp_path / "first")
    second = run(path, tmp_path / "second")
    assert first == second
    assert (tmp_path / "first" / "rows.csv").read_bytes() == (
        tmp_path / "second" / "rows.csv"
    ).read_bytes()
