from __future__ import annotations

import json

import pytest

from .run_forecast_reversal_development import geometric_mean, run_development


def test_geometric_mean_rejects_nonpositive_values() -> None:
    assert geometric_mean([1.0, 4.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        geometric_mean([0.0, 1.0])


def test_tiny_development_run_writes_auditable_outputs(tmp_path) -> None:
    output = tmp_path / "tiny"
    summary = run_development(
        output_dir=output,
        seeds=[7],
        workers=1,
        grid=[
            {
                "cycle_probability": 0.95,
                "maximum_delay": 1,
                "communication_budget": 0.5,
                "total_events": 32,
            }
        ],
        policies=(
            "plugin_joint",
            "exact_joint",
            "state_myopic",
            "fixed_edge_0",
            "fixed_edge_1",
            "fixed_edge_2",
            "fixed_initial",
            "round_robin",
            "random_edge",
            "no_refresh",
        ),
    )
    assert summary["row_count"] == 10
    assert summary["scenario_count"] == 1
    assert summary["all_finite"] is True
    assert (output / "endpoints.csv").is_file()
    loaded = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert loaded["endpoints_sha256"] == summary["endpoints_sha256"]

