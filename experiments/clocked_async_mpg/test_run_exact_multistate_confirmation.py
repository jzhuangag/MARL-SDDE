from __future__ import annotations

from pathlib import Path

from .run_exact_multistate_confirmation import DEFAULT_CONFIG, load_config, run


def test_frozen_exact_confirmation_config_is_static_valid() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config["seed_count"] == 64
    assert len(config["couplings"])*len(config["service_ratios"]) == 16
    assert len(config["primary_service_ratios"])*len(config["couplings"]) == 12


def test_tiny_unregistered_config_executes_without_mutating_default() -> None:
    config = load_config(DEFAULT_CONFIG)
    tiny = dict(config)
    tiny["couplings"] = [0.0, 0.24]
    tiny["service_ratios"] = [1.0, 2.0, 4.0, 8.0]
    tiny["primary_service_ratios"] = [2.0, 4.0, 8.0]
    tiny["seed_count"] = 2
    tiny["seed_namespace"] = "unit-test-unregistered"
    rows, summary = run(tiny)
    assert len(rows) == 32
    assert summary["row_count"] == 32
    assert Path(DEFAULT_CONFIG).is_file()
