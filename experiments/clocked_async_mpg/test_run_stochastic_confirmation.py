from __future__ import annotations

import copy
from pathlib import Path

import pytest

from .run_stochastic_confirmation import (
    DEFAULT_CONFIG,
    analyze,
    check_reproduction,
    load_config,
    run,
    write_results,
)


def _tiny_config() -> dict:
    config = copy.deepcopy(load_config(DEFAULT_CONFIG))
    config["couplings"] = [0.08]
    config["service_ratios"] = [2.0]
    config["primary_service_ratios"] = [2.0]
    config["seed_count"] = 2
    config["seed_namespace"] = "stochastic-confirmation-unit-test"
    config["maximum_time"] = 25.0
    return config


def test_frozen_configuration_is_valid_and_disjoint() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config["seed_count"] == 64
    assert config["seed_namespace"] not in config["development_namespaces_excluded"]


def test_tiny_confirmation_has_complete_unique_rows() -> None:
    config = _tiny_config()
    rows, summary = run(config, workers=1)
    assert len(rows) == 10
    assert summary["row_count"] == 10
    assert summary["gates"]["S1_schema_unique_finite"]
    assert summary["gates"]["S8_transition_accounting_valid"]
    assert summary["gates"]["S9_registered_delay_valid"]


def test_duplicate_endpoint_is_rejected() -> None:
    config = _tiny_config()
    rows, _ = run(config, workers=1)
    with pytest.raises(ValueError, match="duplicate"):
        analyze(rows+[rows[0]], config)


def test_written_results_are_byte_reproducible(tmp_path: Path) -> None:
    config = _tiny_config()
    rows, summary = run(config, workers=1)
    first = tmp_path/"first"
    second = tmp_path/"second"
    write_results(first, rows, summary, DEFAULT_CONFIG)
    write_results(second, rows, summary, DEFAULT_CONFIG)
    result = check_reproduction(first, second)
    assert result["S12_byte_exact_reproduction"]
