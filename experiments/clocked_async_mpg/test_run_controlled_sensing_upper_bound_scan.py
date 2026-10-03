from __future__ import annotations

from pathlib import Path

import pytest

from . import run_controlled_sensing_upper_bound_scan as runner


CONFIG = Path("docs/controlled_sensing_upper_bound_config.json")


def test_frozen_upper_bound_scope_and_hash() -> None:
    config = runner._load_config(CONFIG)
    specifications = runner._specifications(config)
    assert runner._sha256(CONFIG) == runner.EXPECTED_CONFIG_SHA256
    assert len(specifications) == 120
    assert len({tuple(spec.values()) for spec in specifications}) == 120
    assert len(specifications) * len(config["maximum_ages"]) == 360


def test_one_cell_has_the_required_cost_ordering() -> None:
    config = runner._load_config(CONFIG)
    row = runner._run_one(
        {
            "step": 0.5,
            "arrival": 0.5,
            "persistence": 0.8,
            "rotation_fraction": 0.5,
            "budget": 0.25,
        },
        config,
    )
    assert row["exact_phase_log_cost"] <= row["perfect_sensing_log_cost"] + 1e-9
    assert row["perfect_sensing_log_cost"] <= row["fixed_log_cost"] + 1e-9
    assert row["perfect_sensing_call_rate"] <= 0.25 + 1e-9
    assert row["age_convergence_gap"] <= 1e-6


def test_stationary_potential_control_uses_no_calls() -> None:
    config = runner._load_config(CONFIG)
    row = runner._run_one(
        {
            "step": 0.8,
            "arrival": 0.1,
            "persistence": 0.95,
            "rotation_fraction": 0.0,
            "budget": 0.5,
        },
        config,
    )
    assert row["perfect_sensing_call_rate"] == pytest.approx(0.0, abs=1e-10)
    assert row["perfect_sensing_log_cost"] == pytest.approx(
        row["fixed_log_cost"], abs=1e-10
    )


def test_configuration_is_analytic_and_nonformal() -> None:
    config = runner._load_config(CONFIG)
    assert config["formal_evidence"] is False
    assert config["experiment"] == "LCO-U0-ANALYTIC-UPPER-BOUND"
