from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from . import run_passive_secant_development as runner


CONFIG = Path("docs/passive_secant_development_config.json")


def _short_config() -> dict:
    config = json.loads(json.dumps(runner._load_config(CONFIG)))
    config["horizon"] = 64
    return config


def _spec(**overrides: float | int) -> dict:
    values: dict[str, float | int] = {
        "seed": 84001,
        "step": 0.5,
        "arrival": 0.5,
        "persistence": 0.95,
        "rotation_fraction": 0.5,
        "budget": 0.5,
        "gradient_noise": 0.0,
    }
    values.update(overrides)
    return values


def test_frozen_config_hash_and_scope() -> None:
    config = runner._load_config(CONFIG)
    specifications = runner._specifications(config)
    assert runner._sha256(CONFIG) == runner.EXPECTED_CONFIG_SHA256
    assert len(specifications) == 1920
    assert len({tuple(spec.values()) for spec in specifications}) == 1920
    assert len(specifications) * config["horizon"] == 1_966_080


def test_stationary_potential_never_buys_optimism() -> None:
    row = runner._run_one(_spec(rotation_fraction=0.0), _short_config())
    assert row["passive_calls"] == 0
    assert row["passive_log_energy_rate"] == pytest.approx(
        row["never_log_energy_rate"]
    )


def test_passive_sensor_has_no_policy_specific_query_counter() -> None:
    row = runner._run_one(_spec(), _short_config())
    assert "probe_calls" not in row
    assert "sensor_queries" not in row
    assert row["informative_secants"] <= row["eligible_secants"]


def test_every_adaptive_method_respects_the_allowance() -> None:
    row = runner._run_one(_spec(gradient_noise=0.05), _short_config())
    assert row["passive_calls"] <= row["allowance"]
    assert row["exact_phase_calls"] <= row["allowance"]


def test_one_path_is_deterministic() -> None:
    first = runner._run_one(_spec(gradient_noise=0.05), _short_config())
    second = runner._run_one(_spec(gradient_noise=0.05), _short_config())
    assert first == second


def test_current_mandatory_gradient_precedes_current_action() -> None:
    source = inspect.getsource(runner._run_one)
    gradient_index = source.index("current_observed_gradient =")
    fingerprint_index = source.index("fingerprint = passive_secant_fingerprint")
    decision_index = source.index("passive_anchor = bool")
    transition_index = source.index("states = np.einsum")
    assert gradient_index < fingerprint_index < decision_index < transition_index


def test_configuration_is_development_only() -> None:
    assert runner._load_config(CONFIG)["formal_evidence"] is False
