from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from . import run_dual_control_value_development as runner


CONFIG = Path("docs/dual_control_value_development_config.json")


def _short_config() -> dict:
    config = runner._load_config(CONFIG)
    config = json.loads(json.dumps(config))
    config["horizon"] = 64
    return config


def _spec(**overrides: float | int) -> dict:
    values: dict[str, float | int] = {
        "seed": 83001,
        "step": 0.5,
        "arrival": 0.5,
        "persistence": 0.95,
        "rotation_fraction": 0.5,
        "budget": 0.5,
        "fingerprint_noise": 0.0,
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


def test_configuration_is_explicitly_development_only() -> None:
    config = runner._load_config(CONFIG)
    assert config["formal_evidence"] is False


def test_stationary_potential_never_purchases_a_query() -> None:
    row = runner._run_one(
        _spec(rotation_fraction=0.0, fingerprint_noise=0.05), _short_config()
    )
    assert row["dual_calls"] == 0
    assert row["myopic_calls"] == 0
    assert row["dual_log_energy_rate"] == pytest.approx(
        row["never_log_energy_rate"]
    )


def test_mixed_path_respects_every_hard_allowance() -> None:
    row = runner._run_one(_spec(), _short_config())
    for prefix in ("dual", "myopic", "exact_phase"):
        assert 0 <= row[f"{prefix}_calls"] <= row["allowance"]
    assert row["information_induced_calls"] <= row["dual_calls"]
    assert row["dual_informative_fingerprints"] <= row["dual_calls"]


def test_one_path_is_deterministic() -> None:
    first = runner._run_one(_spec(fingerprint_noise=0.05), _short_config())
    second = runner._run_one(_spec(fingerprint_noise=0.05), _short_config())
    assert first == second


def test_current_fingerprint_cannot_enter_its_own_decision() -> None:
    source = inspect.getsource(runner._run_one)
    decision_index = source.index("dual_decision = choose_dual_use_lookahead")
    transition_index = source.index("states = np.einsum")
    observation_index = source.index("beliefs[name], observed = _observe")
    assert decision_index < transition_index < observation_index


def test_invalid_workers_are_rejected_by_contract() -> None:
    # The CLI performs this check before constructing an executor.
    source = inspect.getsource(runner.main)
    assert 'if args.workers <= 0:' in source
    assert 'raise ValueError("workers must be positive")' in source
