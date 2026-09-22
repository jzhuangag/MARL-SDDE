from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pytest

from .run_dual_use_sensor_development import (
    _load_config,
    _run_payload,
    _run_one,
    _specifications,
)


CONFIG = Path("docs/dual_use_sensor_development_config.json")


def test_development_config_hash_and_scale_are_frozen() -> None:
    config = _load_config(CONFIG)
    specifications = _specifications(config)
    assert len(specifications) == 5760
    assert len(specifications) * config["horizon"] == 5_898_240
    assert config["formal_evidence"] is False


def test_development_specifications_use_only_new_seeds() -> None:
    config = _load_config(CONFIG)
    seeds = {specification["seed"] for specification in _specifications(config)}
    assert seeds == set(range(83001, 83009))


@pytest.mark.parametrize("rotation_fraction", [0.0, 0.25, 1.0])
def test_small_development_path_is_finite_and_budgeted(
    rotation_fraction: float,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["horizon"] = 32
    specification = {
        "seed": 999001,
        "step": 0.5,
        "arrival": 0.1,
        "persistence": 0.8,
        "rotation_fraction": rotation_fraction,
        "budget": 0.25,
        "fingerprint_noise": 0.05,
        "probe_period": 8,
    }
    row = _run_one(specification, config)
    assert row["sensor_calls"] <= row["allowance"]
    assert row["exact_phase_calls"] <= row["allowance"]
    assert row["informative_fingerprints"] <= row["sensor_calls"]
    if rotation_fraction == 0.0:
        assert row["sensor_calls"] == 0
        assert row["sensor_log_energy_rate"] == pytest.approx(
            row["never_log_energy_rate"]
        )


def test_development_run_is_deterministic_for_one_path() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["horizon"] = 64
    specification = _specifications(config)[123]
    assert _run_one(specification, config) == _run_one(specification, config)


def test_ordered_worker_execution_matches_serial_rows() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["horizon"] = 16
    specifications = _specifications(config)[:4]
    payloads = [(specification, config) for specification in specifications]
    serial = [_run_payload(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=2) as executor:
        parallel = list(executor.map(_run_payload, payloads, chunksize=1))
    assert parallel == serial
