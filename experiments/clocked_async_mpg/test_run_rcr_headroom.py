from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .run_rcr_headroom import (
    EXPECTED_CONFIG_SHA256,
    _best_static_vector,
    _generate_certificates,
    _load_config,
    _risk_rows,
    _schedule_risks,
    _specifications,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "docs" / "rcr_headroom_config.json"


def test_frozen_config_hash_and_scale() -> None:
    config = _load_config(CONFIG_PATH)
    assert EXPECTED_CONFIG_SHA256 == "875e692868e6696e5c4dd13c029a3e5e88914bb61e164a422243c9ef4c7e9d36"
    assert len(_specifications(config)) == 2048
    assert len(_specifications(config)) * config["horizon"] == 393216


def test_stationary_certificate_is_time_invariant() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    specification = next(
        spec for spec in _specifications(config) if spec.profile == "stationary"
    )
    d, v = _generate_certificates(specification, config)
    np.testing.assert_allclose(d, np.broadcast_to(d[0], d.shape))
    np.testing.assert_allclose(v, np.broadcast_to(v[0], v.shape))


def test_static_box_qp_dominates_uncorrected_and_full_when_available() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    specification = next(
        spec for spec in _specifications(config) if spec.profile == "mixed"
    )
    d, v = _generate_certificates(specification, config)
    second = config["integrand_second_moment"]
    batch = specification.effective_batch_size
    alpha = _best_static_vector(d, v, second, batch)
    selected = np.mean(_risk_rows(alpha, d, v, second, batch))
    no = np.mean(_risk_rows(np.zeros(specification.agents), d, v, second, batch))
    full = np.mean(_risk_rows(np.ones(specification.agents), d, v, second, batch))
    assert selected <= min(no, full) + 1e-10


def test_static_box_qp_is_robust_across_frozen_profiles() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for specification in _specifications(config)[:64]:
        d, v = _generate_certificates(specification, config)
        alpha = _best_static_vector(
            d,
            v,
            config["integrand_second_moment"],
            specification.effective_batch_size,
        )
        assert np.all((alpha >= 0.0) & (alpha <= 1.0))


def test_schedulers_respect_exact_refresh_allowance() -> None:
    risks = np.asarray([0.3, 0.1, 0.8, 0.2] * 8)
    schedules = _schedule_risks(
        risks, refresh_risk=0.05, budget=0.25, period=4
    )
    assert schedules["causal_refreshes"] <= schedules["allowance"] == 8
    assert schedules["oracle_mean"] <= schedules["periodic_mean"] + 1e-15
