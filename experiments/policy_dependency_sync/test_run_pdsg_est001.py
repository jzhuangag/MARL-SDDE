from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.policy_dependency_sync.run_pdsg_est001 import (
    build_scenario,
    run_audit,
    stationary_ar_noise,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "pdsg_est001_manifest_v2_20260906.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_scenario_is_deterministic_and_respects_registered_norms() -> None:
    manifest = _manifest()
    left = build_scenario(manifest, 8, 0.16, 0.25, 0.01)
    right = build_scenario(manifest, 8, 0.16, 0.25, 0.01)
    np.testing.assert_allclose(left.current_gradient, right.current_gradient)
    np.testing.assert_allclose(left.base_gradient, right.base_gradient)
    assert len(left.true_candidates) == 8
    assert np.linalg.norm(left.current_gradient) <= 1.0 + 1e-12
    assert max(map(np.linalg.norm, left.true_candidates)) <= 1.02 + 1e-12


def test_stationary_ar_generator_has_requested_shape() -> None:
    values = stationary_ar_noise(
        np.random.default_rng(3), 17, 0.8, (3, 2), 0.7
    )
    assert values.shape == (17, 3, 2)
    assert np.isfinite(values).all()


def test_small_audit_is_finite_complete_and_fully_charged() -> None:
    manifest = _manifest()
    rows = [(128, 0.5, 2, 0.08, 0.0, 0.0)]
    action_rows, seed_rows, summary = run_audit(
        manifest, seeds=[93001, 93002], parameter_rows=rows
    )
    assert len(seed_rows) == 2
    assert len(action_rows) == 6
    assert summary["scenario_count"] == 1
    assert all(row["charged_samples"] == 256 for row in seed_rows)
    assert all(
        row["control_rollout_id"] != row["update_rollout_id"]
        for row in seed_rows
    )
    assert all(np.isfinite(row["error_to_bound"]) for row in action_rows)
