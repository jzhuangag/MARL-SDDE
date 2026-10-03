import json
from pathlib import Path

import pytest

from experiments.policy_dependency_sync.hvp_cost_audit import (
    _build_state,
    configurations,
    measured_call,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "docs" / "policy_dependency_hvp_cost_manifest_20260906.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_frozen_manifest_has_expected_configuration_count():
    manifest = _manifest()
    validate_manifest(manifest)
    assert len(configurations(manifest)) == 72


def test_one_configuration_has_nonzero_owner_and_cross_derivatives():
    manifest = _manifest()
    config = configurations(manifest)[0]
    state = _build_state(config, manifest)
    baseline_time, baseline_norm = measured_call(state, False)
    hvp_time, hvp_norm = measured_call(state, True)
    assert baseline_time > 0.0
    assert hvp_time > 0.0
    assert baseline_norm > 0.0
    assert hvp_norm > 0.0


def test_manifest_rejects_gpu_mutation():
    manifest = _manifest()
    manifest["runtime"]["device"] = "cuda"
    with pytest.raises(ValueError):
        validate_manifest(manifest)
