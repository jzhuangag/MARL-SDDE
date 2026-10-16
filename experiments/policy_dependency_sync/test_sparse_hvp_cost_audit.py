import json
from pathlib import Path

import pytest

from experiments.policy_dependency_sync.sparse_hvp_cost_audit import (
    _build_state,
    configurations,
    measured_call,
    structural_support_check,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "docs" / "sparse_policy_dependency_hvp_manifest_20260906.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_frozen_manifest_has_expected_configuration_count():
    manifest = _manifest()
    validate_manifest(manifest)
    assert len(configurations(manifest)) == 108


def test_local_factor_excludes_non_neighbor_parameters():
    manifest = _manifest()
    state = _build_state(configurations(manifest)[0], manifest)
    local_nonzero, outside_unused = structural_support_check(state)
    assert local_nonzero
    assert outside_unused


def test_measured_paths_are_finite_and_nonzero():
    manifest = _manifest()
    state = _build_state(configurations(manifest)[0], manifest)
    for alignment_hvp in (False, True):
        elapsed, norm = measured_call(state, alignment_hvp)
        assert elapsed > 0.0
        assert norm > 0.0


def test_manifest_rejects_missing_non_neighbor():
    manifest = _manifest()
    manifest["model_grid"]["agents"] = [8]
    with pytest.raises(ValueError):
        validate_manifest(manifest)
