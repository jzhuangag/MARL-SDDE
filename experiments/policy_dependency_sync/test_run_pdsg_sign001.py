from __future__ import annotations

import json
from pathlib import Path

from .run_pdsg_sign001 import (
    EXPECTED_MANIFEST_SHA256,
    PROPOSED_POLICY,
    load_manifest,
    sha256,
)


MANIFEST = Path("docs/pdsg_sign001_manifest_20260906.json")


def test_manifest_hash_and_identifier_are_frozen() -> None:
    assert sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    manifest = load_manifest(MANIFEST)
    assert manifest["experiment_id"] == "PDSG-SIGN-001"


def test_manifest_grid_and_seed_counts_are_exact() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["grid"]["cell_count"] == 24
    assert manifest["grid"]["active_cell_count"] == 16
    assert manifest["grid"]["uncoupled_control_count"] == 8
    assert manifest["seeds"]["count"] == 16
    assert manifest["model"]["launches"] == 160


def test_proposed_and_all_strong_baselines_are_declared() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["policies"][0] == PROPOSED_POLICY
    assert len(manifest["policies"]) == 25
    assert len(set(manifest["policies"])) == 25
    assert sum(name.startswith("fixed_offsets") for name in manifest["policies"]) == 10


def test_frozen_gates_are_complete() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["gates"]) == {f"S{index}" for index in range(1, 13)}
