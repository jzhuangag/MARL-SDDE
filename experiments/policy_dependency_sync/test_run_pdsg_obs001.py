from __future__ import annotations

import inspect
import json
from pathlib import Path

from .async_packet_game import _signed_h4_choice
from .run_pdsg_obs001 import (
    EXPECTED_MANIFEST_SHA256,
    ORACLE_POLICY,
    PROPOSED_POLICY,
    load_manifest,
)
from .run_pdsg_sign001 import sha256


MANIFEST = Path("docs/pdsg_obs001_manifest_20260906.json")


def test_manifest_hash_and_identifier_are_frozen() -> None:
    assert sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    assert load_manifest(MANIFEST)["experiment_id"] == "PDSG-OBS-001"


def test_population_and_seed_isolation_are_exact() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["grid"]["cell_count"] == 24
    assert manifest["seeds"]["start"] == 61001
    assert manifest["seeds"]["count"] == 16
    assert not set(range(61001, 61017)) & set(manifest["seeds"]["excluded"])
    assert manifest["model"]["launches"] == 160


def test_policy_roles_and_gates_are_exact() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["policies"][0] == PROPOSED_POLICY
    assert len(manifest["policies"]) == 25
    assert ORACLE_POLICY not in manifest["policies"]
    assert manifest["oracle_diagnostic_policy"] == ORACLE_POLICY
    assert set(manifest["gates"]) == {f"O{index}" for index in range(1, 15)}


def test_online_score_interface_excludes_environment_target() -> None:
    parameters = inspect.signature(_signed_h4_choice).parameters
    assert "target" not in parameters
    assert "score_reference" in parameters


def test_no_result_directory_exists_before_preregistration() -> None:
    root = Path("experiments/policy_dependency_sync/results")
    assert not (root / "pdsg_obs001_primary").exists()
    assert not (root / "pdsg_obs001_reproduction").exists()
