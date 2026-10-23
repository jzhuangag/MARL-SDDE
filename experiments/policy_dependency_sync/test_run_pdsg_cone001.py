import inspect
import json
from pathlib import Path

import numpy as np

from .pistonball_tail import trajectory_tube_residual
from .run_pdsg_cone001 import (
    EXPECTED_MANIFEST_SHA256,
    edge_set,
    evaluate_gates,
    jaccard,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "pdsg_cone001_manifest_20260906.json"


def test_frozen_manifest_hash_and_workload():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    cells = 3 * 3 * 3
    trajectories = cells * (128 + 256)
    assert cells == 27
    assert trajectories == 10368
    assert manifest["gpu_authorized"] is False


def test_reward_is_not_read_by_trajectory_residual():
    source = inspect.getsource(trajectory_tube_residual)
    assert ".rewards" not in source
    assert "reward" not in source.lower()


def test_edge_set_and_jaccard():
    left = edge_set({0: (1, 2), 1: (0,), 2: ()})
    right = edge_set({0: (1,), 1: (0,), 2: ()})
    assert left == frozenset({(0, 1), (0, 2), (1, 0)})
    assert np.isclose(jaccard(left, right), 2 / 3)


def test_gate_evaluator_rejects_incomplete_result():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    gates = evaluate_gates([], [], manifest)
    assert gates["C1"] is False
    assert all(name in gates for name in (f"C{i}" for i in range(1, 12)))
