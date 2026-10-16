import json
from pathlib import Path

import numpy as np

from exp018a_direct_gradient_config import PILOT_SEEDS, Q_LEVELS, RHO_LEVELS
from exp018b_direct_gradient_config import (
    FORMAL_SEEDS,
    STATIC_MANIFEST_HASH,
    build_static_manifest,
    expected_rows,
    expected_source_gradient_evaluations,
)
from run_exp018b_direct_gradient_formal import rows_for_seed, static_validate


def test_formal_seed_registry_is_unique_and_disjoint() -> None:
    assert len(FORMAL_SEEDS) == 192
    assert len(set(FORMAL_SEEDS)) == len(FORMAL_SEEDS)
    assert set(FORMAL_SEEDS).isdisjoint(PILOT_SEEDS)


def test_formal_workload() -> None:
    assert expected_rows() == 18_432
    assert expected_source_gradient_evaluations() == 50_688


def test_static_manifest_and_boundaries() -> None:
    manifest = build_static_manifest()
    assert len(STATIC_MANIFEST_HASH) == 64
    assert manifest["q1_crn_rule"].startswith("private source 1")
    assert manifest["inference"]["familywise_alpha"] == 0.05
    assert len(manifest["primary_endpoints"]) == 2


def test_static_validate_is_cpu_and_outcome_free() -> None:
    result = static_validate()
    assert result["scientific_trajectories_generated"] == 0
    assert not result["gpu_required"]
    assert result["q1_crn_exact_by_construction"]


def test_q1_rows_are_exact_across_rho(monkeypatch) -> None:
    from run_exp018b_direct_gradient_formal import PROJECTION_COLUMNS
    from run_exp018a_direct_gradient import TransitionBank
    import run_exp018b_direct_gradient_formal as runner

    def fake_bank(*args, **kwargs):
        return TransitionBank(
            np.zeros((33, 2, 4), dtype=np.float32),
            np.zeros((33, 2, 4), dtype=np.float32),
            np.zeros((33, 2), dtype=np.float32),
            np.zeros((33, 2), dtype=np.bool_),
        )

    def fake_projections(bank, task, checkpoint):
        values = np.arange(33 * len(PROJECTION_COLUMNS), dtype=float).reshape(33, -1)
        return values, "same", "same"

    monkeypatch.setattr(runner, "TASKS", {"cartpole": {}})
    monkeypatch.setattr(runner, "MIXING_PROFILES", {"fast": {}})
    monkeypatch.setattr(runner, "CHECKPOINTS", {"init": 1})
    monkeypatch.setattr(runner, "generate_independent_transition_bank", fake_bank)
    monkeypatch.setattr(runner, "source_gradient_projections", fake_projections)
    rows = rows_for_seed(123)
    q1 = [row for row in rows if row["q"] == 1]
    for column in PROJECTION_COLUMNS:
        assert len({row[column] for row in q1}) == 1


def test_preregistration_json_is_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    registry = json.loads((root / "docs" / "exp018b_seed_registry.json").read_text())
    manifest = json.loads((root / "docs" / "exp018b_static_manifest.json").read_text())
    assert tuple(registry["formal_seeds"]) == FORMAL_SEEDS
    assert manifest["static_manifest_hash"] == STATIC_MANIFEST_HASH
