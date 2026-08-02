import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analyze_exp018a_direct_gradient import PROJECTION_COLUMNS
from exp018a_direct_gradient_config import (
    PILOT_SEEDS,
    Q_LEVELS,
    RHO_LEVELS,
    STATIC_MANIFEST_HASH,
    build_static_manifest,
    expected_rows,
    expected_source_gradient_evaluations,
    variance_factor,
)
from run_exp018a_direct_gradient import (
    TransitionBank,
    ValueNetwork,
    pairwise_share,
    parameter_hash,
    projection_matrix,
    source_assignment,
    source_gradient_projections,
    static_validate,
)


def synthetic_bank(dimension: int, sources: int = 3, length: int = 8) -> TransitionBank:
    rng = np.random.RandomState(1801 + dimension)
    states = rng.normal(size=(sources, length, dimension)).astype(np.float32)
    following = rng.normal(size=(sources, length, dimension)).astype(np.float32)
    rewards = rng.normal(size=(sources, length)).astype(np.float32)
    terminated = np.zeros((sources, length), dtype=np.bool_)
    return TransitionBank(states, following, rewards, terminated)


def test_seed_registry_is_unique() -> None:
    assert len(PILOT_SEEDS) == 64
    assert len(set(PILOT_SEEDS)) == len(PILOT_SEEDS)


def test_manifest_hash_is_stable() -> None:
    assert len(STATIC_MANIFEST_HASH) == 64
    assert build_static_manifest()["formal_seeds"] is None


def test_expected_workload() -> None:
    assert expected_rows() == 6144
    assert expected_source_gradient_evaluations() == 16896


@pytest.mark.parametrize("rho", RHO_LEVELS)
def test_variance_factor_q1(rho: float) -> None:
    assert variance_factor(1, rho) == pytest.approx(1.0)


def test_variance_factor_saturation() -> None:
    assert variance_factor(32, 0.9) == pytest.approx(0.903125)
    assert variance_factor(32, 0.0) == pytest.approx(1.0 / 32.0)


def test_q_paths_are_theoretically_nonincreasing() -> None:
    for rho in RHO_LEVELS:
        path = [variance_factor(q, rho) for q in Q_LEVELS]
        assert all(left >= right for left, right in zip(path, path[1:]))


def test_source_assignment_is_nested_across_q() -> None:
    assignment = source_assignment(PILOT_SEEDS[0], "cartpole", "fast_regeneration", 0.5)
    assert len(assignment) == 32
    assert np.array_equal(assignment[:4], assignment[:16][:4])


def test_pairwise_share_exact_extremes() -> None:
    assert pairwise_share(0, 32)[1] == 0.0
    assert pairwise_share(32, 32)[1] == 1.0
    assert pairwise_share(1, 1) == (0, 0.0)


def test_projection_matrix_is_deterministic_and_normalized() -> None:
    left = projection_matrix(101, 7)
    right = projection_matrix(101, 7)
    assert torch.equal(left, right)
    assert torch.sum(left[0] ** 2).item() == pytest.approx(1.0)


def test_gradient_projection_does_not_update_parameters() -> None:
    bank = synthetic_bank(4)
    projections, before, after = source_gradient_projections(bank, "cartpole", 1234)
    assert projections.shape == (3, len(PROJECTION_COLUMNS))
    assert np.isfinite(projections).all()
    assert before == after


def test_parameter_hash_changes_when_parameter_changes() -> None:
    model = ValueNetwork(4)
    before = parameter_hash(model)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert parameter_hash(model) != before


def test_independent_regeneration_clock_is_source_local() -> None:
    from run_exp018a_direct_gradient import generate_independent_transition_bank

    source = inspect.getsource(generate_independent_transition_bank)
    assert "for source in range(source_count)" in source
    assert "global_regeneration" not in source


def test_static_validate_is_outcome_free_cpu() -> None:
    result = static_validate()
    assert result["scientific_trajectories_generated"] == 0
    assert not result["gpu_required"]
    assert result["expected_rows"] == 6144


def test_manifest_scientific_boundaries() -> None:
    boundaries = build_static_manifest()["boundaries"]
    assert any("no online participation controller" in value for value in boundaries)
    assert any("no nonlinear convergence" in value for value in boundaries)


def test_frozen_document_json_is_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    registry = json.loads((root / "docs" / "exp018a_seed_registry.json").read_text())
    gates = json.loads((root / "docs" / "exp018a_gate_table.json").read_text())
    manifest = json.loads((root / "docs" / "exp018a_static_manifest.json").read_text())
    assert tuple(registry["pilot_seeds"]) == PILOT_SEEDS
    assert gates["formal_seeds"] is None
    assert manifest["static_manifest_hash"] == STATIC_MANIFEST_HASH


def test_no_result_directory_exists_at_preregistration() -> None:
    root = Path(__file__).resolve().parents[2]
    results = root / "experiments" / "nonlinear_markov_td" / "results"
    matches = list(results.glob("exp018a_pilot_*")) if results.exists() else []
    assert matches == []
