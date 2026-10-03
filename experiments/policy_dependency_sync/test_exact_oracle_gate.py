import itertools
import json
from pathlib import Path

import numpy as np

from experiments.policy_dependency_sync.exact_oracle_gate import (
    Cell,
    all_fixed_mappings,
    build_cells,
    markov_states,
    potential_matrices,
    simulate,
    validate_manifest,
)


MANIFEST = Path("docs/policy_dependency_sync_oracle_manifest_20260906.json")


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_and_registered_cell_populations_are_valid():
    manifest = load_manifest()
    validate_manifest(manifest)
    cells = build_cells(manifest)
    assert cells
    assert {cell.topology for cell in cells} == set(
        manifest["model"]["topology_families"]
    )


def test_quadratic_potentials_are_symmetric_positive_definite():
    manifest = load_manifest()
    for cell in build_cells(manifest):
        matrices, weights = potential_matrices(cell, manifest)
        assert np.allclose(matrices, matrices.transpose(0, 2, 1))
        assert np.allclose(weights, weights.transpose(0, 2, 1))
        assert np.min(np.linalg.eigvalsh(matrices)) >= 0.599999999


def test_markov_generator_is_deterministic_and_nonconstant():
    first = markov_states(92001, 180, 3, 0.8)
    second = markov_states(92001, 180, 3, 0.8)
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) > 1


def test_fixed_mapping_registry_is_complete():
    mappings = all_fixed_mappings(4)
    assert len(mappings) == 3**4
    for mapping in mappings:
        assert all(mapping[i] != i for i in range(4))


def test_budgeted_policies_respect_pathwise_message_cap():
    manifest = load_manifest()
    cell = Cell("rotating_dense", 0.9, 0.8, 5, 0.5, "imbalanced")
    mapping = all_fixed_mappings(4)[0]
    for policy in (
        "no_refresh",
        "periodic_full",
        "round_robin_all",
        "oldest_cache",
        "largest_mismatch",
        "active_oldest",
        "weighted_mismatch",
        "one_step_oracle",
        "best_fixed_mapping",
    ):
        kwargs = {"fixed_mapping": mapping} if policy == "best_fixed_mapping" else {}
        result = simulate(cell, policy, 92001, manifest, **kwargs)
        assert result["finite"]
        assert result["messages"] <= cell.message_rate * manifest["model"]["horizon"]


def test_uncoupled_dynamics_do_not_depend_on_refresh_policy():
    manifest = load_manifest()
    cell = Cell("uncoupled", 0.0, 0.8, 5, 1.0, "round_robin")
    risks = []
    for policy in (
        "no_refresh",
        "periodic_full",
        "largest_mismatch",
        "weighted_mismatch",
        "one_step_oracle",
        "unbudgeted_full_refresh",
    ):
        risks.append(simulate(cell, policy, 92001, manifest)["risk"])
    assert max(risks) - min(risks) < 1e-12


def test_delayed_refresh_uses_only_past_parameter_versions():
    manifest = load_manifest()
    zero = Cell("rotating_path", 0.9, 0.95, 0, 1.0, "round_robin")
    delayed = Cell("rotating_path", 0.9, 0.95, 5, 1.0, "round_robin")
    values = [
        simulate(cell, "weighted_mismatch", seed, manifest)["risk"]
        for cell, seed in itertools.product((zero, delayed), (92001, 92002))
    ]
    assert all(np.isfinite(values))
    assert values[:2] != values[2:]

