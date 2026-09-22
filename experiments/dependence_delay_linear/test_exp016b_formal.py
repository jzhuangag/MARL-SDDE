"""Static integrity tests for the EXP-016B formal freeze."""

from __future__ import annotations

from run_exp016b_formal import formal_seeds, load_registry
from run_exp016b_pilot import load_frozen_bundle


def test_formal_registry_and_implementation_hashes() -> None:
    registry = load_registry()
    assert registry["outcomes_present_at_freeze"] is False
    assert len(formal_seeds(registry)) == 192


def test_formal_seeds_are_unique_and_disjoint_from_pilot() -> None:
    registry = load_registry()
    seeds = formal_seeds(registry)
    pilot = set(load_frozen_bundle()["seeds"]["pilot_seeds"])
    assert len(seeds) == len(set(seeds))
    assert set(seeds).isdisjoint(pilot)
    assert set(seeds).isdisjoint({90340101, 90340102})


def test_formal_workload_is_frozen() -> None:
    registry = load_registry()
    rows_per_seed = 1_376_256 // 96
    assert rows_per_seed * len(formal_seeds(registry)) == 2_752_512
