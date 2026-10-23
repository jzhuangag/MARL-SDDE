from __future__ import annotations

from .audit_multiwalker_cache_contract import (
    chain_neighbors,
    run_multiwalker_cache_contract,
)


def test_chain_neighbor_universe_is_local() -> None:
    assert chain_neighbors(0, 5) == (1,)
    assert chain_neighbors(2, 5) == (1, 3)
    assert chain_neighbors(4, 5) == (3,)


def test_multiwalker_cache_contract_is_exact_and_reproducible() -> None:
    arguments = dict(seeds=(95300, 95301), walkers=5, cycles=12)
    first = run_multiwalker_cache_contract(**arguments)
    second = run_multiwalker_cache_contract(**arguments)
    assert first == second
    assert first.launches == first.receipts == 24
    assert first.maximum_degree == 2
    assert first.nonnull_refreshes > 0
    assert first.optional_policy_bytes > 0
    assert first.changed_cached_actions > 0
    assert first.action_bound_violations == 0
    assert first.nonfinite_observations == 0
    assert first.exact_cache_copy_failures == 0
    assert first.receipt_order_failures == 0
