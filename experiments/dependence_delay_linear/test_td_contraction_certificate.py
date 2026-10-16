"""Deterministic tests; no new scientific seeds or sampled efficacy trials."""

from dataclasses import replace
import itertools

import numpy as np
import pytest

from observable_transfer_reference import ProbeSpec, exact_cost_contrast, exact_probe_mean
from reusable_transfer_cache import PrefixCache, cached_controller
from td_contraction_certificate import (
    ContractionBatch, ContractionSpec, DirectionalContractionBatch,
    augment_directional_tail, augment_with_tail, bounds_from_means,
    certified_transfer, collection_cost, exact_norm_profile, integrated_tail,
    directional_transfer, exact_terminal_profile, observe_norm_block, terminal_resolvent,
)


def spec(block=4, dimension=2, replicas=8):
    return ContractionSpec(dimension, .6, .5, block, replicas, .005, "law-v1")


@pytest.mark.parametrize("gamma,step", [(0., 1.), (.6, .5), (.9, .2)])
def test_one_vector_matches_full_product_for_every_four_step_path(gamma, step):
    contract = replace(spec(), discount=gamma, step=step)
    for start in (0, 1):
        for tail in itertools.product((0, 1), repeat=4):
            matrix, current, records = np.eye(2), start, []
            for nxt in tail:
                B = np.eye(2)
                B[current, current] -= step
                B[current, nxt] += step*gamma
                matrix = B@matrix
                records.append((current, 0., nxt))
                current = nxt
            sample = observe_norm_block(contract, start, records)
            np.testing.assert_allclose(sample["norm"], np.linalg.norm(matrix, ord=np.inf), atol=1e-14)
            assert sample["transitions"] == 4 and sample["reset_requests"] == 1


def test_fewer_than_d_updates_cannot_contract_full_infinity_norm():
    sample = observe_norm_block(spec(block=2, dimension=3), 0, [(0, 0., 1), (1, 0., 2)])
    assert sample["norm"] == 1
    bound = bounds_from_means(spec(block=2, dimension=3), np.ones(3), np.ones(3))
    assert bound["structural_unvisited_state_obstruction"]
    assert not bound["strict_contraction"]


def test_unvisited_state_blocks_contraction_even_in_a_long_trajectory():
    sample = observe_norm_block(spec(block=10), 0, [(0, 0., 0)]*10)
    assert sample["norm"] == 1


@pytest.mark.parametrize("kappa", [0., .2, .9, 1-1e-12, 1.])
def test_constant_time_tail_matches_direct_sum(kappa):
    for m in (1, 3, 8):
        for h in (1, 8, 17, 128):
            direct = sum(kappa**(t//m) for t in range(m, h))
            np.testing.assert_allclose(integrated_tail(h, m, kappa), direct, atol=1e-10, rtol=1e-12)


def test_first_and_second_moments_are_simultaneously_covered_and_ordered():
    contract = spec(replicas=100)
    out = bounds_from_means(contract, [.4, .5], [.2, .3])
    radius = np.sqrt(np.log(4/.005)/200)
    np.testing.assert_allclose(out["kappa1"], .5+radius)
    np.testing.assert_allclose(out["kappa2"], .3+radius)
    assert out["kappa2"] <= out["kappa1"]
    with pytest.raises(ValueError):
        bounds_from_means(contract, [.5, .5], [.1, .1])


def test_batch_rejects_duplicates_unregistered_law_and_incomplete_states():
    batch = ContractionBatch(spec(block=2, replicas=2))
    records = [(0, 0., 1), (1, 0., 0)]
    batch.add(0, 0, records, "law-v1")
    for index, law in [(0, "law-v1"), (1, "law-v2")]:
        with pytest.raises(ValueError):
            batch.add(0, index, records, law)
    with pytest.raises(ValueError):
        batch.finalize()
    batch.add(0, 1, records, "law-v1")
    for i in range(2):
        batch.add(1, i, [(1, 0., 0), (0, 0., 1)], "law-v1")
    result = batch.finalize()
    assert batch.transitions == 8 and batch.resets == 4
    assert not batch.sums.flags.writeable
    result["kappa1"] = 0  # Returned data cannot alter stored certificate.
    assert batch.bounds["kappa1"] > 0
    with pytest.raises(ValueError):
        batch.add(0, 2, records, "law-v1")


@pytest.mark.parametrize("records", [[], [(1, 0., 0)]*4, [(0, float('nan'), 0)]*4, [(0, 0., 2)]*4])
def test_invalid_observation_blocks_are_rejected(records):
    with pytest.raises(ValueError):
        observe_norm_block(spec(), 0, records)


def test_markov_block_iteration_does_not_require_independent_successive_blocks():
    # Correlated chain eigenvalue .5. Exact full products, not product of means.
    p = np.array([[.8, .2], [.3, .7]])
    profile = exact_norm_profile(p, .6, .5, 8)
    k1, k2 = profile[:, 1, 0].max(), profile[:, 1, 1].max()
    for t in range(1, 9):
        assert profile[:, t-1, 0].max() <= k1**(t//2)+1e-12
        assert profile[:, t-1, 1].max() <= k2**(t//2)+1e-12


def test_tail_bounds_exact_future_transfer_risk_on_a_markov_mrp():
    p, rewards = np.array([[.8, .2], [.3, .7]]), np.array([.1, -.2])
    anchor = ProbeSpec(np.array([.3, -.1]), np.array([[-.3], [.1]]), 0, .6, .5, 2, 2, .2)
    G, g, _ = exact_probe_mean(anchor, p, rewards)
    profile = exact_norm_profile(p, .6, .5, 2)
    exact_bounds = {"kappa1": profile[:, -1, 0].max(), "kappa2": profile[:, -1, 1].max()}
    problem = augment_with_tail({"matrix": G, "linear": g}, anchor.directions,
                                anchor.value_bound, 5, 2, exact_bounds)
    target = (np.eye(2)+.6*p)@rewards  # Finite-return target, bounded by B.
    for beta in ([0.], [.4], [1.]):
        true = exact_cost_contrast(replace(anchor, horizon=5), p, rewards, beta, target)
        b = np.array(beta)
        upper = b@problem["matrix"]@b+2*problem["linear"]@b
        assert true <= upper+1e-12


def test_contraction_is_not_sufficient_for_nontriviality():
    assert integrated_tail(1000, 8, .75) >= 8
    assert integrated_tail(1000, 8, .25) < 8
    # The sharp generic lower bound head >= -4 B m d(beta) is enough.
    B, m, delta = 2., 8, np.array([.4, .7])
    head = {"matrix": np.zeros((2, 2)), "linear": -2*B*m*delta}
    problem = augment_with_tail(head, delta[None, :], B, 1000, m,
                                 {"kappa1": .75, "kappa2": .6})
    assert problem["structural_fallback_on_coverage"]
    out = cached_controller(problem)
    np.testing.assert_array_equal(out["beta"], np.zeros(2))


def test_cost_includes_both_batches_and_strict_saving_boundary():
    for m in (8, 16, 24):
        cost = collection_cost(2, m, 64, 32, 128, 128)
        assert cost["strictly_cheaper_same_n"] == (3*m < 64)
        assert cost["reset_requests"] == 768
    cost = collection_cost(2, 8, 64, 32, 128, 64)
    assert cost["equal_n_criterion_3m_less_than_H"] is None
    assert cost["combined_expected_transitions"] == 10112


def test_composed_controller_checks_law_and_uses_complete_fixed_batches():
    anchor = ProbeSpec(np.array([.5]), np.array([[-.5]]), 0, .2, .5, 1, 1, 1.)
    cache = PrefixCache(anchor, (0,), 2, .005, "law-v1")
    contraction = ContractionBatch(ContractionSpec(1, .2, .5, 1, 2, .005, "law-v1"))
    for i in range(2):
        cache.add(0, i, [], 0, [(0, 0., 0)], "law-v1")
        contraction.add(0, i, [(0, 0., 0)], "law-v1")
    cache.finalize()
    contraction.finalize()
    out = certified_transfer(cache, contraction, 0, 64, anchor.value, anchor.directions,
                              [0.], [[1.]], law_tag="law-v1")
    assert out["structural_fallback"] and out["beta"][0] == 0
    assert out["joint_failure_probability"] == .01
    with pytest.raises(ValueError):
        certified_transfer(cache, contraction, 0, 64, anchor.value, anchor.directions,
                             [0.], [[1.]], law_tag="law-v2")


def test_terminal_resolvent_starts_at_zero_not_at_one_more_block():
    for h in (1, 3, 8, 11):
        for k in (0., .4, 1.):
            direct = sum(k**(r//3) for r in range(max(0, h-3)))
            np.testing.assert_allclose(terminal_resolvent(h, 3, k), direct)


def test_shared_directional_block_charges_transitions_only_once():
    E = np.array([[1., .2], [-1., .3]])
    batch = DirectionalContractionBatch(spec(block=2, replicas=2), E)
    for s in (0, 1):
        for i in range(2):
            batch.add(s, i, [(s, 0., 1-s), (1-s, 0., s)], "law-v1")
    batch.finalize()
    assert batch.global_batch.transitions == 8 and batch.global_batch.resets == 4
    assert batch.global_batch.spec.delta == batch.spec.delta/2
    assert not batch.basis.flags.writeable
    saved = batch.terminal
    saved["first"][:] = 0
    assert batch.terminal["first"].max() > 0
    assert np.all(batch.terminal["second"] <= np.max(np.abs(E), axis=0)*batch.terminal["first"]+1e-12)


def test_directional_tail_controls_signed_mappings_and_off_basis_residual():
    p, rewards = np.array([[.8, .2], [.3, .7]]), np.array([.1, -.2])
    anchor = ProbeSpec(np.array([.3, -.1]), np.array([[-.3], [.1]]), 0, .6, .5, 2, 2, .2)
    G, g, _ = exact_probe_mean(anchor, p, rewards)
    profile = exact_norm_profile(p, .6, .5, 2)
    terminal = exact_terminal_profile(p, .6, .5, 2, anchor.directions)
    bounds = {"kappa1": profile[:, -1, 0].max(), "kappa2": profile[:, -1, 1].max()}
    from reusable_transfer_cache import query_upper
    A = np.array([[.5, -.2]])
    D = anchor.directions@A+np.array([[.01, -.02], [.02, .03]])
    head = query_upper(anchor, G, g, {"gram_radius": 0., "linear_radius": 0., "return_bias_radius": 0.},
                        2, anchor.value, D, [0.], A)
    upper = augment_directional_tail(head, 5, 2, anchor.value_bound, A,
                                      head["direction_residuals"], terminal[0, -1, :, 0],
                                      terminal[0, -1, :, 1], bounds)
    target = (np.eye(2)+.6*p)@rewards
    for beta in ([0., 0.], [.3, .2], [1., 0.]):
        true = exact_cost_contrast(replace(anchor, directions=D, horizon=5), p, rewards, beta, target)
        b = np.array(beta)
        assert true <= b@upper["matrix"]@b+2*upper["linear"]@b+1e-12
    assert np.linalg.eigvalsh(upper["matrix"]).min() >= -1e-12


def test_directional_online_composition_does_not_need_true_kernel_or_value():
    anchor = ProbeSpec(np.array([.5]), np.array([[-.5]]), 0, .2, .5, 1, 1, 1.)
    cache = PrefixCache(anchor, (0,), 2, .005, "law-v1")
    contraction = DirectionalContractionBatch(
        ContractionSpec(1, .2, .5, 1, 2, .005, "law-v1"), anchor.directions)
    for i in range(2):
        cache.add(0, i, [], 0, [(0, 0., 0)], "law-v1")
        contraction.add(0, i, [(0, 0., 0)], "law-v1")
    cache.finalize()
    contraction.finalize()
    out = directional_transfer(cache, contraction, 0, 64, anchor.value,
                                anchor.directions, [0.], [[1.]], law_tag="law-v1")
    assert out["upper_advantage"] <= 0 and out["joint_failure_probability"] == .01
    incompatible = replace(anchor, directions=np.array([[-.4]]))
    other = PrefixCache(incompatible, (0,), 2, .005, "law-v1")
    for i in range(2):
        other.add(0, i, [], 0, [(0, 0., 0)], "law-v1")
    other.finalize()
    with pytest.raises(ValueError):
        directional_transfer(other, contraction, 0, 64, anchor.value, anchor.directions,
                               [0.], [[1.]], law_tag="law-v1")
