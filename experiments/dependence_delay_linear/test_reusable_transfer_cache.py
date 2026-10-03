"""Deterministic algebra and data-contract tests; no efficacy seeds."""

from dataclasses import replace
import inspect

import numpy as np
import pytest

from observable_transfer_reference import ProbeSpec, exact_cost_contrast, exact_probe_mean
from reusable_transfer_cache import (
    PrefixCache, add_short_unroll_tail, cache_radii, cached_controller, exact_prefix_means,
    qualification_report, query_upper, reuse_cost,
)
from rl_collaboration_interface_audit import return_moments, value_oracle


def fixture_spec():
    return ProbeSpec(np.array([.2, -.1]), np.array([[.4, -.3], [-.2, .5]]),
                     0, .6, .4, 3, 2, .5)


@pytest.mark.parametrize("state", [0, 1])
def test_all_prefixes_equal_direct_random_time_expectations(state):
    spec, p, r = fixture_spec(), np.array([[.8, .2], [.3, .7]]), np.array([0., .5])
    Gs, gs, mass = exact_prefix_means(spec, p, r, state)
    np.testing.assert_allclose(mass, 1., atol=1e-13)
    for h in range(1, 4):
        G, g, _ = exact_probe_mean(replace(spec, start_state=state, horizon=h), p, r)
        np.testing.assert_allclose(Gs[h-1], G, atol=1e-13)
        np.testing.assert_allclose(gs[h-1], g, atol=1e-13)


@pytest.mark.parametrize("h", [1, 2, 3])
def test_parameters_and_signed_direction_coordinates_can_change(h):
    spec, p, rewards = fixture_spec(), np.array([[.8, .2], [.3, .7]]), np.array([0., .5])
    Gs, gs, _ = exact_prefix_means(spec, p, rewards, 0)
    c, A = np.array([.1, -.2]), np.array([[.7, -.1], [.1, .4]])
    v, D = spec.value+spec.directions@c, spec.directions@A
    zero = {"gram_radius": 0., "linear_radius": 0., "return_bias_radius": 0.}
    problem = query_upper(spec, Gs[h-1], gs[h-1], zero, h, v, D, c, A)
    target, _ = return_moments(p, rewards, spec.discount, spec.return_length)
    for beta in [np.array([.3, .4]), np.array([1., 0.]), np.zeros(2)]:
        truth = exact_cost_contrast(replace(spec, value=v, directions=D, horizon=h), p, rewards, beta, target)
        predicted = beta@problem["matrix"]@beta+2*problem["linear"]@beta
        np.testing.assert_allclose(predicted, truth, atol=2e-13)
    assert problem["value_residual"] < 1e-16
    assert np.max(problem["direction_residuals"]) == 0


@pytest.mark.parametrize("h", [1, 3])
def test_off_anchor_error_and_return_bias_bound_the_real_risk(h):
    spec, p, r = fixture_spec(), np.array([[.8, .2], [.3, .7]]), np.array([0., .5])
    Gs, gs, _ = exact_prefix_means(spec, p, r, 0)
    c, A = np.array([.1, -.2]), np.array([[.7, -.1], [.1, .4]])
    v = spec.value+spec.directions@c+np.array([.03, -.02])
    D = spec.directions@A+np.array([[.02, -.01], [-.04, .03]])
    terms = cache_radii(spec, (0, 1), 128, .01, h)
    # Isolate deterministic label+representation error, not empirical coverage.
    terms.update(gram_radius=0., linear_radius=0.)
    out = query_upper(spec, Gs[h-1], gs[h-1], terms, h, v, D, c, A)
    assert np.linalg.eigvalsh(out["matrix"]).min() >= -1e-12
    assert out["value_residual"] > 0 and out["residual_quadratic"] > 0
    target = value_oracle(p, r, spec.discount)
    for beta in [np.array([.2, .3]), np.array([1., 0.]), np.array([0., 1.]), np.zeros(2)]:
        truth = exact_cost_contrast(replace(spec, value=v, directions=D, horizon=h), p, r, beta, target)
        upper = beta@out["matrix"]@beta+2*out["linear"]@beta
        assert truth <= upper+1e-12


def filled_cache():
    spec = fixture_spec()
    cache = PrefixCache(spec, (0,), 2, .01, "law-v1")
    returns = [(0, 0., 0)]*2
    cache.add(0, 0, [], 0, returns, "law-v1")
    cache.add(0, 1, [(0, 0., 0)]*2, 0, returns, "law-v1")
    cache.finalize()
    return cache


def test_prefix_denominator_is_all_replicas_and_full_cost_is_recorded():
    cache = filled_cache()
    E = cache.anchor.directions[0]
    # Only the K=0 sample contributes at h=1, but denominator is n=2.
    np.testing.assert_allclose(cache.bins_G[0][0], 3*np.outer(E, E)/2)
    assert (cache.training, cache.returns, cache.resets) == (2, 4, 4)
    assert not cache.bins_G[0].flags.writeable
    with pytest.raises(ValueError):
        cache.finalize()
    with pytest.raises(ValueError):
        cache.add(0, 2, [], 0, [(0, 0., 0)]*2, "law-v1")


def test_partial_batch_and_duplicate_evidence_fail_closed():
    cache = PrefixCache(fixture_spec(), (0, 1), 2, .01, "law-v1")
    cache.add(0, 0, [], 0, [(0, 0., 0)]*2, "law-v1")
    with pytest.raises(ValueError):
        cache.add(0, 0, [], 0, [(0, 0., 0)]*2, "law-v1")
    with pytest.raises(ValueError):
        cache.finalize()


@pytest.mark.parametrize("change", [
    {"law_tag": "law-v2"}, {"step": .5}, {"discount": .7},
    {"state": 1}, {"horizon": 4}, {"horizon": 0},
])
def test_model_state_and_horizon_changes_are_not_silently_reused(change):
    cache = filled_cache()
    args = dict(state=0, horizon=2, value=cache.anchor.value,
                directions=cache.anchor.directions, coordinates=np.zeros(2), mapping=np.eye(2),
                law_tag="law-v1", discount=.6, step=.4)
    args.update(change)
    with pytest.raises(ValueError):
        cache.query(**args)


def test_repeated_queries_do_not_recount_samples_or_spend_new_delta():
    cache = filled_cache()
    before = (cache.counts.copy(), cache.training, cache.returns, cache.resets)
    for h in (3, 2, 1, 2):
        problem = cache.query(0, h, cache.anchor.value, cache.anchor.directions,
                              np.zeros(2), np.eye(2), law_tag="law-v1", discount=.6, step=.4)
        out = cached_controller(problem)
        assert out["upper_advantage"] <= 0
    assert before == (cache.counts, cache.training, cache.returns, cache.resets)
    assert cache_radii(cache.anchor, (0,), 2, .01, 1)["entries"] == 15


def test_controller_has_no_transition_model_or_value_truth_inputs():
    forbidden = {"P", "p", "true_value", "target", "V_star"}
    assert not forbidden.intersection(inspect.signature(PrefixCache.query).parameters)
    assert not forbidden.intersection(inspect.signature(query_upper).parameters)


def test_arithmetic_amortization_is_not_equal_accuracy_or_budget_safety():
    spec = replace(fixture_spec(), horizon=64, return_length=32)
    cost = reuse_cost(spec, (0, 1), 128, range(64, 0, -1))
    assert cost["cached_expected_transitions"] == 16256
    assert cost["direct_expected_transitions_same_n"] == 391168
    assert cost["prefix_coefficient_scalars"] == 768
    assert cost["expected_probe_transitions_per_query"] == 254
    short = reuse_cost(replace(spec, horizon=8), (0, 1), 128, [min(h, 8) for h in range(64, 0, -1)])
    assert short["expected_probe_transitions_per_query"] == 142
    assert not cost["same_total_budget_no_harm"]
    assert "not equal confidence" in cost["comparison_scope"]


def test_changed_current_value_uncertainty_has_required_cross_term():
    spec = fixture_spec()
    c, A = np.array([.2, -.1]), np.eye(2)
    v, D = spec.value+spec.directions@c, spec.directions
    G, g = np.eye(2), np.array([-.2, .3])
    terms = {"gram_radius": .2, "linear_radius": .1, "return_bias_radius": .05}
    out = query_upper(spec, G, g, terms, 2, v, D, c, A)
    np.testing.assert_allclose(out["linear"], g+G@c+(.2*.3+.1+.05)*np.ones(2), atol=1e-13)


def test_qualification_is_not_an_efficacy_authorization():
    out = qualification_report()
    assert out["all_state_prefix_affine_identity_max_error"] < 1e-12
    assert out["off_anchor_margin"] >= 0
    assert out["analytic_nonvacuity"]["uses_exact_means_not_collected_data"]
    assert out["analytic_nonvacuity"]["upper_advantage"] < 0
    assert not out["decision"]["new_efficacy_pilot_authorized"]
    assert not out["decision"]["global_low_cost_claim"]


def test_short_unroll_comparison_charges_missing_future_risk():
    spec, p, rewards = fixture_spec(), np.array([[.8, .2], [.3, .7]]), np.array([0., .5])
    Gs, gs, _ = exact_prefix_means(spec, p, rewards, 0)
    problem = {"matrix": Gs[0], "linear": gs[0]}
    extended = add_short_unroll_tail(problem, spec.directions, spec.value_bound, 1, 3)
    target, _ = return_moments(p, rewards, spec.discount, spec.return_length)
    for beta in [np.array([.3, .4]), np.array([1., 0.]), np.zeros(2)]:
        truth = exact_cost_contrast(spec, p, rewards, beta, target)
        upper = beta@extended["matrix"]@beta+2*extended["linear"]@beta
        assert truth <= upper+1e-12
    same = add_short_unroll_tail(problem, spec.directions, spec.value_bound, 1, 1)
    np.testing.assert_array_equal(same["matrix"], problem["matrix"])
    np.testing.assert_array_equal(same["linear"], problem["linear"])
