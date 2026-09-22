"""Deterministic protocol and algebra tests; never new scientific seeds."""

from dataclasses import replace
import itertools

import numpy as np
import pytest

from delayed_training_risk_ledger import (
    DelayedRiskLedger, LedgerSpec, colored_radius, qualification_report,
    safe_scale, score, worst_excess,
)


def spec(**kwargs):
    defaults = dict(dimension=2, reward_bound=1., discount=.5, return_length=2,
                    delivery_delay=1, horizon=12, delta=.01,
                    initial_allowance=1., allowance_per_step=.2, law_tag="fixed-MRP")
    return LedgerSpec(**(defaults | kwargs))


def test_scalar_reservation_is_exact_worst_value_contrast():
    for c, l in itertools.product(np.linspace(-1., 1., 9), repeat=2):
        contrasts = [(c-v)**2-(l-v)**2 for v in (-1., 1.)]
        assert worst_excess(c, l, 1.) == pytest.approx(max(contrasts))


def test_closed_form_scale_maximizes_feasible_current_state_transfer():
    for c, l, capacity in itertools.product((-.9, -.2, .4, .8), repeat=3):
        capacity = abs(capacity)
        alpha = safe_scale(c, l, 1., capacity)
        assert 0 <= alpha <= 1
        assert worst_excess(l+alpha*(c-l), l, 1.) <= capacity+1e-12
        if alpha < 1-1e-8:
            assert worst_excess(l+(alpha+1e-8)*(c-l), l, 1.) > capacity


def test_zero_allowance_recreates_cold_start_without_known_safe_direction():
    assert safe_scale(.5, 0., 1., 0.) == 0
    assert safe_scale(.5, .5, 1., 0.) == 1
    assert safe_scale(.5, 0., 1., .1) > 0


def test_near_identical_values_use_bounded_rounding_backoff():
    c, l = np.nextafter(1., 2.), 1.
    capacity = .75*worst_excess(c, l, 2.)
    s = spec(dimension=1, initial_allowance=capacity, allowance_per_step=0.)
    action = DelayedRiskLedger(s).act([c], [l], 0, law_tag=s.law_tag)
    assert action["reservation"] <= capacity
    assert action["upper"] <= action["allowance"]
    assert action["parameters"][0] == l


def test_conditional_return_contrast_and_truncation_bias_on_markov_paths():
    P, r, gamma, L = np.array([[.8, .2], [.3, .7]]), np.array([.2, -.7]), .6, 3
    V = np.linalg.solve(np.eye(2)-gamma*P, r)
    VL = sum(np.linalg.matrix_power(gamma*P, k)@r for k in range(L))
    c, l = .8, -.4
    for start in (0, 1):
        average = 0.
        for rest in itertools.product((0, 1), repeat=L):
            state, probability, Y = start, 1., 0.
            for k, nxt in enumerate(rest):
                Y += gamma**k*r[state]
                probability *= P[state, nxt]
                state = nxt
            average += probability*score(c, l, Y)
        assert average == pytest.approx((c-VL[start])**2-(l-VL[start])**2)
        exact = (c-V[start])**2-(l-V[start])**2
        assert abs(average-exact) <= 2*abs(c-l)*gamma**L/(1-gamma)


def test_raw_returns_arrive_once_at_fixed_lag_with_overlap_fully_charged():
    s = spec(dimension=1, initial_allowance=100., allowance_per_step=0.)
    ledger, matured = DelayedRiskLedger(s), []
    rewards = [.1, .3, -.2, .5, 0.]
    for t, r in enumerate(rewards):
        ledger.act([.5], [0.], 0, law_tag=s.law_tag)
        arrived = ledger.observe(0, r, 0, law_tag=s.law_tag)
        assert len(arrived) == (t >= s.lag-1)
        matured.extend(arrived)
    assert [x["birth"] for x in matured] == [0, 1, 2]
    for item in matured:
        t = item["birth"]
        assert item["return"] == pytest.approx(rewards[t]+.5*rewards[t+1])
    assert ledger.actor_transitions == 5  # Not multiplied by overlapping labels.
    assert ledger.return_reward_reuses == 9
    assert ledger.processed_labels == 3
    assert ledger.final_status()["pending_labels"] == 2


def test_only_executed_shrunk_action_enters_the_risk_ledger():
    s = spec(dimension=1, discount=0., return_length=1, delivery_delay=0,
             initial_allowance=.2, allowance_per_step=0.)
    ledger = DelayedRiskLedger(s)
    action = ledger.act([.5], [0.], 0, law_tag=s.law_tag)
    assert 0 < action["parameters"][0] < .5
    arrived = ledger.observe(0, .7, 0, law_tag=s.law_tag)
    assert arrived[0]["actual"] == action["parameters"][0]
    assert arrived[0]["reservation"] == pytest.approx(.2)
    assert ledger.settled_score_bias == pytest.approx(score(action["parameters"][0], 0., .7))


def test_late_noisy_refresh_cannot_revoke_a_previous_bound():
    s = spec(dimension=1, discount=0., return_length=1, delivery_delay=0,
             initial_allowance=.2, allowance_per_step=0.)
    ledger = DelayedRiskLedger(s)
    before = ledger.act([.5], [0.], 0, law_tag=s.law_tag)["upper"]
    ledger.observe(0, -1., 0, law_tag=s.law_tag)
    naive = ledger.settled_score_bias+colored_radius(ledger.counts, s)
    assert naive > .2  # Simply replacing the old certificate would break admission.
    assert ledger.upper <= before <= .2


def test_matured_credit_is_usable_but_each_label_is_counted_only_once():
    # Generic predictable predictions, NOT a claimed TD-benefit experiment.
    s = spec(dimension=1, discount=0., return_length=1, delivery_delay=0,
             horizon=1024, initial_allowance=1024., allowance_per_step=0.)
    ledger = DelayedRiskLedger(s)
    for _ in range(s.horizon):
        ledger.act([0.], [1.], 0, law_tag=s.law_tag)
        ledger.observe(0, 0., 0, law_tag=s.law_tag)
    assert ledger.upper < 0
    assert ledger.settled_score_bias == -1024.
    assert ledger.processed_labels == ledger.counts.sum() == 1024
    assert ledger.pending_upper == 0 and not ledger.pending


def test_last_pending_action_is_not_forgiven_or_drained_with_free_samples():
    s = spec(horizon=2, return_length=3, delivery_delay=2)
    ledger = DelayedRiskLedger(s)
    for t in range(2):
        ledger.act([.3, 0.], [0., 0.], 0, law_tag=s.law_tag)
        ledger.observe(0, .1, 0, law_tag=s.law_tag)
    out = ledger.final_status()
    assert out["pending_labels"] == 2 and out["pending_upper"] > 0
    assert out["processed_labels"] == 0 and out["actor_transitions"] == 2
    with pytest.raises(ValueError):
        ledger.observe(0, 0., 0, law_tag=s.law_tag)
    with pytest.raises(ValueError):
        ledger.act([0., 0.], [0., 0.], 0, law_tag=s.law_tag)


def test_stream_rejects_duplicate_missing_or_wrong_law_transitions():
    s, ledger = spec(), DelayedRiskLedger(spec())
    with pytest.raises(ValueError):
        ledger.observe(0, 0., 1, law_tag=s.law_tag)
    with pytest.raises(ValueError):
        ledger.act([0., 0.], [0., 0.], 0, law_tag="changed")
    ledger.act([.1, 0.], [0., 0.], 0, law_tag=s.law_tag)
    with pytest.raises(ValueError):
        ledger.act([0., 0.], [0., 0.], 0, law_tag=s.law_tag)
    with pytest.raises(ValueError):
        ledger.observe(0, 2., 1, law_tag=s.law_tag)
    ledger.observe(0, 0., 1, law_tag=s.law_tag)
    with pytest.raises(ValueError):
        ledger.act([0., 0.], [0., 0.], 0, law_tag=s.law_tag)


def test_recursive_actual_and_own_shadow_are_not_same_history_comparators():
    # Deterministic two-state chain; no oracle enters the controller. Both
    # actual and local shadow learn on every raw record after the risk is charged.
    s, eta = spec(initial_allowance=.3, allowance_per_step=.2), .4
    ledger = DelayedRiskLedger(s)
    actual, shadow, cumulative = np.array([.7, -.4]), np.array([.7, -.4]), 0.
    # Zero rewards => V*=0. The externally provided donor is only a test proposal.
    for t in range(s.horizon):
        state, nxt = t % 2, 1-t % 2
        candidate = .6*actual+.4*np.array([-.4, .5])
        chosen = ledger.act(candidate, shadow, state, law_tag=s.law_tag)
        actual = chosen["parameters"].copy()
        cumulative += actual[state]**2-shadow[state]**2
        assert cumulative <= chosen["allowance"]+1e-12
        actual[state] += eta*(s.discount*actual[nxt]-actual[state])
        shadow[state] += eta*(s.discount*shadow[nxt]-shadow[state])
        ledger.observe(state, 0., nxt, law_tag=s.law_tag)
    assert not np.allclose(actual, shadow)
    assert ledger.actor_transitions == s.horizon


def test_small_stochastic_tree_exhaustively_checks_the_stated_risk_not_MSE():
    # Unit-scale exhaustive 2^6 paths, not a seed-based performance benchmark.
    s = spec(dimension=1, discount=0., horizon=6, initial_allowance=.1,
             allowance_per_step=.1, return_length=2, delivery_delay=0, delta=.1)
    failure_probability = 0.
    for rewards in itertools.product((-1., 1.), repeat=s.horizon):
        ledger, actual, shadow, excess = DelayedRiskLedger(s), .2, .2, 0.
        failed = False
        for reward in rewards:
            chosen = ledger.act([.8*actual+.2], [shadow], 0, law_tag=s.law_tag)
            actual = float(chosen["parameters"][0])
            excess += actual*actual-shadow*shadow  # True mean reward = 0.
            failed |= excess > chosen["allowance"]+1e-12
            actual = .5*actual+.5*reward
            shadow = .5*shadow+.5*reward
            ledger.observe(0, reward, 0, law_tag=s.law_tag)
        failure_probability += failed/2**s.horizon
    assert failure_probability <= s.delta


def test_visited_state_guarantee_cannot_be_relabelled_full_state_MSE():
    report = qualification_report()
    assert report["unvisited_state_counterexample"]["full_state_contrasts"] == [-.5, 1.5]
    assert report["unvisited_state_counterexample"]["visited_risk_in_both_laws"] == 0.
    s = spec(initial_allowance=0., allowance_per_step=0.)
    action = DelayedRiskLedger(s).act([0., 1.], [0., 0.], 0, law_tag=s.law_tag)
    assert action["alpha"] == 1. and action["reservation"] == 0.
    np.testing.assert_array_equal(action["parameters"], [0., 1.])
    # Even irreducible, geometrically mixing kernels can hide the state over
    # a finite horizon when no uniform quantitative coverage is assumed.
    epsilon, T = 1e-4, 64
    assert (1-epsilon)**T > .99


def test_cumulative_allowance_does_not_bound_a_reflected_Lyapunov_debt():
    r = qualification_report()["cumulative_vs_reflected_counterexample"]
    assert max(r["prefixes"]) == 0
    assert r["reflected_queue"] == [0., 1.]


@pytest.mark.parametrize("change", [{"return_length": 0}, {"delivery_delay": -1},
                                  {"initial_allowance": -1}, {"discount": 1.}])
def test_invalid_contracts_are_rejected(change):
    with pytest.raises(ValueError):
        spec(**change)
