"""Tests for EXP-010A multistate certificate transfer."""

import numpy as np

from multistate_certificate import (
    aggregate_td_noise,
    build_transfer_mrp,
    candidate_actions,
    exact_pair_tv,
    select_action,
)


def test_transfer_mrp_and_pair_chain_are_stationary() -> None:
    model = build_transfer_mrp(0.9)
    transition = model["transition"]
    pair_transition = model["pair_transition"]
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert np.allclose(model["stationary"] @ transition, model["stationary"])
    assert np.allclose(pair_transition.sum(axis=1), 1.0)
    assert np.allclose(
        model["pair_weights"] @ pair_transition,
        model["pair_weights"],
    )
    assert np.allclose(model["mean"] @ model["theta_star"], model["b_vector"])


def test_exact_pair_tv_decreases_under_thinning() -> None:
    model = build_transfer_mrp(0.98)
    values = [exact_pair_tv(model, gap) for gap in (1, 10, 100, 500)]
    assert all(left >= right for left, right in zip(values, values[1:]))
    assert values[-1] < values[0]


def test_noise_saturates_under_full_pair_sharing() -> None:
    model = build_transfer_mrp(0.0)
    reference = aggregate_td_noise(model, 1, 1.0)
    for num_agents in (2, 8, 32):
        assert np.isclose(
            aggregate_td_noise(model, num_agents, 1.0), reference
        )


def test_selected_action_is_strictly_certified_and_budget_valid() -> None:
    model = build_transfer_mrp(0.9)
    actions = candidate_actions(model, rho=0.9, maximum_delay=8)
    selected = select_action(actions)
    assert selected["eta"] > 0.0
    assert selected["sharp_factor"] < 1.0
    assert selected["effective_monotonicity"] > 0.0
    assert (
        selected["updates"] * selected["update_cost"]
        <= selected["resource_budget"]
    )
