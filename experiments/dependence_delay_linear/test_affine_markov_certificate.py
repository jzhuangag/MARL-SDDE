"""Tests for the affine decorrelated Markov-TD finite-time certificate."""

import numpy as np

from affine_markov_certificate import (
    affine_bound_components,
    affine_candidate_actions,
    affine_finite_time_bound,
    innovation_bound,
    select_affine_action,
)
from multistate_certificate import (
    aggregate_td_noise,
    build_transfer_mrp,
    certificate_constants,
)


def test_zero_delay_stationary_limit_matches_closed_form() -> None:
    model = build_transfer_mrp(0.0)
    constants = certificate_constants(
        model, 4, 0.0, np.zeros(4, dtype=int), delta=0.0
    )
    omega = aggregate_td_noise(model, 4, 0.0)
    eta = 0.001
    result = affine_bound_components(
        constants, omega, innovation_bound(model), 0.0, eta
    )
    expected_c = (
        1.0
        - eta * constants["monotonicity"]
        + 2.0 * eta * eta * constants["curvature"]
    )
    expected_d = 2.0 * eta * eta * omega
    assert np.isclose(result["contraction"], expected_c)
    assert np.isclose(result["forcing"], expected_d)
    assert np.isclose(
        result["residual"], expected_d / (1.0 - expected_c)
    )


def test_delay_young_parameter_reconstructs_square_factor() -> None:
    model = build_transfer_mrp(0.9)
    constants = certificate_constants(
        model, 32, 0.0, np.arange(32) % 9, delta=1e-4
    )
    omega = aggregate_td_noise(model, 32, 0.0)
    result = affine_bound_components(
        constants, omega, innovation_bound(model), 1e-4, 1e-4
    )
    expected = (
        np.sqrt(result["a_delta"]) + np.sqrt(result["h_delay"])
    ) ** 2
    assert np.isclose(result["contraction"], expected)
    assert result["young_lambda"] > 0.0


def test_finite_time_bound_decreases_to_residual() -> None:
    components = {"contraction": 0.9, "forcing": 0.01}
    short = affine_finite_time_bound(1.0, 1, 0, components)
    long = affine_finite_time_bound(1.0, 100, 0, components)
    assert long["finite_time_bound"] < short["finite_time_bound"]
    assert long["finite_time_bound"] >= long["residual"]
    assert np.isclose(long["residual"], 0.1)


def test_selected_multistate_action_is_strictly_certified() -> None:
    model = build_transfer_mrp(0.9)
    actions = affine_candidate_actions(
        model, rho=0.9, maximum_delay=8, agent_counts=(1, 2)
    )
    selected = select_affine_action(actions)
    assert selected["eta"] > 0.0
    assert selected["contraction"] < 1.0
    assert selected["finite_time_bound"] >= selected["residual"]
    assert (
        selected["updates"] * selected["update_cost"]
        <= selected["resource_budget"]
    )
