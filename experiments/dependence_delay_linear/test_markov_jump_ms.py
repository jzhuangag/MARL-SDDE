"""Tests for the exact EXP-008B Markov-jump operators."""

import numpy as np

from exact_lifted_ms import dense_lifted_matrix, delay_counts
from markov_jump_ms import (
    aggregate_same_time_curvature,
    covariance_operator_coefficients,
    direct_conditional_operator,
    homogeneous_delays,
    minimum_decorrelation_gap,
    mixing_tv_after_gap,
    polynomial_matrix,
    registered_expanding_td_model,
    registered_delays,
    registered_td_model,
    sharp_theorem_steps,
    theorem_safe_step,
    thinned_persistence,
)


def test_registered_model_is_normalized_and_monotone() -> None:
    model = registered_td_model()
    assert np.allclose(model["weights"].sum(axis=1), 1.0)
    symmetric = 0.5 * (
        model["stationary_mean"] + model["stationary_mean"].T
    )
    assert np.min(np.linalg.eigvalsh(symmetric)) > 0.0


def test_conditional_moment_operator_matches_direct_enumeration() -> None:
    model = registered_td_model()
    delays = registered_delays(2, 2)
    rho = 0.37
    eta = 0.23
    conditional = dense_lifted_matrix(
        eta,
        model["conditional_means"][0],
        model["jacobians"],
        model["weights"][0],
        delay_counts(delays),
        rho,
    )
    direct = direct_conditional_operator(
        model, delays, rho, mode=0, eta=eta
    )
    assert np.max(np.abs(conditional - direct)) <= 1e-11


def test_iid_reduction_at_half_persistence() -> None:
    model = registered_td_model()
    delays = registered_delays(3, 2)
    coefficients = covariance_operator_coefficients(
        model, delays, rho=0.9, persistence=0.5
    )
    markov = polynomial_matrix(coefficients["markov"], 0.15)
    iid = polynomial_matrix(coefficients["iid"], 0.15)
    assert abs(
        np.max(np.abs(np.linalg.eigvals(markov)))
        - np.max(np.abs(np.linalg.eigvals(iid)))
    ) <= 1e-11


def test_one_agent_same_time_curvature_is_rho_invariant() -> None:
    model = registered_td_model()
    low, low_k, _ = aggregate_same_time_curvature(model, 1, 0.0)
    high, high_k, _ = aggregate_same_time_curvature(model, 1, 0.9)
    assert np.allclose(low, high, atol=1e-14)
    assert abs(low_k - high_k) <= 1e-14


def test_expanding_scalar_td_model_has_required_local_and_global_signs() -> None:
    model = registered_expanding_td_model()
    assert np.allclose(
        model["jacobians"].reshape(-1),
        np.asarray((0.1, -0.8, 2.2, 0.4)),
    )
    assert model["conditional_means"][0, 0, 0] < 0.0
    assert model["conditional_means"][1, 0, 0] > 0.0
    assert np.allclose(model["stationary_mean"], np.asarray(((0.655,),)))


def test_decorrelation_gap_is_minimal_and_theorem_root_has_slack() -> None:
    model = registered_expanding_td_model()
    lipschitz = 2.2
    monotonicity = 0.655
    target = monotonicity / (4.0 * lipschitz)
    gap = minimum_decorrelation_gap(0.98, target)
    assert mixing_tv_after_gap(0.98, gap) <= target
    assert mixing_tv_after_gap(0.98, gap - 1) > target
    persistence = thinned_persistence(0.98, gap)
    assert 0.5 < persistence < 0.98
    result = theorem_safe_step(
        model,
        num_agents=4,
        rho=0.9,
        delays=homogeneous_delays(4, 2),
        delta=mixing_tv_after_gap(0.98, gap),
    )
    assert result["eta"] <= 1.0 / result["lipschitz"]
    assert (
        result["polynomial_at_eta"] < result["right_hand_side"]
    )
    assert result["contraction_coefficient"] < 1.0


def test_sharp_theorem_scalar_solves_are_internal_and_improve_delay() -> None:
    model = registered_expanding_td_model()
    target = 0.655 / (4.0 * 2.2)
    gap = minimum_decorrelation_gap(0.98, target)
    result = sharp_theorem_steps(
        model,
        num_agents=4,
        rho=0.9,
        delays=homogeneous_delays(4, 2),
        delta=mixing_tv_after_gap(0.98, gap),
    )
    assert abs(result["sharp_root_factor"] - 1.0) <= 1e-12
    assert 0.0 < result["rate_eta"] < result["sharp_root"]
    assert result["rate_factor"] < result["sharp_safe_factor"]
    assert result["sharp_safe_eta"] > result["coarse_eta"]
