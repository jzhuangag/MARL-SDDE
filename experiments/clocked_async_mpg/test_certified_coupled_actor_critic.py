from __future__ import annotations

import numpy as np
import pytest

from .certified_coupled_actor_critic import (
    certified_drift_coefficients,
    certified_drift_upper,
    choose_certified_coupled_scales,
    clipped_margin_progress_lower,
    next_critic_radius,
)


BASE = {
    "gradient_norm": 1.1,
    "actor_error_radius": 0.18,
    "critic_radius": 0.55,
    "critic_gradient_radius": 0.08,
    "critic_contraction": 0.8,
    "target_sensitivity": 0.7,
    "block_smoothness": 1.4,
    "history_curvature": 0.25,
    "critic_weight": 0.6,
}


def test_quadratic_matches_direct_certificate_recursion() -> None:
    alpha, beta = 0.21, 0.62
    drift = certified_drift_upper(alpha=alpha, beta=beta, **BASE)
    gradient = BASE["gradient_norm"]
    alignment = gradient * (gradient - BASE["actor_error_radius"])
    actor = (
        -alpha * alignment
        + 0.5
        * (BASE["block_smoothness"] + BASE["history_curvature"])
        * alpha**2
        * gradient**2
    )
    propagated = next_critic_radius(
        alpha=alpha,
        beta=beta,
        gradient_norm=gradient,
        critic_radius=BASE["critic_radius"],
        critic_gradient_radius=BASE["critic_gradient_radius"],
        critic_contraction=BASE["critic_contraction"],
        target_sensitivity=BASE["target_sensitivity"],
    )
    critic = BASE["critic_weight"] * (
        propagated**2 - BASE["critic_radius"] ** 2
    )
    assert drift == pytest.approx(actor + critic, abs=1e-13)


def test_certificate_qp_is_convex_and_no_worse_than_zero_action() -> None:
    decision = choose_certified_coupled_scales(
        alpha_cap=0.7, beta_cap=1.0, **BASE
    )
    assert np.linalg.eigvalsh(decision.quadratic)[0] >= -1e-12
    assert decision.certified_drift <= 1e-12
    assert 0.0 <= decision.alpha <= 0.7
    assert 0.0 <= decision.beta <= 1.0


def test_uncertified_actor_packet_is_rejected_while_critic_can_contract() -> None:
    values = dict(BASE)
    values["actor_error_radius"] = 2.0
    decision = choose_certified_coupled_scales(
        alpha_cap=0.7, beta_cap=1.0, **values
    )
    assert decision.alpha == pytest.approx(0.0, abs=1e-12)
    assert decision.beta > 0.0
    assert decision.next_critic_radius < decision.critic_radius


def test_critic_action_turns_off_below_its_certificate_floor() -> None:
    values = dict(BASE)
    values["critic_radius"] = 0.05
    values["critic_gradient_radius"] = 0.08
    values["critic_contraction"] = 0.8
    decision = choose_certified_coupled_scales(
        alpha_cap=0.0, beta_cap=1.0, **values
    )
    assert decision.beta == pytest.approx(0.0, abs=1e-12)
    assert decision.next_critic_radius == pytest.approx(values["critic_radius"])


def test_target_motion_produces_cross_action_curvature() -> None:
    linear, quadratic, _ = certified_drift_coefficients(**BASE)
    assert quadratic[0, 1] < 0.0
    no_target = dict(BASE)
    no_target["target_sensitivity"] = 0.0
    _, reduced, _ = certified_drift_coefficients(**no_target)
    assert reduced[0, 1] == pytest.approx(0.0, abs=1e-14)
    assert np.isfinite(linear).all()


def test_actor_smoothness_and_critic_radius_bound_actual_composite_change() -> None:
    decision = choose_certified_coupled_scales(
        alpha_cap=0.7, beta_cap=1.0, **BASE
    )
    gradient = BASE["gradient_norm"]
    true_gradient = gradient - BASE["actor_error_radius"]
    actor_change = (
        -decision.alpha * true_gradient * gradient
        + 0.5 * BASE["block_smoothness"] * decision.alpha**2 * gradient**2
    )
    actual_critic_error = BASE["critic_radius"]
    adverse_critic_noise = BASE["critic_gradient_radius"]
    next_error = (
        (1.0 - BASE["critic_contraction"] * decision.beta)
        * actual_critic_error
        + decision.beta * adverse_critic_noise
        + BASE["target_sensitivity"] * decision.alpha * gradient
    )
    certificate_change = actor_change + BASE["critic_weight"] * (
        next_error**2 - BASE["critic_radius"] ** 2
    )
    assert certificate_change <= decision.certified_drift + 1e-12


def test_clipped_margin_lower_bound_is_dominated_by_qp_progress() -> None:
    alpha_cap = 0.7
    decision = choose_certified_coupled_scales(
        alpha_cap=alpha_cap, beta_cap=1.0, **BASE
    )
    lower = clipped_margin_progress_lower(alpha_cap=alpha_cap, **{
        key: BASE[key]
        for key in (
            "gradient_norm",
            "actor_error_radius",
            "critic_radius",
            "target_sensitivity",
            "block_smoothness",
            "history_curvature",
            "critic_weight",
        )
    })
    assert -decision.certified_drift >= lower - 1e-12


@pytest.mark.parametrize(
    "mutation",
    [
        {"gradient_norm": -1.0},
        {"critic_contraction": 0.0},
        {"critic_weight": 0.0},
        {"history_curvature": -0.1},
    ],
)
def test_invalid_certificate_inputs_are_rejected(mutation: dict[str, float]) -> None:
    values = dict(BASE)
    values.update(mutation)
    with pytest.raises(ValueError):
        certified_drift_coefficients(**values)


def test_beta_cap_must_stay_inside_contraction_interval() -> None:
    with pytest.raises(ValueError):
        choose_certified_coupled_scales(alpha_cap=0.5, beta_cap=1.3, **BASE)

