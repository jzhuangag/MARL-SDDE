from __future__ import annotations

import numpy as np
import pytest

from .coupled_actor_critic_drift import (
    box_qp_objective,
    choose_coupled_actor_critic_scales,
    coupled_drift_coefficients,
    direct_expected_drift,
    solve_two_dimensional_box_qp,
)


def test_two_dimensional_solver_matches_dense_grid() -> None:
    quadratic = np.asarray([[2.4, -0.7], [-0.7, 1.3]])
    linear = np.asarray([-0.83, -0.41])
    decision = solve_two_dimensional_box_qp(
        linear=linear,
        quadratic=quadratic,
        upper=np.asarray([0.8, 0.9]),
    )
    grid_a = np.linspace(0.0, 0.8, 801)
    grid_b = np.linspace(0.0, 0.9, 901)
    aa, bb = np.meshgrid(grid_a, grid_b, indexing="ij")
    values = (
        linear[0] * aa
        + linear[1] * bb
        + 0.5 * quadratic[0, 0] * aa * aa
        + quadratic[0, 1] * aa * bb
        + 0.5 * quadratic[1, 1] * bb * bb
    )
    brute = float(np.min(values))
    assert decision.objective <= brute + 2e-6


def test_exact_drift_expansion_agrees_with_direct_evaluation() -> None:
    inputs = {
        "actor_error": np.asarray([0.8, -0.4, 0.2]),
        "birth_actor_snapshot": np.asarray([0.8, -0.1, 0.5]),
        "game_hessian": np.asarray(
            [[1.5, 0.2, -0.1], [0.2, 1.3, 0.15], [-0.1, 0.15, 1.1]]
        ),
        "owner": 0,
        "critic_error": 0.35,
        "critic_bias_sensitivity": 0.4,
        "critic_target_sensitivity": 0.7,
        "critic_contraction": 0.8,
        "critic_weight": 1.2,
        "actor_noise_variance": 0.03,
        "critic_noise_variance": 0.02,
    }
    linear, quadratic, _ = coupled_drift_coefficients(**inputs)
    action = np.asarray([0.17, 0.42])
    expanded = box_qp_objective(action, linear, quadratic)
    assert direct_expected_drift(alpha=action[0], beta=action[1], **inputs) == pytest.approx(
        expanded, abs=1e-13
    )
    assert np.linalg.eigvalsh(quadratic)[0] >= -1e-12


def test_owner_snapshot_difference_does_not_create_staleness() -> None:
    common = {
        "actor_error": np.asarray([0.4, -0.2]),
        "game_hessian": np.asarray([[1.4, 0.3], [0.3, 1.2]]),
        "owner": 0,
        "critic_error": 0.1,
        "critic_bias_sensitivity": 0.5,
        "critic_target_sensitivity": 0.4,
        "critic_contraction": 0.7,
        "critic_weight": 1.0,
        "actor_noise_variance": 0.0,
        "critic_noise_variance": 0.0,
    }
    _, _, first = coupled_drift_coefficients(
        birth_actor_snapshot=np.asarray([100.0, -0.2]), **common
    )
    _, _, second = coupled_drift_coefficients(
        birth_actor_snapshot=np.asarray([-100.0, -0.2]), **common
    )
    assert first["strategic_staleness"] == pytest.approx(0.0)
    assert second["strategic_staleness"] == pytest.approx(0.0)
    assert first["packet_gradient"] == pytest.approx(second["packet_gradient"])


def test_teammate_motion_and_critic_error_decompose_packet_bias() -> None:
    actor = np.asarray([0.7, -0.5])
    birth = np.asarray([0.7, 0.1])
    hessian = np.asarray([[1.3, 0.4], [0.4, 1.1]])
    _, _, diagnostics = coupled_drift_coefficients(
        actor_error=actor,
        birth_actor_snapshot=birth,
        game_hessian=hessian,
        owner=0,
        critic_error=0.25,
        critic_bias_sensitivity=0.6,
        critic_target_sensitivity=0.5,
        critic_contraction=0.8,
        critic_weight=1.0,
        actor_noise_variance=0.0,
        critic_noise_variance=0.0,
    )
    difference = diagnostics["packet_gradient"] - diagnostics["current_gradient"]
    assert difference == pytest.approx(
        diagnostics["strategic_staleness"] + diagnostics["critic_bias"]
    )


def test_moving_critic_target_creates_a_genuine_joint_action() -> None:
    inputs = {
        "actor_error": np.asarray([0.9, -0.35]),
        "birth_actor_snapshot": np.asarray([0.9, 0.25]),
        "game_hessian": np.asarray([[1.5, 0.45], [0.45, 1.4]]),
        "owner": 0,
        "critic_error": 0.55,
        "critic_bias_sensitivity": 0.5,
        "critic_target_sensitivity": 1.1,
        "critic_contraction": 0.75,
        "critic_weight": 1.4,
        "actor_noise_variance": 0.02,
        "critic_noise_variance": 0.01,
    }
    coupled = choose_coupled_actor_critic_scales(
        alpha_cap=0.8, beta_cap=0.9, **inputs
    )
    linear, quadratic, _ = coupled_drift_coefficients(**inputs)
    diagonal = np.diag(np.diag(quadratic))
    uncoupled_action = solve_two_dimensional_box_qp(
        linear=linear,
        quadratic=diagonal,
        upper=np.asarray([0.8, 0.9]),
    ).action
    uncoupled_true_drift = box_qp_objective(uncoupled_action, linear, quadratic)
    assert coupled.cross_curvature != pytest.approx(0.0)
    assert coupled.expected_drift < uncoupled_true_drift - 1e-5
    assert 0.0 <= coupled.alpha <= 0.8
    assert 0.0 <= coupled.beta <= 0.9


@pytest.mark.parametrize(
    "mutation",
    [
        {"owner": 4},
        {"critic_weight": 0.0},
        {"actor_noise_variance": -0.1},
        {"game_hessian": np.asarray([[1.0, 2.0], [2.0, 1.0]])},
    ],
)
def test_invalid_drift_inputs_are_rejected(mutation: dict[str, object]) -> None:
    inputs: dict[str, object] = {
        "actor_error": np.asarray([0.5, -0.2]),
        "birth_actor_snapshot": np.asarray([0.5, -0.1]),
        "game_hessian": np.asarray([[1.2, 0.2], [0.2, 1.1]]),
        "owner": 0,
        "critic_error": 0.2,
        "critic_bias_sensitivity": 0.4,
        "critic_target_sensitivity": 0.6,
        "critic_contraction": 0.7,
        "critic_weight": 1.0,
        "actor_noise_variance": 0.01,
        "critic_noise_variance": 0.01,
    }
    inputs.update(mutation)
    with pytest.raises(ValueError):
        coupled_drift_coefficients(**inputs)
