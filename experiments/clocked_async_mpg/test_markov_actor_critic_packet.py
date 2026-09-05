from __future__ import annotations

import math

import numpy as np
import pytest

from .markov_actor_critic_packet import (
    bounded_version_displacement,
    build_actor_critic_packet_certificate,
    contracted_critic_radius,
    simultaneous_vector_mean_radius,
)


def test_joint_schedule_radius_has_declared_union_bound_formula() -> None:
    radius = simultaneous_vector_mean_radius(
        vector_dimension=4,
        trajectory_count=64,
        trajectory_norm_bound=3.0,
        scheduled_packet_count=200,
        joint_coordinate_count=11,
        failure_probability=0.02,
    )
    expected = math.sqrt(4) * 3.0 * math.sqrt(
        2.0 * math.log(2.0 * 200 * 11 / 0.02) / 64
    )
    assert radius == pytest.approx(expected)


def test_joint_schedule_radius_shrinks_with_independent_trajectories() -> None:
    common = dict(
        vector_dimension=7,
        trajectory_norm_bound=2.0,
        scheduled_packet_count=50,
        joint_coordinate_count=13,
        failure_probability=0.05,
    )
    small = simultaneous_vector_mean_radius(trajectory_count=16, **common)
    large = simultaneous_vector_mean_radius(trajectory_count=64, **common)
    assert large == pytest.approx(small / 2.0)


def test_zero_packet_norm_has_zero_radius() -> None:
    assert simultaneous_vector_mean_radius(
        vector_dimension=2,
        trajectory_count=1,
        trajectory_norm_bound=0.0,
        scheduled_packet_count=1,
        joint_coordinate_count=2,
        failure_probability=0.1,
    ) == 0.0


def test_actor_and_critic_radii_are_additive_without_hidden_discount() -> None:
    certificate = build_actor_critic_packet_certificate(
        actor_statistical_radius=0.10,
        actor_truncation_radius=0.02,
        actor_critic_sensitivity=0.4,
        birth_critic_radius=0.5,
        actor_policy_lipschitz=0.7,
        policy_version_displacement=0.3,
        critic_statistical_radius=0.08,
        critic_policy_lipschitz=0.6,
        critic_parameter_lipschitz=0.9,
        critic_version_displacement=0.2,
    )
    assert certificate.actor_critic_bias_radius == pytest.approx(0.20)
    assert certificate.actor_version_radius == pytest.approx(0.21)
    assert certificate.actor_total_radius == pytest.approx(0.53)
    assert certificate.critic_policy_version_radius == pytest.approx(0.18)
    assert certificate.critic_parameter_version_radius == pytest.approx(0.18)
    assert certificate.critic_total_radius == pytest.approx(0.44)


def test_bounded_delay_path_uses_every_intervening_update() -> None:
    assert bounded_version_displacement(
        max_intervening_updates=7, max_update_norm=0.03
    ) == pytest.approx(0.21)


def test_spd_critic_contraction_covers_random_exact_steps() -> None:
    rng = np.random.default_rng(1843)
    for _ in range(200):
        dimension = 5
        basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        eigenvalues = rng.uniform(0.4, 1.7, size=dimension)
        matrix = basis @ np.diag(eigenvalues) @ basis.T
        mu = float(np.min(eigenvalues))
        smoothness = float(np.max(eigenvalues))
        beta = float(rng.uniform(0.0, 1.0 / smoothness))
        error = rng.normal(size=dimension)
        radius = float(np.linalg.norm(error))
        innovation = rng.normal(size=dimension)
        innovation *= 0.13 / max(float(np.linalg.norm(innovation)), 1e-15)
        actual = np.linalg.norm(error - beta * (matrix @ error + innovation))
        bound = contracted_critic_radius(
            critic_radius=radius,
            critic_gradient_radius=0.13,
            strong_convexity=mu,
            smoothness=smoothness,
            step_size=beta,
        )
        assert actual <= bound + 1e-12


def test_critic_step_rejects_mu_only_but_not_smoothness_safe_scale() -> None:
    with pytest.raises(ValueError, match="contraction interval"):
        contracted_critic_radius(
            critic_radius=1.0,
            critic_gradient_radius=0.0,
            strong_convexity=0.5,
            smoothness=4.0,
            step_size=0.4,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vector_dimension": 0},
        {"trajectory_count": 0},
        {"failure_probability": 1.0},
        {"joint_coordinate_count": 1},
    ],
)
def test_invalid_simultaneous_radius_inputs_are_rejected(kwargs: dict[str, float]) -> None:
    values = dict(
        vector_dimension=2,
        trajectory_count=4,
        trajectory_norm_bound=1.0,
        scheduled_packet_count=3,
        joint_coordinate_count=4,
        failure_probability=0.05,
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        simultaneous_vector_mean_radius(**values)


def test_invalid_version_delay_is_rejected() -> None:
    with pytest.raises(ValueError):
        bounded_version_displacement(
            max_intervening_updates=-1, max_update_norm=0.2
        )
