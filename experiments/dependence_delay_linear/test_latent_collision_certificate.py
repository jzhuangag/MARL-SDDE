"""Tests for the latent collision confidence certificate."""

import numpy as np

from latent_collision_certificate import (
    latent_rho_upper,
    minimum_collision_gap,
    sample_hidden_collision,
    stationary_collision_probability,
    symmetric_joint_tv_upper,
    time_uniform_hoeffding_radius,
)


def test_hidden_mixture_collision_identity():
    assert np.isclose(
        stationary_collision_probability(0.6, 0.2),
        0.68,
    )


def test_rho_upper_inverts_collision_with_bias():
    result = latent_rho_upper(
        collisions=60,
        trials=100,
        cumulative_tv_bias=2.0,
        alpha=0.005,
        independent_collision_lower=0.5,
    )
    assert result["collision_upper"] >= 0.62
    assert result["rho_upper"] >= 0.24
    assert 0.0 <= result["rho_upper"] <= 1.0


def test_time_uniform_radius_decreases_eventually():
    assert time_uniform_hoeffding_radius(
        1000, 0.005
    ) < time_uniform_hoeffding_radius(100, 0.005)


def test_minimum_gap_is_exact_and_monotone():
    for persistence in (0.5, 0.9, 0.98):
        gap = minimum_collision_gap(persistence, 0.01)
        assert symmetric_joint_tv_upper(persistence, gap) <= 0.01
        if gap > 1:
            assert (
                symmetric_joint_tv_upper(persistence, gap - 1)
                > 0.01
            )


def test_hidden_collision_sampler_matches_stationary_formula():
    rng = np.random.RandomState(1234)
    states = rng.randint(0, 2, size=3)
    collisions = []
    for _ in range(50000):
        result = sample_hidden_collision(
            rng, states, persistence=0.9, rho=0.6, gap=20
        )
        states = result["states"]
        collisions.append(result["collision"])
    expected = stationary_collision_probability(0.6, 0.5)
    assert abs(np.mean(collisions) - expected) < 0.015
