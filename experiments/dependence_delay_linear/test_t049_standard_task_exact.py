"""Tests for exact standard-task TD lag covariances and switch coupling."""

from __future__ import annotations

import numpy as np

from experiments.dependence_delay_linear.t047_scheduled_participation import (
    prefix_overlap_factor,
)
from experiments.dependence_delay_linear.t049_standard_task_exact import (
    build_exact_projected_task,
    exact_gradient_lag_covariances,
    trajectory_switch_overlap_factor,
)


def frozenlake_task() -> dict[str, object]:
    return build_exact_projected_task(
        environment_id="FrozenLake-v1",
        kwargs={"map_name": "4x4", "is_slippery": True},
        feature_dimension=4,
        discount=0.95,
        policy_epsilon=0.1,
    )


def test_exact_task_invariants() -> None:
    task = frozenlake_task()
    transition = task["continuing_transition"]
    stationary = task["stationary"]
    features = task["features"]
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert np.allclose(stationary @ transition, stationary)
    assert np.allclose(
        features.T @ (stationary[:, None] * features), np.eye(4), atol=1e-10
    )
    assert task["drift_minimum"] > 0.0
    assert 0.0 <= task["mixing_slem"] < 1.0


def test_lag_covariance_orientation_and_zero_lag() -> None:
    task = frozenlake_task()
    lags = exact_gradient_lag_covariances(task, horizon=5)
    assert lags.shape == (9, 4, 4)
    assert np.allclose(lags[4], task["gradient_second_moment"])
    for lag in range(1, 5):
        assert np.allclose(lags[4 - lag], lags[4 + lag].T)


def test_lag_one_matches_direct_edge_enumeration() -> None:
    task = frozenlake_task()
    direct_past_future = np.einsum(
        "s,sud,ue->de",
        task["stationary"],
        task["edge_gradient_sum"],
        task["conditional_gradient"],
        optimize=True,
    )
    lags = exact_gradient_lag_covariances(task, horizon=2)
    assert np.allclose(lags[2], direct_past_future.T, atol=1e-13)


def test_trajectory_switch_factor_matches_prefix_formula() -> None:
    for q_left in (1, 3, 8):
        for q_right in (1, 4, 8):
            for rho in (0.0, 0.2, 0.9, 1.0):
                assert np.isclose(
                    trajectory_switch_overlap_factor(q_left, q_right, rho),
                    prefix_overlap_factor(q_left, q_right, rho),
                )


def test_switch_coupling_preserves_each_actor_marginal() -> None:
    rng = np.random.default_rng(49)
    repetitions = 200_000
    rho = 0.36
    switch = rng.random(repetitions) < np.sqrt(rho)
    common = rng.normal(loc=2.0, scale=3.0, size=repetitions)
    private = rng.normal(loc=2.0, scale=3.0, size=repetitions)
    actor = np.where(switch, common, private)
    assert np.isclose(np.mean(actor), 2.0, atol=0.02)
    assert np.isclose(np.var(actor), 9.0, rtol=0.015)


def test_switch_coupling_monte_carlo_cross_time_overlap() -> None:
    rng = np.random.default_rng(20260803)
    repetitions = 300_000
    q_left = 3
    q_right = 7
    rho = 0.49
    coefficient = 0.4
    maximum_agents = max(q_left, q_right)
    switches = rng.random((repetitions, maximum_agents)) < np.sqrt(rho)
    common_left = rng.normal(size=repetitions)
    common_right = (
        coefficient * common_left
        + np.sqrt(1.0 - coefficient**2) * rng.normal(size=repetitions)
    )
    private_left = rng.normal(size=(repetitions, maximum_agents))
    private_right = (
        coefficient * private_left
        + np.sqrt(1.0 - coefficient**2)
        * rng.normal(size=(repetitions, maximum_agents))
    )
    actors_left = np.where(switches, common_left[:, None], private_left)
    actors_right = np.where(switches, common_right[:, None], private_right)
    average_left = np.mean(actors_left[:, :q_left], axis=1)
    average_right = np.mean(actors_right[:, :q_right], axis=1)
    empirical = float(np.mean(average_right * average_left))
    exact = coefficient * trajectory_switch_overlap_factor(
        q_left, q_right, rho
    )
    assert np.isclose(empirical, exact, rtol=0.012, atol=0.002)


def test_task_build_is_byte_stable_in_public_arrays() -> None:
    first = frozenlake_task()
    second = frozenlake_task()
    assert first["kernel_sha256"] == second["kernel_sha256"]
    assert np.array_equal(first["policy"], second["policy"])
    assert np.array_equal(
        exact_gradient_lag_covariances(first, horizon=8),
        exact_gradient_lag_covariances(second, horizon=8),
    )
