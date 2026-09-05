from __future__ import annotations

import math

import numpy as np
import pytest

from .tabular_actor_critic_interface import (
    finite_horizon_return_bound,
    finite_horizon_value_and_occupancy,
    tabular_critic_geometry,
    tabular_critic_operator_policy_lipschitz_bound,
    tabular_critic_population_operator,
    tabular_value_target_lipschitz_bound,
    td_actor_critic_sensitivity_bound,
    td_actor_population_mean,
)
from .trajectory_interface import exact_policy_gradient


def _random_game(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    agents, states, actions = 2, 2, 2
    profiles = actions**agents
    transition = rng.uniform(0.2, 1.0, size=(states, profiles, states))
    transition /= np.sum(transition, axis=-1, keepdims=True)
    reward = rng.uniform(-0.8, 0.8, size=(states, profiles))
    start = np.asarray([0.45, 0.55])
    logits = rng.normal(scale=0.35, size=(agents, states, actions))
    return transition, reward, start, logits


def test_exact_value_critic_makes_td_actor_mean_equal_policy_gradient() -> None:
    for seed in range(2201, 2221):
        transition, reward, start, logits = _random_game(seed)
        values, occupancy = finite_horizon_value_and_occupancy(
            transition, reward, start, logits, 0.81, 4
        )
        td_mean = td_actor_population_mean(
            transition, reward, start, logits, 0.81, 4, values[:-1]
        )
        exact = exact_policy_gradient(
            transition, reward, start, logits, 0.81, horizon=4
        )[1]
        assert td_mean == pytest.approx(exact, abs=2e-12)
        assert np.sum(occupancy, axis=1) == pytest.approx(np.ones(4))


def test_td_actor_critic_error_bound_covers_exact_population_bias() -> None:
    rng = np.random.default_rng(2301)
    discount, horizon = 0.76, 5
    sensitivity = td_actor_critic_sensitivity_bound(
        horizon=horizon, discount=discount, score_norm_bound=math.sqrt(2.0)
    )
    for seed in range(2302, 2322):
        transition, reward, start, logits = _random_game(seed)
        target = finite_horizon_value_and_occupancy(
            transition, reward, start, logits, discount, horizon
        )[0][:-1]
        critic = target + rng.normal(scale=0.3, size=target.shape)
        exact = exact_policy_gradient(
            transition, reward, start, logits, discount, horizon=horizon
        )[1]
        approximate = td_actor_population_mean(
            transition, reward, start, logits, discount, horizon, critic
        )
        radius = sensitivity * np.linalg.norm(critic - target)
        for agent in range(logits.shape[0]):
            assert np.linalg.norm(approximate[agent] - exact[agent]) <= radius + 1e-12


def test_tabular_critic_operator_has_declared_spd_geometry() -> None:
    transition, reward, start, logits = _random_game(2401)
    geometry = tabular_critic_geometry(
        transition, reward, start, logits, 0.8, 4
    )
    critic = geometry.target + 0.2
    operator = tabular_critic_population_operator(critic, geometry)
    assert operator == pytest.approx(0.2 * geometry.diagonal)
    assert geometry.strong_convexity > 0.0
    assert geometry.strong_convexity <= geometry.smoothness <= 0.25 + 1e-12


def test_value_target_lipschitz_bound_covers_logit_changes() -> None:
    rng = np.random.default_rng(2501)
    discount, horizon = 0.79, 4
    for seed in range(2502, 2522):
        transition, reward, start, logits = _random_game(seed)
        shifted = logits.copy()
        shifted[1] += rng.normal(scale=0.25, size=shifted[1].shape)
        first = finite_horizon_value_and_occupancy(
            transition, reward, start, logits, discount, horizon
        )[0][:-1]
        second = finite_horizon_value_and_occupancy(
            transition, reward, start, shifted, discount, horizon
        )[0][:-1]
        coefficient = tabular_value_target_lipschitz_bound(
            agents_changed=1,
            states=2,
            actions=2,
            horizon=horizon,
            discount=discount,
            reward_bound=float(np.max(np.abs(reward))),
        )
        displacement = np.linalg.norm(shifted[1] - logits[1])
        assert np.linalg.norm(first - second) <= coefficient * displacement + 1e-12


def test_critic_operator_policy_bound_covers_same_critic_comparison() -> None:
    rng = np.random.default_rng(2601)
    discount, horizon = 0.82, 5
    for seed in range(2602, 2622):
        transition, reward, start, logits = _random_game(seed)
        shifted = logits.copy()
        shifted[1] += rng.normal(scale=0.2, size=shifted[1].shape)
        first = tabular_critic_geometry(
            transition, reward, start, logits, discount, horizon
        )
        second = tabular_critic_geometry(
            transition, reward, start, shifted, discount, horizon
        )
        value_bound = finite_horizon_return_bound(
            horizon=horizon,
            discount=discount,
            reward_bound=float(np.max(np.abs(reward))),
        )
        critic = rng.uniform(-value_bound, value_bound, size=first.target.shape)
        difference = np.linalg.norm(
            tabular_critic_population_operator(critic, first)
            - tabular_critic_population_operator(critic, second)
        )
        coefficient = tabular_critic_operator_policy_lipschitz_bound(
            agents_changed=1,
            states=2,
            actions=2,
            horizon=horizon,
            discount=discount,
            reward_bound=float(np.max(np.abs(reward))),
            critic_abs_bound=value_bound,
        )
        displacement = np.linalg.norm(shifted[1] - logits[1])
        assert difference <= coefficient * displacement + 1e-12


def test_one_step_actor_has_no_value_critic_bias() -> None:
    assert td_actor_critic_sensitivity_bound(
        horizon=1, discount=0.8, score_norm_bound=math.sqrt(2.0)
    ) == 0.0


def test_invalid_critic_shape_is_rejected() -> None:
    transition, reward, start, logits = _random_game(2701)
    geometry = tabular_critic_geometry(
        transition, reward, start, logits, 0.8, 3
    )
    with pytest.raises(ValueError):
        tabular_critic_population_operator(np.zeros((2, 2)), geometry)
