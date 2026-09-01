from __future__ import annotations

import math

import numpy as np
import pytest

from .trajectory_interface import (
    enumerate_reinforce_expectation,
    exact_policy_gradient,
    finite_horizon_teammate_gradient_change_bound,
    reinforce_packet_norm_bound,
    softmax_total_variation_lipschitz,
    softmax_nash_gap_certificate,
    truncation_gradient_bias_bound,
)


def _random_game(seed: int = 90701) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    agents, states, actions = 2, 2, 2
    profiles = actions**agents
    transition = rng.uniform(0.1, 1.0, size=(states, profiles, states))
    transition /= np.sum(transition, axis=-1, keepdims=True)
    reward = rng.uniform(-1.0, 1.0, size=(states, profiles))
    start = np.asarray([0.35, 0.65])
    logits = rng.normal(scale=0.4, size=(agents, states, actions))
    return transition, reward, start, logits


def _finite_difference(
    transition: np.ndarray,
    reward: np.ndarray,
    start: np.ndarray,
    logits: np.ndarray,
    discount: float,
    horizon: int | None,
) -> np.ndarray:
    result = np.zeros_like(logits)
    epsilon = 2e-6
    for index in np.ndindex(logits.shape):
        positive = logits.copy()
        negative = logits.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        plus, _ = exact_policy_gradient(
            transition, reward, start, positive, discount, horizon=horizon
        )
        minus, _ = exact_policy_gradient(
            transition, reward, start, negative, discount, horizon=horizon
        )
        result[index] = (plus-minus)/(2.0*epsilon)
    return result


@pytest.mark.parametrize("horizon", [3, None])
def test_exact_markov_game_gradient_matches_finite_difference(
    horizon: int | None,
) -> None:
    transition, reward, start, logits = _random_game()
    _, gradient = exact_policy_gradient(
        transition, reward, start, logits, 0.83, horizon=horizon
    )
    finite = _finite_difference(
        transition, reward, start, logits, 0.83, horizon
    )
    assert gradient == pytest.approx(finite, abs=2e-8)


def test_exact_trajectory_enumeration_is_conditionally_unbiased() -> None:
    transition, reward, start, logits = _random_game(90702)
    dynamic_value, dynamic_gradient = exact_policy_gradient(
        transition, reward, start, logits, 0.77, horizon=3
    )
    enumerated_value, enumerated_gradient = enumerate_reinforce_expectation(
        transition, reward, start, logits, 0.77, horizon=3
    )
    assert enumerated_value == pytest.approx(dynamic_value, abs=1e-12)
    assert enumerated_gradient == pytest.approx(dynamic_gradient, abs=1e-12)


def test_truncation_bias_bound_covers_exact_gradient_gap() -> None:
    transition, reward, start, logits = _random_game(90703)
    discount = 0.72
    infinite = exact_policy_gradient(
        transition, reward, start, logits, discount, horizon=None
    )[1]
    reward_bound = float(np.max(np.abs(reward)))
    # The Euclidean norm of a categorical-softmax score is at most sqrt(2).
    score_bound = math.sqrt(2.0)
    for horizon in (1, 2, 4, 8):
        truncated = exact_policy_gradient(
            transition, reward, start, logits, discount, horizon=horizon
        )[1]
        bound = truncation_gradient_bias_bound(
            horizon, discount, reward_bound, score_bound
        )
        for agent in range(logits.shape[0]):
            assert np.linalg.norm(infinite[agent]-truncated[agent]) <= bound+1e-12


def test_packet_bound_is_finite_and_below_universal_infinite_bound() -> None:
    horizon, discount = 7, 0.8
    bound = reinforce_packet_norm_bound(
        horizon, discount, reward_bound=1.5, score_norm_bound=math.sqrt(2.0)
    )
    universal = math.sqrt(2.0)*1.5/(1.0-discount)**2
    assert 0.0 < bound < universal


def test_softmax_gradient_certificate_covers_exact_unilateral_nash_gaps() -> None:
    for seed in range(90710, 90730):
        transition, reward, start, logits = _random_game(seed)
        certificate = softmax_nash_gap_certificate(
            transition, reward, start, logits, discount=0.78
        )
        assert (
            np.asarray(certificate["nash_gaps"])
            <= np.asarray(certificate["nash_gap_bounds"])+2e-11
        ).all()
        assert (np.asarray(certificate["occupancy_mismatch"]) >= 1.0-1e-12).all()
        assert (
            np.asarray(certificate["minimum_action_probabilities"]) > 0.0
        ).all()


def test_trajectory_interface_rejects_invalid_inputs() -> None:
    transition, reward, start, logits = _random_game(90704)
    transition[0, 0, 0] += 0.2
    with pytest.raises(ValueError):
        exact_policy_gradient(
            transition, reward, start, logits, 0.9, horizon=3
        )
    with pytest.raises(ValueError):
        truncation_gradient_bias_bound(0, 0.9, 1.0, 1.0)


def test_softmax_tv_lipschitz_covers_random_finite_logit_changes() -> None:
    rng = np.random.default_rng(90801)
    for actions in (2, 3, 7):
        coefficient = softmax_total_variation_lipschitz(actions)
        for _ in range(200):
            first = rng.normal(size=actions)
            second = rng.normal(size=actions)
            first_policy = np.exp(first-np.max(first))
            first_policy /= np.sum(first_policy)
            second_policy = np.exp(second-np.max(second))
            second_policy /= np.sum(second_policy)
            total_variation = 0.5*float(np.sum(np.abs(first_policy-second_policy)))
            assert total_variation <= coefficient*np.linalg.norm(first-second)+1e-12


def test_teammate_logit_bound_covers_exact_owner_gradient_change() -> None:
    discount, horizon = 0.78, 4
    for seed in range(90810, 90830):
        transition, reward, start, logits = _random_game(seed)
        shifted = logits.copy()
        rng = np.random.default_rng(seed+10_000)
        shifted[1] += rng.normal(scale=0.35, size=shifted[1].shape)
        first = exact_policy_gradient(
            transition, reward, start, logits, discount, horizon=horizon
        )[1][0]
        second = exact_policy_gradient(
            transition, reward, start, shifted, discount, horizon=horizon
        )[1][0]
        maximum_state_shift = float(
            np.max(np.linalg.norm(shifted[1]-logits[1], axis=-1))
        )
        bound = finite_horizon_teammate_gradient_change_bound(
            horizon,
            discount,
            float(np.max(np.abs(reward))),
            math.sqrt(2.0),
            teammate_actions=logits.shape[-1],
            maximum_state_logit_shift=maximum_state_shift,
        )
        assert np.linalg.norm(first-second) <= bound+1e-12


def test_cross_sensitivity_helpers_reject_malformed_arguments() -> None:
    with pytest.raises(ValueError):
        softmax_total_variation_lipschitz(0)
    with pytest.raises(ValueError):
        finite_horizon_teammate_gradient_change_bound(
            4, 0.8, 1.0, 1.0, 2, -0.1
        )
