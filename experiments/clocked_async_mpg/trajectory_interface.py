"""Exact finite-state checks for a theorem-facing Markov-game packet.

The functions in this module are algebraic validation tools.  They evaluate a
factorized softmax joint policy exactly, rather than running a scientific RL
experiment.
"""

from __future__ import annotations

from itertools import product
import math

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def _softmax(logits: Array) -> Array:
    shifted = logits-np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated/np.sum(exponentiated, axis=-1, keepdims=True)


def _validate_game(
    transition: Array,
    reward: Array,
    start_distribution: Array,
    logits: Array,
    discount: float,
) -> tuple[Array, Array, Array, Array, tuple[tuple[int, ...], ...]]:
    transition = np.asarray(transition, dtype=float)
    reward = np.asarray(reward, dtype=float)
    start = np.asarray(start_distribution, dtype=float)
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 3:
        raise ValueError("logits must have shape (agents, states, actions)")
    agents, states, actions = logits.shape
    profiles = tuple(product(range(actions), repeat=agents))
    if transition.shape != (states, len(profiles), states):
        raise ValueError("transition has the wrong shape")
    if reward.shape != (states, len(profiles)):
        raise ValueError("reward has the wrong shape")
    if start.shape != (states,):
        raise ValueError("start_distribution has the wrong shape")
    if not all(
        np.isfinite(value).all()
        for value in (transition, reward, start, logits)
    ):
        raise ValueError("game arrays must be finite")
    if (transition < 0.0).any() or not np.allclose(
        np.sum(transition, axis=-1), 1.0, atol=1e-12, rtol=0.0
    ):
        raise ValueError("transition rows must be probability distributions")
    if (start < 0.0).any() or not math.isclose(
        float(np.sum(start)), 1.0, abs_tol=1e-12
    ):
        raise ValueError("start_distribution must be a probability distribution")
    if not 0.0 < discount < 1.0:
        raise ValueError("discount must lie strictly between zero and one")
    return transition, reward, start, logits, profiles


def _joint_policy(
    logits: Array, profiles: tuple[tuple[int, ...], ...]
) -> tuple[Array, Array]:
    policies = _softmax(logits)
    agents, states, actions = policies.shape
    joint = np.ones((states, len(profiles)), dtype=float)
    for profile_index, profile in enumerate(profiles):
        for agent in range(agents):
            joint[:, profile_index] *= policies[agent, :, profile[agent]]
    return policies, joint


def exact_policy_gradient(
    transition: Array,
    reward: Array,
    start_distribution: Array,
    logits: Array,
    discount: float,
    *,
    horizon: int | None,
) -> tuple[float, Array]:
    """Return the exact discounted value and all softmax-logit gradients.

    ``horizon=None`` evaluates the infinite discounted objective.  A positive
    integer evaluates the truncated objective with rewards at times
    ``0,...,horizon-1``.
    """

    transition, reward, start, logits, profiles = _validate_game(
        transition, reward, start_distribution, logits, discount
    )
    if horizon is not None and horizon <= 0:
        raise ValueError("horizon must be positive or None")
    policies, joint = _joint_policy(logits, profiles)
    agents, states, actions = policies.shape
    gradients = np.zeros_like(logits)

    if horizon is None:
        policy_transition = np.einsum("sa,san->sn", joint, transition)
        policy_reward = np.sum(joint*reward, axis=1)
        value = np.linalg.solve(
            np.eye(states)-discount*policy_transition, policy_reward
        )
        q_value = reward+discount*np.einsum("san,n->sa", transition, value)
        discounted_occupancy = np.linalg.solve(
            (np.eye(states)-discount*policy_transition).T, start
        )
        for state in range(states):
            for profile_index, profile in enumerate(profiles):
                mass = discounted_occupancy[state]*joint[state, profile_index]
                for agent in range(agents):
                    score = -policies[agent, state].copy()
                    score[profile[agent]] += 1.0
                    gradients[agent, state] += (
                        mass*q_value[state, profile_index]*score
                    )
        return float(start@value), gradients

    values = np.zeros((horizon+1, states), dtype=float)
    q_values = np.zeros((horizon, states, len(profiles)), dtype=float)
    for time in range(horizon-1, -1, -1):
        q_values[time] = reward+discount*np.einsum(
            "san,n->sa", transition, values[time+1]
        )
        values[time] = np.sum(joint*q_values[time], axis=1)

    state_distribution = start.copy()
    for time in range(horizon):
        for state in range(states):
            for profile_index, profile in enumerate(profiles):
                mass = state_distribution[state]*joint[state, profile_index]
                for agent in range(agents):
                    score = -policies[agent, state].copy()
                    score[profile[agent]] += 1.0
                    gradients[agent, state] += (
                        discount**time
                        *mass
                        *q_values[time, state, profile_index]
                        *score
                    )
        policy_transition = np.einsum("sa,san->sn", joint, transition)
        state_distribution = state_distribution@policy_transition
    return float(start@values[0]), gradients


def softmax_nash_gap_certificate(
    transition: Array,
    reward: Array,
    start_distribution: Array,
    logits: Array,
    discount: float,
) -> dict[str, Array]:
    """Compute exact unilateral gaps and a softmax-gradient certificate.

    This helper treats ``reward`` as an identical-interest reward.  For each
    agent, it solves the exact best-response MDP against fixed teammate
    policies and evaluates the occupancy-mismatch/softmax-interiority bound.
    """

    transition, reward, start, logits, profiles = _validate_game(
        transition, reward, start_distribution, logits, discount
    )
    policies, joint = _joint_policy(logits, profiles)
    agents, states, actions = policies.shape
    current_value, gradient = exact_policy_gradient(
        transition,
        reward,
        start,
        logits,
        discount,
        horizon=None,
    )
    current_transition = np.einsum("sa,san->sn", joint, transition)
    current_occupancy = (1.0-discount)*np.linalg.solve(
        (np.eye(states)-discount*current_transition).T, start
    )

    gaps = np.zeros(agents, dtype=float)
    mismatch = np.zeros(agents, dtype=float)
    bounds = np.zeros(agents, dtype=float)
    minimum_probability = np.min(policies, axis=(1, 2))
    for agent in range(agents):
        unilateral_reward = np.zeros((states, actions), dtype=float)
        unilateral_transition = np.zeros((states, actions, states), dtype=float)
        for state in range(states):
            for profile_index, profile in enumerate(profiles):
                teammate_probability = 1.0
                for teammate in range(agents):
                    if teammate != agent:
                        teammate_probability *= policies[
                            teammate, state, profile[teammate]
                        ]
                action = profile[agent]
                unilateral_reward[state, action] += (
                    teammate_probability*reward[state, profile_index]
                )
                unilateral_transition[state, action] += (
                    teammate_probability*transition[state, profile_index]
                )

        value = np.zeros(states, dtype=float)
        for _ in range(100_000):
            q_value = unilateral_reward+discount*np.einsum(
                "san,n->sa", unilateral_transition, value
            )
            updated = np.max(q_value, axis=1)
            if float(np.max(np.abs(updated-value))) <= 1e-14:
                value = updated
                break
            value = updated
        else:
            raise RuntimeError("best-response value iteration did not converge")
        q_value = unilateral_reward+discount*np.einsum(
            "san,n->sa", unilateral_transition, value
        )
        best_actions = np.argmax(q_value, axis=1)
        best_transition = unilateral_transition[
            np.arange(states), best_actions
        ]
        best_occupancy = (1.0-discount)*np.linalg.solve(
            (np.eye(states)-discount*best_transition).T, start
        )
        if np.any((current_occupancy <= 0.0)&(best_occupancy > 0.0)):
            mismatch[agent] = math.inf
            bounds[agent] = math.inf
        else:
            positive = current_occupancy > 0.0
            mismatch[agent] = float(
                np.max(best_occupancy[positive]/current_occupancy[positive])
            )
            bounds[agent] = float(
                mismatch[agent]
                *math.sqrt(states)
                /minimum_probability[agent]
                *np.linalg.norm(gradient[agent])
            )
        gaps[agent] = max(0.0, float(start@value-current_value))
    return {
        "gradient_norms": np.linalg.norm(gradient, axis=(1, 2)),
        "minimum_action_probabilities": minimum_probability,
        "nash_gap_bounds": bounds,
        "nash_gaps": gaps,
        "occupancy_mismatch": mismatch,
    }


def enumerate_reinforce_expectation(
    transition: Array,
    reward: Array,
    start_distribution: Array,
    logits: Array,
    discount: float,
    horizon: int,
) -> tuple[float, Array]:
    """Enumerate every finite trajectory and average the REINFORCE packet."""

    transition, reward, start, logits, profiles = _validate_game(
        transition, reward, start_distribution, logits, discount
    )
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    policies, joint = _joint_policy(logits, profiles)
    agents, states, actions = policies.shape
    expected_value = 0.0
    expected_gradient = np.zeros_like(logits)

    def visit(
        time: int,
        state: int,
        probability: float,
        state_history: list[int],
        profile_history: list[int],
        reward_history: list[float],
    ) -> None:
        nonlocal expected_value, expected_gradient
        if time == horizon:
            discounted_rewards = np.asarray(
                [discount**index*value for index, value in enumerate(reward_history)]
            )
            expected_value += probability*float(np.sum(discounted_rewards))
            absolute_returns = np.cumsum(discounted_rewards[::-1])[::-1]
            estimator = np.zeros_like(logits)
            for index, (visited_state, profile_index) in enumerate(
                zip(state_history, profile_history, strict=True)
            ):
                profile = profiles[profile_index]
                for agent in range(agents):
                    score = -policies[agent, visited_state].copy()
                    score[profile[agent]] += 1.0
                    estimator[agent, visited_state] += absolute_returns[index]*score
            expected_gradient += probability*estimator
            return
        for profile_index in range(len(profiles)):
            action_probability = joint[state, profile_index]
            if action_probability == 0.0:
                continue
            immediate_reward = float(reward[state, profile_index])
            for next_state in range(states):
                next_probability = transition[state, profile_index, next_state]
                if next_probability == 0.0:
                    continue
                visit(
                    time+1,
                    next_state,
                    probability*action_probability*next_probability,
                    state_history+[state],
                    profile_history+[profile_index],
                    reward_history+[immediate_reward],
                )

    for initial_state in range(states):
        if start[initial_state] > 0.0:
            visit(0, initial_state, float(start[initial_state]), [], [], [])
    return expected_value, expected_gradient


def truncation_gradient_bias_bound(
    horizon: int,
    discount: float,
    reward_bound: float,
    score_norm_bound: float,
) -> float:
    """Worst-case norm gap between infinite and truncated score gradients."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < discount < 1.0:
        raise ValueError("discount must lie strictly between zero and one")
    if reward_bound < 0.0 or score_norm_bound < 0.0:
        raise ValueError("bounds must be nonnegative")
    tail = discount**horizon
    return float(
        score_norm_bound
        *reward_bound
        *tail
        *(horizon/(1.0-discount)+1.0/(1.0-discount)**2)
    )


def reinforce_packet_norm_bound(
    horizon: int,
    discount: float,
    reward_bound: float,
    score_norm_bound: float,
) -> float:
    """Pathwise norm bound for one truncated REINFORCE block packet."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < discount < 1.0:
        raise ValueError("discount must lie strictly between zero and one")
    if reward_bound < 0.0 or score_norm_bound < 0.0:
        raise ValueError("bounds must be nonnegative")
    discounted_sum = (1.0-discount**horizon)/(1.0-discount)
    return float(
        score_norm_bound
        *reward_bound
        *(discounted_sum-horizon*discount**horizon)
        /(1.0-discount)
    )
