"""Gaussian Markov minimax lower bound with predictable dual-budget actions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class ProbeAction:
    q: int
    stride: int

    def __post_init__(self) -> None:
        if self.q < 1 or self.stride < 1:
            raise ValueError("q and stride must be positive")


def posterior_covariance(
    *,
    actions: Sequence[ProbeAction],
    prior_variance: float,
    common_variance: float,
    private_variance: float,
    markov_lambda: float,
) -> np.ndarray:
    """Posterior covariance of (theta, current common factor).

    The normalized common-direction observation is
    Y_t = sqrt(q_t) (theta + C_t) + epsilon_t.
    """

    if prior_variance <= 0.0 or common_variance < 0.0 or private_variance <= 0.0:
        raise ValueError("invalid variances")
    if not 0.0 <= markov_lambda < 1.0:
        raise ValueError("markov_lambda must lie in [0,1)")
    covariance = np.diag([prior_variance, common_variance]).astype(float)
    for index, action in enumerate(actions):
        if index:
            coefficient = markov_lambda**action.stride
            transition = np.diag([1.0, coefficient])
            process = np.diag(
                [0.0, (1.0 - coefficient**2) * common_variance]
            )
            covariance = transition @ covariance @ transition.T + process
        observation = math.sqrt(action.q) * np.array([1.0, 1.0])
        innovation = float(observation @ covariance @ observation + private_variance)
        gain_numerator = covariance @ observation
        covariance = covariance - np.outer(gain_numerator, gain_numerator) / innovation
        covariance = (covariance + covariance.T) / 2.0
    return covariance


def dense_fisher_information(
    *,
    actions: Sequence[ProbeAction],
    common_variance: float,
    private_variance: float,
    markov_lambda: float,
) -> float:
    """Fisher information for theta under a fixed irregular action sequence."""

    if common_variance < 0.0 or private_variance <= 0.0:
        raise ValueError("invalid variances")
    if not 0.0 <= markov_lambda < 1.0:
        raise ValueError("markov_lambda must lie in [0,1)")
    if not actions:
        return 0.0
    times = np.zeros(len(actions), dtype=int)
    for index in range(1, len(actions)):
        times[index] = times[index - 1] + actions[index].stride
    roots = np.sqrt(np.asarray([action.q for action in actions], dtype=float))
    distances = np.abs(times[:, None] - times[None, :])
    covariance = (
        common_variance
        * roots[:, None]
        * roots[None, :]
        * markov_lambda**distances
        + private_variance * np.eye(len(actions))
    )
    return float(roots @ np.linalg.solve(covariance, roots))


def fixed_sequence_minimax_risk(
    *,
    actions: Sequence[ProbeAction],
    common_variance: float,
    private_variance: float,
    markov_lambda: float,
) -> float:
    information = dense_fisher_information(
        actions=actions,
        common_variance=common_variance,
        private_variance=private_variance,
        markov_lambda=markov_lambda,
    )
    if information <= 0.0:
        return math.inf
    return float(1.0 / information)


def predictable_dual_budget_lower_bound(
    *,
    action_catalogue: Iterable[ProbeAction],
    message_budget: int,
    environment_budget: int,
    overhead: int,
    delay: int,
    prior_variance: float,
    common_variance: float,
    private_variance: float,
    markov_lambda: float,
) -> dict[str, object]:
    """Exact finite-catalogue Bayes lower bound via deterministic covariance DP.

    Enumeration is intended for theorem verification on small budgets.  The
    returned value lower-bounds every predictable randomized policy because
    every realized feasible action path has posterior variance at least the
    minimum over all feasible deterministic paths.
    """

    catalogue = tuple(sorted(set(action_catalogue)))
    if not catalogue or overhead < 0 or delay < 0:
        raise ValueError("catalogue must be nonempty; overhead/delay nonnegative")
    if message_budget < 0 or environment_budget < 0:
        raise ValueError("budgets must be nonnegative")
    environment_available = max(environment_budget - delay, 0)
    best_variance = float(prior_variance)
    best_actions: tuple[ProbeAction, ...] = ()
    evaluated_paths = 1

    def visit(
        sequence: tuple[ProbeAction, ...],
        message_used: int,
        environment_used: int,
    ) -> None:
        nonlocal best_variance, best_actions, evaluated_paths
        for action in catalogue:
            next_message = message_used + overhead + action.q
            next_environment = environment_used + action.stride
            if next_message > message_budget or next_environment > environment_available:
                continue
            next_sequence = sequence + (action,)
            evaluated_paths += 1
            covariance = posterior_covariance(
                actions=next_sequence,
                prior_variance=prior_variance,
                common_variance=common_variance,
                private_variance=private_variance,
                markov_lambda=markov_lambda,
            )
            variance = float(covariance[0, 0])
            if variance < best_variance:
                best_variance = variance
                best_actions = next_sequence
            visit(next_sequence, next_message, next_environment)

    visit((), 0, 0)
    return {
        "bayes_risk_lower_bound": best_variance,
        "best_action_sequence": best_actions,
        "evaluated_paths": evaluated_paths,
        "environment_available_after_delay": environment_available,
    }


def predictable_dual_budget_minimax_risk(
    *,
    action_catalogue: Iterable[ProbeAction],
    message_budget: int,
    environment_budget: int,
    overhead: int,
    delay: int,
    common_variance: float,
    private_variance: float,
    markov_lambda: float,
) -> dict[str, object]:
    """Exact minimax value over all predictable policies in the subclass."""

    catalogue = tuple(sorted(set(action_catalogue)))
    if not catalogue or overhead < 0 or delay < 0:
        raise ValueError("catalogue must be nonempty; overhead/delay nonnegative")
    if message_budget < 0 or environment_budget < 0:
        raise ValueError("budgets must be nonnegative")
    environment_available = max(environment_budget - delay, 0)
    best_information = 0.0
    best_actions: tuple[ProbeAction, ...] = ()
    evaluated_paths = 1

    def visit(
        sequence: tuple[ProbeAction, ...],
        message_used: int,
        environment_used: int,
    ) -> None:
        nonlocal best_information, best_actions, evaluated_paths
        for action in catalogue:
            next_message = message_used + overhead + action.q
            next_environment = environment_used + action.stride
            if next_message > message_budget or next_environment > environment_available:
                continue
            next_sequence = sequence + (action,)
            evaluated_paths += 1
            information = dense_fisher_information(
                actions=next_sequence,
                common_variance=common_variance,
                private_variance=private_variance,
                markov_lambda=markov_lambda,
            )
            if information > best_information:
                best_information = information
                best_actions = next_sequence
            visit(next_sequence, next_message, next_environment)

    visit((), 0, 0)
    return {
        "minimax_risk": math.inf if best_information == 0.0 else 1.0 / best_information,
        "maximum_fisher_information": best_information,
        "best_action_sequence": best_actions,
        "evaluated_paths": evaluated_paths,
        "environment_available_after_delay": environment_available,
    }
