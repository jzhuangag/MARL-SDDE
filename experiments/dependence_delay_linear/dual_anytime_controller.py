"""Dual anytime confidence tools for unknown mixing and agent sharing."""

from typing import Dict

import numpy as np
from scipy.special import betaln

from predictable_mixing_controller import (
    SERVER_OVERHEAD,
    select_joint_action,
)


ANYTIME_ALPHA = 0.01
INITIAL_PAIR_TRIALS = 128
PAIR_PROBE_COST = SERVER_OVERHEAD + 2


def log_beta_binomial_mixture_ratio(
    probability: float,
    successes: int,
    trials: int,
    prior_shape: float = 0.5,
) -> float:
    """Return log of the beta-binomial mixture likelihood ratio."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("invalid Bernoulli counts")
    if prior_shape <= 0.0:
        raise ValueError("prior_shape must be positive")
    failures = trials - successes
    return float(
        betaln(successes + prior_shape, failures + prior_shape)
        - betaln(prior_shape, prior_shape)
        - successes * np.log(probability)
        - failures * np.log1p(-probability)
    )


def mixture_upper_confidence(
    successes: int,
    trials: int,
    alpha: float,
) -> float:
    """Return an anytime-valid beta-binomial mixture upper confidence bound."""

    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("invalid Bernoulli counts")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if successes == trials:
        return 1.0
    target = float(np.log(1.0 / alpha))
    empirical = successes / float(trials)
    lower = max(empirical, np.finfo(float).eps)
    upper = 1.0 - np.finfo(float).eps
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        value = log_beta_binomial_mixture_ratio(
            midpoint, successes, trials
        )
        if value < target:
            lower = midpoint
        else:
            upper = midpoint
    return float(0.5 * (lower + upper))


def rounded_upper(value: float, resolution: float) -> float:
    """Round a confidence upper bound upward on a deterministic grid."""

    if not 0.0 <= value <= 1.0:
        raise ValueError("value must lie in [0, 1]")
    if not 0.0 < resolution <= 1.0:
        raise ValueError("resolution must lie in (0, 1]")
    return float(min(1.0, np.ceil(value / resolution) * resolution))


def dual_confidence_bounds(
    stays: int,
    transition_trials: int,
    shared: int,
    sharing_trials: int,
    decision: int,
) -> Dict[str, float]:
    """Return simultaneously allocated one-sided bounds for p and rho."""

    if decision < 1:
        raise ValueError("decision must be positive")
    alpha_each = ANYTIME_ALPHA / 2.0
    persistence = mixture_upper_confidence(
        stays, transition_trials, alpha_each
    )
    sharing = mixture_upper_confidence(
        shared, sharing_trials, alpha_each
    )
    return {
        "alpha_each": alpha_each,
        "persistence_upper": float(persistence),
        "rho_upper": float(sharing),
        "certified_persistence": rounded_upper(persistence, 0.002),
        "certified_rho": rounded_upper(sharing, 0.02),
    }


def select_dual_action(
    persistence_upper: float,
    rho_upper: float,
    delay: int,
    resource_budget: int,
    fixed_q=None,
) -> Dict[str, float]:
    """Select the theorem-safe scalar action using both confidence bounds."""

    return select_joint_action(
        persistence_upper,
        rho_upper,
        delay,
        pilot_cost=0,
        fixed_q=fixed_q,
        resource_budget=resource_budget,
    )


def block_observation_counts(
    action: Dict[str, float],
    updates: int,
    leftover_resource: int,
) -> Dict[str, int]:
    """Count naturally observed transitions and charged pair probes."""

    probes = int(leftover_resource) // PAIR_PROBE_COST
    transition_trials = int(updates) * int(action["gap"]) + probes
    sharing_trials = (
        int(updates) if int(action["num_agents"]) >= 2 else 0
    ) + probes
    return {
        "pair_probes": int(probes),
        "transition_trials": int(transition_trials),
        "sharing_trials": int(sharing_trials),
    }
