"""Lyapunov-optimal innovation gains and paid finite-budget allocation.

The Gaussian constant-feature model permits a statistically efficient critic,
which must be compared with efficient fixed-q learners, not constant-step-only
baselines. This module is separate from the frozen CPU-001 implementation.
"""

from dataclasses import dataclass

import numpy as np

from learning_value import Calibration, calibration_cost


@dataclass(frozen=True)
class InnovationAction:
    q: int
    spacing: int
    updates: int
    information_factor: float
    messages: int
    actor_transitions: int


def information_factor(observations: int, retained_persistence: float) -> float:
    if observations < 1 or not 0 <= retained_persistence < 1:
        raise ValueError("invalid sample horizon or persistence")
    r = retained_persistence
    return (observations*(1-r)+2*r)/(1+r)


def innovation_gain(index: int, retained_persistence: float) -> float:
    """Minimizes the exact quadratic Lyapunov drift at each new innovation.

    The first observation initializes the unbiased estimate; subsequent indices
    start at two. No fitted target or hand-selected learning rate is required.
    """
    if index < 1 or not 0 <= retained_persistence < 1:
        raise ValueError("invalid update index or persistence")
    if index == 1:
        return 1.0
    precision_before = information_factor(index-1, retained_persistence)
    innovation_precision = (1-retained_persistence)/(1+retained_persistence)
    return innovation_precision/(precision_before+innovation_precision)


def efficient_actions(config: dict, cell: dict, *, paid: bool) -> list[InnovationAction]:
    pm, pe, _ = calibration_cost(config["calibration_workers"], config["calibration_innovations"], cell["overhead"]) if paid else (0,0,0)
    result = []
    for q in config["q_catalogue"]:
        for b in config["spacing_catalogue"]:
            n = min((config["message_budget"]-pm)//(cell["overhead"]+q),
                    (cell["actor_budget"]-pe)//(q*b))
            if n >= 1:
                result.append(InnovationAction(q, b, n,
                              information_factor(n, cell["persistence"]**b),
                              pm+n*(cell["overhead"]+q), pe+n*q*b))
    return result


def efficient_forecast(action: InnovationAction, calibration: Calibration) -> tuple[float,float,float]:
    return tuple(value/action.information_factor for value in calibration.variance(action.q))


def efficient_select(actions: list[InnovationAction], calibration: Calibration) -> InnovationAction:
    return min(actions, key=lambda a: (efficient_forecast(a,calibration)[0],a.q,a.spacing))


def efficient_true_risk(action: InnovationAction, rho: float) -> float:
    return (rho+(1-rho)/action.q)/action.information_factor


def efficient_estimate(rewards: np.ndarray, retained_persistence: float) -> np.ndarray:
    """Return the full causal GLS estimate trajectory along the leading axis."""
    rewards = np.asarray(rewards, dtype=float)
    if rewards.shape[0] == 0:
        raise ValueError("no observations")
    output = np.empty_like(rewards)
    output[0] = rewards[0]
    r = retained_persistence
    for k in range(1, len(rewards)):
        innovation_observation = (rewards[k]-r*rewards[k-1])/(1-r)
        gain = innovation_gain(k+1, r)
        output[k] = output[k-1]+gain*(innovation_observation-output[k-1])
    return output
