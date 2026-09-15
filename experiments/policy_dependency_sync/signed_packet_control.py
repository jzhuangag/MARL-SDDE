"""Causal signed-drift control for asynchronous policy-version packets.

The functions are deterministic theorem components.  Statistical routines
must supply predictable packet-mean and launch-to-receipt motion bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np


Action = Hashable


@dataclass(frozen=True)
class SignedLaunchChoice:
    action: Action
    predicted_drift: float
    resource_cost: float
    index: float


def smooth_packet_drift_score(
    launch_gradient: np.ndarray,
    packet_mean: np.ndarray,
    packet_second_moment: float,
    smoothness: float,
    step: float,
) -> float:
    """Smoothness upper bound evaluated with the launch-time gradient."""

    gradient = np.asarray(launch_gradient, dtype=float)
    mean = np.asarray(packet_mean, dtype=float)
    if gradient.shape != mean.shape:
        raise ValueError("launch gradient and packet mean must have equal shape")
    if packet_second_moment < float(mean @ mean) - 1e-12:
        raise ValueError("second moment cannot be below squared mean norm")
    if smoothness <= 0.0 or step < 0.0:
        raise ValueError("smoothness must be positive and step nonnegative")
    return float(
        -step * (gradient @ mean)
        + 0.5 * smoothness * step * step * packet_second_moment
    )


def launch_to_receipt_error_bound(
    step: float,
    receipt_gradient_motion_bound: float,
    packet_mean_norm_bound: float,
) -> float:
    """Error from replacing the receipt gradient by its launch value."""

    if min(step, receipt_gradient_motion_bound, packet_mean_norm_bound) < 0.0:
        raise ValueError("motion-bound inputs must be nonnegative")
    return float(step * receipt_gradient_motion_bound * packet_mean_norm_bound)


def choose_one_edge_signed(
    predicted_drift_by_action: Mapping[Action, float],
    resource_cost_by_action: Mapping[Action, float],
    queue: float,
    learning_weight: float,
) -> SignedLaunchChoice:
    """Exactly minimize the one-edge packetized Lyapunov DPP index.

    The action dictionary contains the null action and at most one refresh per
    candidate edge.  Forming and scanning it is linear in local degree.
    """

    if not predicted_drift_by_action:
        raise ValueError("at least one launch action is required")
    if set(predicted_drift_by_action) != set(resource_cost_by_action):
        raise ValueError("drift and cost actions must match")
    if queue < 0.0 or learning_weight <= 0.0:
        raise ValueError("queue must be nonnegative and weight positive")
    if min(resource_cost_by_action.values()) < 0.0:
        raise ValueError("resource costs must be nonnegative")
    action = min(
        predicted_drift_by_action,
        key=lambda item: (
            learning_weight * predicted_drift_by_action[item]
            + queue * resource_cost_by_action[item],
            repr(item),
        ),
    )
    drift = float(predicted_drift_by_action[action])
    cost = float(resource_cost_by_action[action])
    return SignedLaunchChoice(
        action=action,
        predicted_drift=drift,
        resource_cost=cost,
        index=float(learning_weight * drift + queue * cost),
    )


def signed_selection_regret_bound(
    true_drift_by_action: Mapping[Action, float],
    estimated_drift_by_action: Mapping[Action, float],
    resource_cost_by_action: Mapping[Action, float],
    queue: float,
    learning_weight: float,
) -> tuple[float, float]:
    """True DPP regret and the standard twice-uniform-error bound."""

    if not (
        set(true_drift_by_action)
        == set(estimated_drift_by_action)
        == set(resource_cost_by_action)
    ):
        raise ValueError("true, estimated, and cost actions must match")
    choice = choose_one_edge_signed(
        estimated_drift_by_action,
        resource_cost_by_action,
        queue,
        learning_weight,
    )
    indices = {
        action: learning_weight * true_drift_by_action[action]
        + queue * resource_cost_by_action[action]
        for action in true_drift_by_action
    }
    regret = float(indices[choice.action] - min(indices.values()))
    uniform_error = max(
        abs(estimated_drift_by_action[action] - true_drift_by_action[action])
        for action in true_drift_by_action
    )
    return regret, float(2.0 * learning_weight * uniform_error)


def paired_stationarity_upper(
    initial_suboptimality: float,
    initial_queue: float,
    launches: int,
    learning_weight: float,
    descent_mass: float,
    queue_drift_constant: float,
    comparator_remainder_sum: float,
    score_error_sum: float,
    terminal_inflight_boundary: float = 0.0,
) -> float:
    """Average stationarity bound after draining all launched packets.

    ``descent_mass`` is the coefficient multiplying each squared launch-time
    gradient in the comparator drift bound.  The terminal boundary is zero
    after a drain and otherwise explicitly prices unmatched in-flight packets.
    """

    values = (
        initial_suboptimality,
        initial_queue,
        queue_drift_constant,
        comparator_remainder_sum,
        score_error_sum,
        terminal_inflight_boundary,
    )
    if min(values) < 0.0:
        raise ValueError("stationarity-bound inputs must be nonnegative")
    if launches <= 0 or learning_weight <= 0.0 or descent_mass <= 0.0:
        raise ValueError("launches, weight, and descent mass must be positive")
    numerator = initial_suboptimality
    numerator += 0.5 * initial_queue * initial_queue / learning_weight
    numerator += launches * queue_drift_constant / learning_weight
    numerator += comparator_remainder_sum + 2.0 * score_error_sum
    numerator += terminal_inflight_boundary
    return float(numerator / (descent_mass * launches))


def favorable_phase_margin(
    gradient_norm: float,
    packet_bias_bound: float,
    packet_variance: float,
    receipt_gradient_motion_bound: float,
    smoothness: float,
    step: float,
    descent_fraction: float,
) -> float:
    """Sufficient signed-drift margin for one comparator packet.

    A nonnegative return certifies
    ``D <= -descent_fraction * step * ||gradient||^2`` when the packet mean is
    within ``packet_bias_bound`` of the launch gradient and the receipt
    gradient moves by at most ``receipt_gradient_motion_bound``.
    """

    values = (
        gradient_norm,
        packet_bias_bound,
        packet_variance,
        receipt_gradient_motion_bound,
        step,
        descent_fraction,
    )
    if min(values) < 0.0 or smoothness <= 0.0:
        raise ValueError("phase-bound inputs must be nonnegative")
    if descent_fraction > 1.0:
        raise ValueError("descent fraction cannot exceed one")
    candidate_norm = gradient_norm + packet_bias_bound
    adverse = step * gradient_norm * packet_bias_bound
    adverse += 0.5 * smoothness * step * step * (
        candidate_norm * candidate_norm + packet_variance
    )
    adverse += step * receipt_gradient_motion_bound * candidate_norm
    available = (1.0 - descent_fraction) * step * gradient_norm * gradient_norm
    return float(available - adverse)
