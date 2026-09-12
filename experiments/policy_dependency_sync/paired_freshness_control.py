"""Causal two-stage Lyapunov controls for policy-cache freshness.

The launch rule values the cache-state transition under a communication queue.
The receipt rule applies the scalar smoothness minimizer to the packet that the
selected cached behavior actually generated.  Counterfactual packet alignment
is deliberately not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Hashable, Mapping

from .joint_factor_lyapunov import projected_quadratic_minimizer


@dataclass(frozen=True)
class LaunchChoice:
    action: Hashable
    utility_upper: float
    reset_benefit: float
    communication_cost: float
    index: float


@dataclass(frozen=True)
class ReceiptChoice:
    packet_weight: float
    certified_alignment: float
    effective_linear_gain: float
    quadratic_curvature: float
    drift_upper: float


def choose_launch_cache_action(
    *,
    utility_upper_by_action: Mapping[Hashable, float],
    communication_cost_by_action: Mapping[Hashable, float],
    communication_queue: float,
    utility_weight: float,
    feasible_by_action: Mapping[Hashable, bool] | None = None,
    reset_benefit_by_action: Mapping[Hashable, float] | None = None,
) -> LaunchChoice:
    """Minimize ``-V U^U(a) - B(a) + Q c(a)`` on the feasible set."""

    actions = set(utility_upper_by_action)
    if not actions or actions != set(communication_cost_by_action):
        raise ValueError("utility and cost mappings must share nonempty keys")
    feasible = (
        {action: True for action in actions}
        if feasible_by_action is None
        else dict(feasible_by_action)
    )
    reset = (
        {action: 0.0 for action in actions}
        if reset_benefit_by_action is None
        else dict(reset_benefit_by_action)
    )
    if actions != set(feasible) or not any(feasible.values()):
        raise ValueError("feasibility must cover all actions and retain one action")
    if actions != set(reset):
        raise ValueError("reset benefits must cover every action")
    if communication_queue < 0.0 or utility_weight <= 0.0:
        raise ValueError("queue is nonnegative and utility weight is positive")
    if any(
        not isfinite(float(value))
        for value in (
            *utility_upper_by_action.values(),
            *communication_cost_by_action.values(),
            *reset.values(),
        )
    ):
        raise ValueError("utilities and costs must be finite")
    if min(communication_cost_by_action.values()) < 0.0:
        raise ValueError("communication costs must be nonnegative")
    if min(reset.values()) < 0.0:
        raise ValueError("cache reset benefits must be nonnegative")

    rows = [
        LaunchChoice(
            action=action,
            utility_upper=float(utility_upper_by_action[action]),
            reset_benefit=float(reset[action]),
            communication_cost=float(communication_cost_by_action[action]),
            index=float(
                -utility_weight * float(utility_upper_by_action[action])
                - float(reset[action])
                + communication_queue * float(communication_cost_by_action[action])
            ),
        )
        for action in actions
        if feasible[action]
    ]
    return min(
        rows,
        key=lambda row: (
            row.index,
            row.communication_cost,
            str(row.action),
        ),
    )


def choose_receipt_packet_weight(
    *,
    observed_launch_alignment: float,
    packet_norm_upper: float,
    reference_error_upper: float,
    learning_smoothness: float,
    launch_to_receipt_motion_upper: float,
    maximum_packet_weight: float,
    learning_weight: float = 1.0,
    cache_gradient_alignment: float = 0.0,
    outgoing_cache_weight: float = 0.0,
) -> ReceiptChoice:
    """Minimize the exact fixed-universe composite receipt bound.

    ``cache_gradient_alignment`` is ``<G, sum beta(theta-chi)>`` at receipt.
    It is observed from current parameters/caches rather than predicted at
    launch.  Set both cache arguments to zero for the topology-robust core.
    """

    values = (
        observed_launch_alignment,
        packet_norm_upper,
        reference_error_upper,
        learning_smoothness,
        launch_to_receipt_motion_upper,
        maximum_packet_weight,
        learning_weight,
        cache_gradient_alignment,
        outgoing_cache_weight,
    )
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("receipt inputs must be finite")
    if (
        packet_norm_upper <= 0.0
        or reference_error_upper < 0.0
        or learning_smoothness <= 0.0
        or launch_to_receipt_motion_upper < 0.0
        or maximum_packet_weight < 0.0
        or learning_weight <= 0.0
        or outgoing_cache_weight < 0.0
    ):
        raise ValueError("invalid receipt-time bounds")

    certified = float(
        observed_launch_alignment
        - packet_norm_upper
        * (
            reference_error_upper
            + learning_smoothness * launch_to_receipt_motion_upper
        )
    )
    effective_gain = float(
        learning_weight * certified + cache_gradient_alignment
    )
    curvature = float(
        packet_norm_upper**2
        * (learning_weight * learning_smoothness + outgoing_cache_weight)
    )
    weight, drift = projected_quadratic_minimizer(
        linear_gain=effective_gain,
        quadratic_curvature=curvature,
        maximum_weight=maximum_packet_weight,
    )
    return ReceiptChoice(
        packet_weight=weight,
        certified_alignment=certified,
        effective_linear_gain=effective_gain,
        quadratic_curvature=curvature,
        drift_upper=drift,
    )


def communication_queue_update(
    *,
    queue: float,
    queue_step: float,
    realized_cost: float,
    average_budget: float,
) -> float:
    """One reflected virtual-queue update with the theorem's scaling."""

    values = (queue, queue_step, realized_cost, average_budget)
    if any(not isfinite(float(value)) for value in values) or min(values) < 0.0:
        raise ValueError("queue inputs must be finite and nonnegative")
    if queue_step <= 0.0:
        raise ValueError("queue step must be positive")
    return float(max(0.0, queue + queue_step * (realized_cost - average_budget)))
