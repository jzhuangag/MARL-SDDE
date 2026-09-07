"""Joint refresh and packet-weight minimization for factored async MARL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping


@dataclass(frozen=True)
class JointFactorChoice:
    action: Hashable
    packet_weight: float
    index: float
    effective_linear_gain: float
    quadratic_curvature: float
    reset_benefit: float
    communication_cost: float


def projected_quadratic_minimizer(
    *, linear_gain: float, quadratic_curvature: float, maximum_weight: float
) -> tuple[float, float]:
    """Minimize ``-gain * alpha + curvature * alpha^2 / 2`` on an interval."""

    if quadratic_curvature <= 0.0 or maximum_weight < 0.0:
        raise ValueError("curvature must be positive and maximum weight nonnegative")
    alpha = min(max(float(linear_gain) / float(quadratic_curvature), 0.0), maximum_weight)
    value = -float(linear_gain) * alpha + 0.5 * float(quadratic_curvature) * alpha**2
    return float(alpha), float(value)


def choose_joint_factor_action(
    *,
    alignment_lower_by_action: Mapping[Hashable, float],
    reset_benefit_by_action: Mapping[Hashable, float],
    communication_cost_by_action: Mapping[Hashable, float],
    communication_queue: float,
    learning_weight: float,
    gradient_norm_upper: float,
    learning_smoothness: float,
    receipt_motion_upper: float,
    receipt_cache_linear_upper: float,
    outgoing_cache_weight: float,
    maximum_packet_weight: float,
) -> JointFactorChoice:
    """Exactly minimize the certified local mixed discrete--continuous index.

    The action-specific lower alignment is the only learned quantity. All
    remaining terms are predictable upper bounds. The candidate set can be a
    null action plus bounded-degree one-edge refreshes.
    """

    actions = set(alignment_lower_by_action)
    if not actions or actions != set(reset_benefit_by_action) or actions != set(
        communication_cost_by_action
    ):
        raise ValueError("all nonempty action mappings must have the same keys")
    nonnegative = (
        communication_queue,
        learning_weight,
        gradient_norm_upper,
        learning_smoothness,
        receipt_motion_upper,
        receipt_cache_linear_upper,
        outgoing_cache_weight,
        maximum_packet_weight,
        *reset_benefit_by_action.values(),
        *communication_cost_by_action.values(),
    )
    if min(nonnegative) < 0.0 or learning_weight <= 0.0 or learning_smoothness <= 0.0:
        raise ValueError("bounds are nonnegative and learning terms strictly positive")

    curvature = gradient_norm_upper**2 * (
        learning_weight * learning_smoothness + outgoing_cache_weight
    )
    if curvature <= 0.0:
        raise ValueError("positive curvature requires a positive gradient bound")

    rows: list[JointFactorChoice] = []
    for action in actions:
        effective_gain = (
            learning_weight
            * (
                float(alignment_lower_by_action[action])
                - learning_smoothness
                * gradient_norm_upper
                * receipt_motion_upper
            )
            - gradient_norm_upper * receipt_cache_linear_upper
        )
        packet_weight, packet_value = projected_quadratic_minimizer(
            linear_gain=effective_gain,
            quadratic_curvature=curvature,
            maximum_weight=maximum_packet_weight,
        )
        reset = float(reset_benefit_by_action[action])
        cost = float(communication_cost_by_action[action])
        index = packet_value - reset + communication_queue * cost
        rows.append(
            JointFactorChoice(
                action=action,
                packet_weight=packet_weight,
                index=float(index),
                effective_linear_gain=float(effective_gain),
                quadratic_curvature=float(curvature),
                reset_benefit=reset,
                communication_cost=cost,
            )
        )
    return min(rows, key=lambda row: (row.index, str(row.action)))


def joint_queue_cap(
    *,
    maximum_packet_gain: float,
    minimum_quadratic_curvature: float,
    maximum_reset_benefit: float,
    minimum_positive_cost: float,
    queue_step: float,
    maximum_cost: float,
    average_budget: float,
) -> float:
    """Pathwise queue cap when the zero-cost, zero-weight null is feasible."""

    if min(
        maximum_packet_gain,
        maximum_reset_benefit,
        queue_step,
        maximum_cost,
        average_budget,
    ) < 0.0:
        raise ValueError("queue-cap inputs must be nonnegative")
    if minimum_quadratic_curvature <= 0.0 or minimum_positive_cost <= 0.0:
        raise ValueError("curvature and minimum positive cost must be positive")
    if maximum_cost < minimum_positive_cost:
        raise ValueError("maximum cost must dominate minimum positive cost")
    maximum_action_advantage = (
        maximum_packet_gain**2 / (2.0 * minimum_quadratic_curvature)
        + maximum_reset_benefit
    )
    threshold = maximum_action_advantage / minimum_positive_cost
    maximum_increment = queue_step * max(maximum_cost - average_budget, 0.0)
    return float(threshold + maximum_increment)
