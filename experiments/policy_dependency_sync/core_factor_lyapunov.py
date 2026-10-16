"""Topology-robust core Lyapunov control for factored asynchronous MARL."""

from __future__ import annotations

from typing import Hashable, Mapping

from .joint_factor_lyapunov import (
    JointFactorChoice,
    joint_queue_cap,
    projected_quadratic_minimizer,
)


def choose_core_factor_action(
    *,
    alignment_lower_by_action: Mapping[Hashable, float],
    communication_cost_by_action: Mapping[Hashable, float],
    communication_queue: float,
    learning_weight: float,
    gradient_norm_upper: float,
    learning_smoothness: float,
    receipt_motion_upper: float,
    maximum_packet_weight: float,
) -> JointFactorChoice:
    """Minimize the topology-robust ``VF + Q^2/(2 nu)`` drift index."""

    return choose_core_factor_action_variable_bounds(
        alignment_lower_by_action=alignment_lower_by_action,
        communication_cost_by_action=communication_cost_by_action,
        gradient_norm_upper_by_action={
            action: gradient_norm_upper for action in alignment_lower_by_action
        },
        communication_queue=communication_queue,
        learning_weight=learning_weight,
        learning_smoothness=learning_smoothness,
        receipt_motion_upper=receipt_motion_upper,
        maximum_packet_weight=maximum_packet_weight,
    )


def choose_core_factor_action_variable_bounds(
    *,
    alignment_lower_by_action: Mapping[Hashable, float],
    communication_cost_by_action: Mapping[Hashable, float],
    gradient_norm_upper_by_action: Mapping[Hashable, float],
    communication_queue: float,
    learning_weight: float,
    learning_smoothness: float,
    receipt_motion_upper: float,
    maximum_packet_weight: float,
) -> JointFactorChoice:
    """Minimize the core edge--weight index with action-specific clipping."""

    actions = set(alignment_lower_by_action)
    mappings = (communication_cost_by_action, gradient_norm_upper_by_action)
    if not actions or any(actions != set(mapping) for mapping in mappings):
        raise ValueError("all nonempty action mappings must have the same keys")
    nonnegative = (
        communication_queue,
        learning_weight,
        learning_smoothness,
        receipt_motion_upper,
        maximum_packet_weight,
        *communication_cost_by_action.values(),
        *gradient_norm_upper_by_action.values(),
    )
    if (
        min(nonnegative) < 0.0
        or learning_weight <= 0.0
        or learning_smoothness <= 0.0
    ):
        raise ValueError("bounds are nonnegative and learning terms strictly positive")
    if min(gradient_norm_upper_by_action.values()) <= 0.0:
        raise ValueError("every action requires a positive gradient bound")

    rows: list[JointFactorChoice] = []
    for action in actions:
        gradient_bound = float(gradient_norm_upper_by_action[action])
        curvature = (
            learning_weight * learning_smoothness * gradient_bound * gradient_bound
        )
        effective_gain = learning_weight * (
            float(alignment_lower_by_action[action])
            - learning_smoothness * gradient_bound * receipt_motion_upper
        )
        packet_weight, packet_value = projected_quadratic_minimizer(
            linear_gain=effective_gain,
            quadratic_curvature=curvature,
            maximum_weight=maximum_packet_weight,
        )
        cost = float(communication_cost_by_action[action])
        rows.append(
            JointFactorChoice(
                action=action,
                packet_weight=packet_weight,
                index=float(packet_value + communication_queue * cost),
                effective_linear_gain=float(effective_gain),
                quadratic_curvature=float(curvature),
                reset_benefit=0.0,
                communication_cost=cost,
            )
        )
    return min(
        rows,
        key=lambda row: (
            row.index,
            row.communication_cost,
            row.packet_weight,
            str(row.action),
        ),
    )


def core_queue_cap(
    *,
    maximum_packet_gain: float,
    minimum_quadratic_curvature: float,
    minimum_positive_cost: float,
    queue_step: float,
    maximum_cost: float,
    average_budget: float,
) -> float:
    """Pathwise queue cap for the topology-robust core controller."""

    return joint_queue_cap(
        maximum_packet_gain=maximum_packet_gain,
        minimum_quadratic_curvature=minimum_quadratic_curvature,
        maximum_reset_benefit=0.0,
        minimum_positive_cost=minimum_positive_cost,
        queue_step=queue_step,
        maximum_cost=maximum_cost,
        average_budget=average_budget,
    )
