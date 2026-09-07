"""Exact cache-energy identities for composite policy-freshness Lyapunov control."""

from __future__ import annotations

from typing import Mapping

import numpy as np


Vector = np.ndarray
Edge = tuple[int, int]


def cache_mismatch_energy(
    current_by_donor: Mapping[int, Vector],
    cache_by_edge: Mapping[Edge, Vector],
    weight_by_edge: Mapping[Edge, float],
) -> float:
    if set(cache_by_edge) != set(weight_by_edge):
        raise ValueError("cache and weight edges must match")
    if min(weight_by_edge.values(), default=0.0) < 0.0:
        raise ValueError("cache-energy weights must be nonnegative")
    total = 0.0
    for (donor, _recipient), cached in cache_by_edge.items():
        current = np.asarray(current_by_donor[donor], dtype=float)
        cached_array = np.asarray(cached, dtype=float)
        if current.shape != cached_array.shape:
            raise ValueError("current and cached blocks must have equal shape")
        total += 0.5 * weight_by_edge[(donor, _recipient)] * float(
            np.sum((current - cached_array) ** 2)
        )
    return float(total)


def topology_motion_increment(
    current_by_donor: Mapping[int, Vector],
    cache_by_edge: Mapping[Edge, Vector],
    old_weight_by_edge: Mapping[Edge, float],
    new_weight_by_edge: Mapping[Edge, float],
) -> float:
    """Exact cache-energy jump caused only by changing fixed-universe weights.

    A dynamic active set is represented by zero weight on inactive edges in
    one fixed edge universe.  Policies and caches are held fixed in this
    identity; refresh and receipt motion must be accounted separately.
    """

    universe = set(cache_by_edge)
    if universe != set(old_weight_by_edge) or universe != set(new_weight_by_edge):
        raise ValueError("cache and both weight mappings must share one edge universe")
    if min(old_weight_by_edge.values(), default=0.0) < 0.0 or min(
        new_weight_by_edge.values(), default=0.0
    ) < 0.0:
        raise ValueError("cache-energy weights must be nonnegative")
    total = 0.0
    for edge, cached in cache_by_edge.items():
        donor, _recipient = edge
        current = np.asarray(current_by_donor[donor], dtype=float)
        cached_array = np.asarray(cached, dtype=float)
        if current.shape != cached_array.shape:
            raise ValueError("current and cached blocks must have equal shape")
        weight_change = float(new_weight_by_edge[edge] - old_weight_by_edge[edge])
        total += 0.5 * weight_change * float(np.sum((current - cached_array) ** 2))
    return float(total)


def topology_motion_positive_upper(
    *,
    policy_cache_diameter_upper: float,
    old_weight_by_edge: Mapping[Edge, float],
    new_weight_by_edge: Mapping[Edge, float],
) -> float:
    """Upper-bound the positive topology jump on a fixed edge universe."""

    if set(old_weight_by_edge) != set(new_weight_by_edge):
        raise ValueError("both weight mappings must share one edge universe")
    if policy_cache_diameter_upper < 0.0:
        raise ValueError("the policy-cache diameter bound must be nonnegative")
    if min(old_weight_by_edge.values(), default=0.0) < 0.0 or min(
        new_weight_by_edge.values(), default=0.0
    ) < 0.0:
        raise ValueError("cache-energy weights must be nonnegative")
    positive_weight_motion = sum(
        max(float(new_weight_by_edge[edge] - old_weight_by_edge[edge]), 0.0)
        for edge in old_weight_by_edge
    )
    return float(0.5 * policy_cache_diameter_upper**2 * positive_weight_motion)


def refresh_reset_benefit(
    current: Vector, cached: Vector, weight: float
) -> float:
    if weight < 0.0:
        raise ValueError("weight must be nonnegative")
    current_array = np.asarray(current, dtype=float)
    cached_array = np.asarray(cached, dtype=float)
    if current_array.shape != cached_array.shape:
        raise ValueError("current and cached blocks must have equal shape")
    delta = current_array - cached_array
    return float(0.5 * weight * np.sum(delta**2))


def donor_receipt_energy_increment(
    *,
    current: Vector,
    gradient: Vector,
    step: float,
    cache_by_recipient: Mapping[int, Vector],
    weight_by_recipient: Mapping[int, float],
) -> float:
    """Exact full outgoing-cache energy change under theta <- theta-step*g."""

    if set(cache_by_recipient) != set(weight_by_recipient):
        raise ValueError("recipient cache and weight keys must match")
    if step < 0.0 or min(weight_by_recipient.values(), default=0.0) < 0.0:
        raise ValueError("step and weights must be nonnegative")
    current_array = np.asarray(current, dtype=float)
    gradient_array = np.asarray(gradient, dtype=float)
    if current_array.shape != gradient_array.shape:
        raise ValueError("current block and gradient must have equal shape")
    weighted_displacement = np.zeros_like(current_array)
    total_weight = 0.0
    for recipient, cached in cache_by_recipient.items():
        cached_array = np.asarray(cached, dtype=float)
        if cached_array.shape != current_array.shape:
            raise ValueError("cached blocks must match current block shape")
        weight = float(weight_by_recipient[recipient])
        weighted_displacement += weight * (current_array - cached_array)
        total_weight += weight
    return float(
        -step * np.dot(gradient_array.ravel(), weighted_displacement.ravel())
        + 0.5 * step * step * total_weight * np.sum(gradient_array**2)
    )


def donor_receipt_increment_upper(
    *,
    step: float,
    gradient_norm_upper: float,
    weighted_cache_displacement_norm: float,
    total_outgoing_weight: float,
) -> float:
    values = (
        step,
        gradient_norm_upper,
        weighted_cache_displacement_norm,
        total_outgoing_weight,
    )
    if min(values) < 0.0:
        raise ValueError("receipt-bound inputs must be nonnegative")
    return float(
        step * gradient_norm_upper * weighted_cache_displacement_norm
        + 0.5
        * step
        * step
        * total_outgoing_weight
        * gradient_norm_upper
        * gradient_norm_upper
    )


def stale_edge_activation_margin(
    *,
    reset_benefit: float,
    learning_drift_disadvantage: float,
    learning_weight: float,
    communication_queue: float,
    message_cost: float,
) -> float:
    """Positive means the edge strictly beats null in the composite index."""

    values = (
        reset_benefit,
        learning_drift_disadvantage,
        learning_weight,
        communication_queue,
        message_cost,
    )
    if min(values) < 0.0:
        raise ValueError("activation-margin inputs must be nonnegative")
    return float(
        reset_benefit
        - learning_weight * learning_drift_disadvantage
        - communication_queue * message_cost
    )
