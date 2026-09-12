"""Low-complexity decisions induced by the policy-sync Lyapunov bound."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Hashable


Edge = tuple[Hashable, Hashable]


def select_refresh_edges(
    edge_benefit: Mapping[Edge, float],
    edge_cost: Mapping[Edge, float],
    lyapunov_weight: float,
    communication_queue: float,
    max_edges: int | None = None,
) -> tuple[Edge, ...]:
    """Minimize a separable dispatch-time drift-plus-penalty upper bound.

    Refreshing edge ``e`` changes the bound by
    ``-V*edge_benefit[e] + Q*edge_cost[e]``.  With an optional cardinality
    limit, choosing the largest positive net benefits is exactly optimal.
    """

    if lyapunov_weight <= 0.0 or communication_queue < 0.0:
        raise ValueError("lyapunov_weight must be positive and queue nonnegative")
    if set(edge_benefit) != set(edge_cost):
        raise ValueError("benefit and cost must contain identical edge keys")
    scored = []
    for edge, benefit in edge_benefit.items():
        cost = float(edge_cost[edge])
        if cost < 0.0:
            raise ValueError("edge costs must be nonnegative")
        net = lyapunov_weight * float(benefit) - communication_queue * cost
        if net > 0.0:
            scored.append((net, edge))
    scored.sort(key=lambda item: (-item[0], repr(item[1])))
    if max_edges is not None:
        if max_edges < 0:
            raise ValueError("max_edges must be nonnegative")
        scored = scored[:max_edges]
    return tuple(edge for _, edge in scored)


def receipt_step(
    alignment: float,
    gradient_second_moment: float,
    curvature: float,
    maximum_step: float,
) -> float:
    """Exact minimizer of ``-alpha*A + Lambda*alpha^2*B/2`` on an interval."""

    if gradient_second_moment < 0.0 or curvature <= 0.0 or maximum_step < 0.0:
        raise ValueError("invalid quadratic-bound parameters")
    if alignment <= 0.0 or gradient_second_moment == 0.0:
        return 0.0
    return min(float(maximum_step), float(alignment) / (curvature * gradient_second_moment))


def queue_update(queue: float, charged_cost: float, target_cost: float) -> float:
    if queue < 0.0 or charged_cost < 0.0 or target_cost < 0.0:
        raise ValueError("queue and costs must be nonnegative")
    return max(0.0, float(queue) + float(charged_cost) - float(target_cost))
