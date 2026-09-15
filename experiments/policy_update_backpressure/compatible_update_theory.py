"""Outcome-free algebra for compatible asynchronous MARL policy updates."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


Array = np.ndarray


def joint_gain_lower_bound(
    signals: Array,
    radii: Array,
    steps: Array,
    cross_smoothness: Array,
) -> float:
    """Block-smooth lower bound including pairwise update interference."""

    signals = np.asarray(signals, dtype=float)
    radii = np.asarray(radii, dtype=float)
    steps = np.asarray(steps, dtype=float)
    matrix = np.asarray(cross_smoothness, dtype=float)
    n = signals.size
    if radii.shape != (n,) or steps.shape != (n,) or matrix.shape != (n, n):
        raise ValueError("incompatible block-gain shapes")
    if min(signals.min(initial=0), radii.min(initial=0), steps.min(initial=0)) < 0:
        raise ValueError("signals, radii and steps must be nonnegative")
    if (matrix < 0).any() or not np.allclose(matrix, matrix.T):
        raise ValueError("cross-smoothness matrix must be symmetric nonnegative")
    linear = np.sum(steps*(signals*signals-radii*signals))
    scaled = steps*signals
    curvature = 0.5*float(scaled@matrix@scaled)
    return float(linear-curvature)


def conflict_graph(cross_smoothness: Array, *, threshold: float = 0.0) -> tuple[frozenset[int], ...]:
    matrix = np.asarray(cross_smoothness, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("conflict graph needs a square matrix")
    if threshold < 0 or (matrix < 0).any():
        raise ValueError("invalid conflict threshold or matrix")
    return tuple(
        frozenset(
            j for j in range(matrix.shape[0])
            if i != j and max(matrix[i, j], matrix[j, i]) > threshold
        )
        for i in range(matrix.shape[0])
    )


def is_independent(nodes: Iterable[int], adjacency: tuple[frozenset[int], ...]) -> bool:
    chosen = frozenset(int(node) for node in nodes)
    if any(node < 0 or node >= len(adjacency) for node in chosen):
        raise ValueError("node outside graph")
    return all(not (adjacency[node] & chosen) for node in chosen)


def path_max_weight_independent_set(weights: Array, ready: Array | None = None) -> tuple[int, ...]:
    """Exact maximum-weight independent ready set on a path in O(n)."""

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or (weights < 0).any() or not np.isfinite(weights).all():
        raise ValueError("path weights must be finite and nonnegative")
    if ready is None:
        available = np.ones(weights.size, dtype=bool)
    else:
        available = np.asarray(ready, dtype=bool)
        if available.shape != weights.shape:
            raise ValueError("ready mask shape mismatch")
    n = weights.size
    if n == 0:
        return ()
    value = np.zeros(n+1, dtype=float)
    take = np.zeros(n+1, dtype=bool)
    for length in range(1, n+1):
        skip_value = value[length-1]
        take_value = -np.inf
        if available[length-1]:
            take_value = weights[length-1]+(value[length-2] if length >= 2 else 0.0)
        if take_value > skip_value+1e-15:
            value[length] = take_value
            take[length] = True
        else:
            value[length] = skip_value
    selected: list[int] = []
    length = n
    while length > 0:
        if take[length]:
            selected.append(length-1)
            length -= 2
        else:
            length -= 1
    return tuple(reversed(selected))


def greedy_maximal_weight_independent_set(
    weights: Array,
    adjacency: tuple[frozenset[int], ...],
    ready: Array | None = None,
) -> tuple[int, ...]:
    """Deterministic descending-weight maximal independent set."""

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(adjacency),) or (weights < 0).any():
        raise ValueError("weight/graph mismatch")
    available = (
        np.ones(weights.size, dtype=bool)
        if ready is None else np.asarray(ready, dtype=bool)
    )
    if available.shape != weights.shape:
        raise ValueError("ready mask shape mismatch")
    order = sorted(
        (index for index in range(weights.size) if available[index]),
        key=lambda index: (-weights[index], index),
    )
    selected: list[int] = []
    blocked: set[int] = set()
    for node in order:
        if node in blocked:
            continue
        selected.append(node)
        blocked.add(node)
        blocked.update(adjacency[node])
    return tuple(sorted(selected))


def queue_update(queues: Array, arrivals: Array, selected: Iterable[int]) -> Array:
    queues = np.asarray(queues, dtype=float)
    arrivals = np.asarray(arrivals, dtype=float)
    if queues.shape != arrivals.shape or (queues < 0).any() or (arrivals < 0).any():
        raise ValueError("invalid queue state")
    service = np.zeros_like(queues)
    for node in selected:
        if node < 0 or node >= queues.size:
            raise ValueError("service node outside queue vector")
        service[node] = 1.0
    return np.maximum(queues-service, 0.0)+arrivals


def exact_quadratic_queue_drift(
    queues: Array, arrivals: Array, selected: Iterable[int]
) -> float:
    updated = queue_update(queues, arrivals, selected)
    return float(0.5*(updated@updated-queues@queues))

