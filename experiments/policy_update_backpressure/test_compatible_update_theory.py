from __future__ import annotations

import itertools

import numpy as np
import pytest

from .compatible_update_theory import (
    conflict_graph,
    exact_quadratic_queue_drift,
    greedy_maximal_weight_independent_set,
    is_independent,
    joint_gain_lower_bound,
    path_max_weight_independent_set,
    queue_update,
)


def test_independent_updates_have_additive_gain() -> None:
    signals = np.asarray([1.2, 0.9, 1.1, 0.8])
    radii = np.asarray([0.1, 0.1, 0.15, 0.05])
    steps = np.asarray([0.4, 0.0, 0.5, 0.0])
    matrix = np.diag([1.0, 1.2, 0.8, 1.1])
    matrix[0, 1] = matrix[1, 0] = 0.7
    matrix[1, 2] = matrix[2, 1] = 0.6
    matrix[2, 3] = matrix[3, 2] = 0.5
    joint = joint_gain_lower_bound(signals, radii, steps, matrix)
    separate = 0.0
    for index in (0, 2):
        local_steps = np.zeros(4)
        local_steps[index] = steps[index]
        separate += joint_gain_lower_bound(signals, radii, local_steps, matrix)
    assert joint == pytest.approx(separate)
    adjacency = conflict_graph(matrix)
    assert is_independent((0, 2), adjacency)


def test_conflicting_individually_good_updates_can_be_jointly_harmful() -> None:
    signals = np.ones(2)
    radii = np.zeros(2)
    matrix = np.asarray([[1.0, 4.0], [4.0, 1.0]])
    first = joint_gain_lower_bound(signals, radii, np.asarray([0.5, 0.0]), matrix)
    second = joint_gain_lower_bound(signals, radii, np.asarray([0.0, 0.5]), matrix)
    together = joint_gain_lower_bound(signals, radii, np.asarray([0.5, 0.5]), matrix)
    assert first > 0 and second > 0
    assert together < 0


def test_path_dynamic_program_matches_brute_force() -> None:
    rng = np.random.default_rng(20260901)
    for n in range(1, 11):
        for _ in range(40):
            weights = rng.uniform(0.0, 3.0, size=n)
            ready = rng.random(n) > 0.25
            selected = path_max_weight_independent_set(weights, ready)
            assert all(ready[index] for index in selected)
            assert all(b-a > 1 for a, b in zip(selected, selected[1:]))
            best = 0.0
            for mask in itertools.product((False, True), repeat=n):
                nodes = [i for i, flag in enumerate(mask) if flag]
                if any(not ready[i] for i in nodes):
                    continue
                if any(b-a == 1 for a, b in zip(nodes, nodes[1:])):
                    continue
                best = max(best, float(sum(weights[i] for i in nodes)))
            assert sum(weights[i] for i in selected) == pytest.approx(best)


def test_greedy_output_is_ready_independent_and_maximal() -> None:
    matrix = np.zeros((6, 6))
    for left, right in ((0, 1), (1, 2), (1, 4), (3, 4), (4, 5)):
        matrix[left, right] = matrix[right, left] = 1.0
    adjacency = conflict_graph(matrix)
    ready = np.asarray([True, True, True, True, False, True])
    chosen = greedy_maximal_weight_independent_set(
        np.asarray([1.0, 3.0, 2.0, 2.5, 9.0, 1.5]), adjacency, ready
    )
    assert is_independent(chosen, adjacency)
    assert all(ready[index] for index in chosen)
    for node in np.flatnonzero(ready):
        if node not in chosen:
            assert adjacency[node] & set(chosen)


def test_queue_drift_matches_explicit_update() -> None:
    queues = np.asarray([2.0, 0.0, 3.0, 1.0])
    arrivals = np.asarray([0.0, 1.0, 0.5, 0.0])
    selected = (0, 3)
    updated = queue_update(queues, arrivals, selected)
    assert np.allclose(updated, [1.0, 1.0, 3.5, 0.0])
    expected = 0.5*(updated@updated-queues@queues)
    assert exact_quadratic_queue_drift(queues, arrivals, selected) == pytest.approx(expected)
