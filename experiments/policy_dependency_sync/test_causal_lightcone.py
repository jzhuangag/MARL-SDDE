import itertools

import numpy as np

from .causal_lightcone import queue_update, receipt_step, select_refresh_edges


def test_edge_threshold_matches_brute_force_cardinality_problem():
    benefit = {("a", 0): 0.7, ("b", 0): 0.1, ("c", 0): 0.5}
    cost = {("a", 0): 0.2, ("b", 0): 0.8, ("c", 0): 0.1}
    selected = select_refresh_edges(benefit, cost, 2.0, 0.5, max_edges=2)
    keys = tuple(benefit)

    def objective(subset):
        return sum(-2.0 * benefit[edge] + 0.5 * cost[edge] for edge in subset)

    feasible = [
        subset
        for size in range(3)
        for subset in itertools.combinations(keys, size)
    ]
    optimum = min(feasible, key=objective)
    assert set(selected) == set(optimum)


def test_queue_price_removes_low_value_edges():
    benefit = {(0, 1): 0.2, (2, 1): 0.8}
    cost = {(0, 1): 1.0, (2, 1): 1.0}
    assert select_refresh_edges(benefit, cost, 1.0, 0.5) == ((2, 1),)


def test_receipt_step_is_quadratic_minimizer():
    step = receipt_step(3.0, 2.0, 4.0, 1.0)
    assert np.isclose(step, 3.0 / 8.0)
    grid = np.linspace(0.0, 1.0, 10001)
    objective = -3.0 * grid + 0.5 * 4.0 * 2.0 * grid**2
    assert abs(step - grid[np.argmin(objective)]) <= 1e-4
    assert receipt_step(-1.0, 2.0, 4.0, 1.0) == 0.0


def test_queue_update_charges_realized_cost():
    assert queue_update(2.0, 0.4, 0.7) == 1.7
    assert queue_update(0.1, 0.2, 0.7) == 0.0
