import numpy as np
import pytest

from experiments.policy_dependency_sync.expected_drift import (
    choose_drift_plus_penalty,
    queue_update,
    realized_selection_regret,
)


def test_noisy_selection_regret_is_bounded_by_twice_sup_error():
    rng = np.random.default_rng(90210)
    actions = tuple(range(7))
    for _ in range(2_000):
        truth = {action: float(rng.normal()) for action in actions}
        estimate = {
            action: truth[action] + float(rng.normal(scale=0.3))
            for action in actions
        }
        regret, bound = realized_selection_regret(truth, estimate)
        assert regret >= -1e-12
        assert regret <= bound + 1e-12


def test_large_queue_selects_zero_cost_action_under_bounded_scores():
    scores = {"none": 0.8, "edge_1": -0.9, "edge_2": -0.7}
    costs = {"none": 0.0, "edge_1": 1.0, "edge_2": 1.0}
    choice = choose_drift_plus_penalty(scores, costs, queue=2.0)
    assert choice.action == "none"
    assert choice.cost == 0.0


def test_queue_identity_gives_pathwise_average_budget_bound():
    budget = 0.37
    costs = (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
    queue = 0.0
    initial = queue
    for cost in costs:
        queue = queue_update(queue, cost, budget)
    average_cost = sum(costs) / len(costs)
    assert average_cost <= budget + (queue - initial) / len(costs) + 1e-12


@pytest.mark.parametrize(
    "queue,cost,budget",
    [(-1.0, 0.0, 0.5), (0.0, -1.0, 0.5), (0.0, 0.0, -0.5)],
)
def test_queue_rejects_negative_inputs(queue, cost, budget):
    with pytest.raises(ValueError):
        queue_update(queue, cost, budget)
