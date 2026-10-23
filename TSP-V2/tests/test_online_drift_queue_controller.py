import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "online_drift_queue_controller.py"
)
SPEC = importlib.util.spec_from_file_location("online_drift_queue_controller", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_controller(**overrides):
    arguments = {
        "costs": {
            1: MODULE.OnlineActionCost(messages=10, environment_ticks=5),
            2: MODULE.OnlineActionCost(messages=20, environment_ticks=5),
            4: MODULE.OnlineActionCost(messages=40, environment_ticks=5),
        },
        "total_messages": 200,
        "total_environment_ticks": 50,
        "decisions": 10,
        "eta": 0.1,
        "exploration": 0.1,
        "progress_scale": 1.0,
        "forgetting": 1.0,
        "queue_weight": 1.0,
        "seed": 17,
    }
    arguments.update(overrides)
    return MODULE.OnlineDriftQueueController(**arguments)


def test_rejects_budget_that_cannot_reserve_cheapest_action():
    with pytest.raises(ValueError, match="cheapest"):
        make_controller(total_messages=99)


def test_validation_observation_must_be_bounded_and_ordered():
    controller = make_controller()
    controller.decide()
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        controller.observe(risk_before=1.1, risk_after=0.2)


def test_observation_updates_only_selected_bandit_coordinate():
    controller = make_controller(exploration=0.0)
    decision = controller.decide()
    before = dict(controller.log_weights)
    observation = controller.observe(risk_before=0.8, risk_after=0.2)
    assert observation.progress == pytest.approx(0.6)
    assert observation.bounded_reward == pytest.approx(0.8)
    for q in controller.actions:
        if q == decision.action:
            assert controller.log_weights[q] > before[q]
        else:
            assert controller.log_weights[q] == before[q]


def test_negative_progress_decreases_selected_coordinate():
    controller = make_controller(exploration=0.0, progress_scale=0.2)
    decision = controller.decide()
    before = controller.log_weights[decision.action]
    controller.observe(risk_before=0.2, risk_after=0.3)
    assert controller.log_weights[decision.action] < before


def test_queue_prices_expensive_actions():
    controller = make_controller(exploration=0.0)
    base = controller.probabilities()
    controller.message_queue = 10.0
    priced = controller.probabilities()
    assert base[1] == pytest.approx(base[4])
    assert priced[1] > priced[2] > priced[4]


def test_reserve_mask_guarantees_pathwise_dual_budget_safety():
    controller = make_controller(total_messages=130)
    for _ in range(10):
        decision = controller.decide()
        controller.observe(risk_before=0.5, risk_after=0.5)
        assert controller.message_remaining >= 0
        assert controller.environment_remaining >= 0
        if controller.decisions_remaining:
            assert 1 in controller._reserve_feasible_actions()
    assert controller.decisions_remaining == 0


def test_decide_requires_pending_observation():
    controller = make_controller()
    controller.decide()
    with pytest.raises(RuntimeError, match="pending"):
        controller.decide()


def test_registered_progress_scale_clips_bandit_reward_only():
    controller = make_controller(progress_scale=0.1)
    controller.decide()
    observation = controller.observe(risk_before=0.9, risk_after=0.1)
    assert observation.progress == pytest.approx(0.8)
    assert observation.bounded_reward == pytest.approx(1.0)


def make_window_controller(**overrides):
    arguments = {
        "costs": {
            1: MODULE.OnlineActionCost(messages=10, environment_ticks=5),
            2: MODULE.OnlineActionCost(messages=20, environment_ticks=5),
            4: MODULE.OnlineActionCost(messages=40, environment_ticks=5),
        },
        "total_messages": 400,
        "total_environment_ticks": 50,
        "decisions": 10,
        "window": 4,
        "bonus_scale": 0.2,
        "probe_interval": 5,
        "progress_scale": 0.1,
        "queue_weight": 20.0,
    }
    arguments.update(overrides)
    return MODULE.SlidingWindowDriftQueueController(**arguments)


def test_window_controller_samples_every_action_once():
    controller = make_window_controller()
    selected = []
    for _ in range(3):
        selected.append(controller.decide().action)
        controller.observe(risk_before=0.5, risk_after=0.49)
    assert selected == [1, 2, 4]


def test_window_controller_prefers_observed_progress_after_initialization():
    controller = make_window_controller()
    after = {1: 0.50, 2: 0.48, 4: 0.45}
    for _ in range(3):
        action = controller.decide().action
        controller.observe(risk_before=0.5, risk_after=after[action])
    assert controller.decide().action == 4


def test_window_controller_preserves_dual_budgets():
    controller = make_window_controller(total_messages=130)
    for _ in range(10):
        controller.decide()
        controller.observe(risk_before=0.5, risk_after=0.5)
    assert controller.message_remaining >= 0
    assert controller.environment_remaining >= 0

