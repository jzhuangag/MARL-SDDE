import importlib.util
import sys
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
MODULE_PATH = EXPERIMENTS / "run_marl_online_drift_queue.py"
SPEC = importlib.util.spec_from_file_location("run_marl_online_drift_queue", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_validation_risk_is_bounded_and_decreases_with_return():
    low = MODULE.bounded_validation_risk(-10.0, 2.0)
    middle = MODULE.bounded_validation_risk(0.0, 2.0)
    high = MODULE.bounded_validation_risk(10.0, 2.0)
    assert 0.0 < high < middle < low < 1.0


def test_action_cost_charges_two_validation_packets_and_trajectories():
    costs = MODULE.action_costs(
        [1, 4],
        rollout_length=25,
        train_updates_per_decision=3,
        validation_q=1,
        validation_horizon=10,
        server_overhead=100,
    )
    assert costs[1].messages == 3 * (100 + 25) + 2 * (100 + 10)
    assert costs[4].messages == 3 * (100 + 100) + 2 * (100 + 10)
    assert costs[1].environment_ticks == 3 * 25 + 20
    assert costs[4].environment_ticks == 3 * 25 + 20


def test_fixed_q_uses_both_budgets():
    assert MODULE.fixed_q_updates(
        4,
        rollout_length=25,
        server_overhead=100,
        message_budget=1_000,
        environment_budget=10_000,
    ) == 5
    assert MODULE.fixed_q_updates(
        4,
        rollout_length=25,
        server_overhead=100,
        message_budget=100_000,
        environment_budget=200,
    ) == 8


class _Algorithm:
    def __init__(self, value):
        self.actor = torch.nn.Linear(2, 1)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.1)
        with torch.no_grad():
            self.actor.weight.fill_(value)
            self.actor.bias.fill_(value)


class _Critic:
    def __init__(self, value):
        self.critic = torch.nn.Linear(2, 1)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.1)
        with torch.no_grad():
            self.critic.weight.fill_(value)
            self.critic.bias.fill_(value)


class _Normalizer(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("mean", torch.tensor([value], dtype=torch.float32))


class _Runner:
    def __init__(self, value):
        self.actor = [_Algorithm(value), _Algorithm(value)]
        self.critic = _Critic(value)
        self.value_normalizer = _Normalizer(value)


def test_sync_copies_learned_state_without_replacing_runner_objects():
    source = _Runner(3.0)
    target = _Runner(-2.0)
    original_actor = target.actor[0].actor
    MODULE.synchronize_training_state(source, [target])
    assert target.actor[0].actor is original_actor
    assert torch.equal(
        target.actor[0].actor.weight,
        source.actor[0].actor.weight,
    )
    assert torch.equal(target.critic.critic.weight, source.critic.critic.weight)
    assert torch.equal(target.value_normalizer.mean, source.value_normalizer.mean)


def test_nonjoint_cost_minimum_is_rejected_by_controller():
    controller_module = sys.modules["online_drift_queue_controller"]
    with pytest.raises(ValueError, match="jointly minimize"):
        controller_module.SlidingWindowDriftQueueController(
            costs={
                1: controller_module.OnlineActionCost(10, 20),
                2: controller_module.OnlineActionCost(20, 10),
            },
            total_messages=200,
            total_environment_ticks=200,
            decisions=5,
            window=2,
            bonus_scale=0.1,
            probe_interval=3,
            progress_scale=0.1,
            queue_weight=1.0,
        )
