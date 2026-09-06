from __future__ import annotations

import inspect

from .run_async_pistonball_training import (
    TrainingConfig,
    _scheduler_action,
    run_training,
)


def test_small_training_state_machine_is_finite_and_budget_feasible() -> None:
    config = TrainingConfig(
        n_agents=4,
        launches=5,
        rollout_horizon=1,
        max_cycles=6,
        replay_capacity=16,
        batch_size=2,
        budget_rate=0.5,
        cone_shell=2,
        maximum_delay=2,
        evaluations=2,
        evaluation_episodes=1,
    )
    result = run_training(
        scheduler="signed_lyapunov",
        seed=75001,
        config=config,
        device_name="cpu",
    )
    assert result["finite"]
    assert result["budget_feasible"]
    assert result["remaining_packets_after_drain"] == 0
    assert result["launched_gradient_packets"] == 4
    assert result["received_packets"] == 4
    assert result["launched_actor_transitions"] == 5 * 4
    assert result["launched_actor_transitions"] == 4 * sum(
        row["segment_cycles"] for row in result["launch_trace"]
    )
    assert len(result["evaluations"]) == 2


def test_scheduler_action_has_no_environment_outcome_argument() -> None:
    signature = inspect.signature(_scheduler_action)
    prohibited = {"reward", "return", "next_state", "termination", "truncation"}
    assert prohibited.isdisjoint(signature.parameters)


def test_training_refills_launch_horizon_after_episode_termination() -> None:
    config = TrainingConfig(
        n_agents=4,
        launches=2,
        rollout_horizon=4,
        max_cycles=2,
        replay_capacity=16,
        batch_size=2,
        budget_rate=0.0,
        cone_shell=2,
        maximum_delay=1,
        evaluations=2,
        evaluation_episodes=1,
    )
    result = run_training(
        scheduler="no_refresh",
        seed=75002,
        config=config,
        device_name="cpu",
    )
    assert result["launched_actor_transitions"] == 2 * 4 * 4
    assert all(row["segment_cycles"] == 4 for row in result["launch_trace"])
    assert result["evaluations"][-1]["actor_transitions"] == 2 * 4 * 4
    assert result["received_packets"] == result["launched_gradient_packets"]
    assert result["remaining_packets_after_drain"] == 0
