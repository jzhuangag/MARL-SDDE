from __future__ import annotations

import pytest

from .certified_factor_async import (
    AsyncFactorConfig,
    forecast_reversal_config,
    generate_common_path,
    periodic_budget_slot,
    simulate_async_factor_policy,
)


def test_forecast_reversal_has_zero_stationary_factor_means_and_rank_change() -> None:
    config = forecast_reversal_config(cycle_probability=0.95, total_events=32)
    import numpy as np

    from .markov_alignment_certificate import finite_horizon_markov_alignment

    scores = np.asarray(config.score_by_action)
    transition = np.asarray(config.transition)
    assert np.allclose(scores.mean(axis=1), 0.0)
    myopic = np.argmax(scores, axis=0)
    future = np.argmax(
        np.asarray(
            [
                [
                    finite_horizon_markov_alignment(
                        transition=transition,
                        score_by_state=scores[action],
                        initial_state=state,
                        horizon=config.packet_horizon,
                    )
                    for state in range(3)
                ]
                for action in range(3)
            ]
        ),
        axis=0,
    )
    assert tuple(myopic) == (2, 0, 2)
    assert tuple(future) == (0, 1, 2)

    slow = forecast_reversal_config(cycle_probability=0.05, total_events=32)
    slow_transition = np.asarray(slow.transition)
    slow_future = np.argmax(
        np.asarray(
            [
                [
                    finite_horizon_markov_alignment(
                        transition=slow_transition,
                        score_by_state=scores[action],
                        initial_state=state,
                        horizon=slow.packet_horizon,
                    )
                    for state in range(3)
                ]
                for action in range(3)
            ]
        ),
        axis=0,
    )
    assert tuple(slow_future) == tuple(myopic)


def test_periodic_budget_prefix() -> None:
    config = AsyncFactorConfig(
        total_events=12,
        communication_budget=0.5,
    )
    slots = [periodic_budget_slot(event, 0.5) for event in range(16)]
    assert sum(slots) == 8


def test_common_path_is_reproducible_and_has_declared_shapes() -> None:
    config = AsyncFactorConfig(
        total_events=12,
    )
    first = generate_common_path(config=config, seed=11)
    second = generate_common_path(config=config, seed=11)
    assert all((left == right).all() for left, right in zip(first, second))
    assert first[0].size == config.total_events * config.packet_horizon + 1


def test_certified_controller_charges_warmup_and_respects_long_run_budget() -> None:
    config = AsyncFactorConfig(
        total_events=272,
        communication_budget=0.5,
        maximum_delay=2,
        certificate_refresh_period=16,
    )
    result = simulate_async_factor_policy(
        policy="certified_joint", seed=17, config=config
    )
    assert result.messages <= (
        config.communication_budget * result.total_events
        + result.final_queue / config.queue_step
        + 1e-12
    )
    assert result.environment_transitions == result.total_events * config.packet_horizon
    assert result.applied_packets == result.positive_weight_packets
    assert result.warmup_events == 0
    assert result.terminal_potential >= 0.0


def test_exact_joint_and_fixed_policy_are_finite() -> None:
    config = AsyncFactorConfig(
        total_events=68,
        communication_budget=0.5,
        maximum_delay=3,
    )
    for policy in (
        "plugin_joint",
        "exact_joint",
        "fixed_initial",
        "state_myopic",
        "round_robin",
        "no_refresh",
    ):
        result = simulate_async_factor_policy(policy=policy, seed=23, config=config)
        assert result.terminal_potential >= 0.0
        assert result.average_potential >= 0.0
        if policy == "no_refresh":
            assert result.messages == 0
        elif policy in ("plugin_joint", "exact_joint"):
            assert result.messages <= (
                config.communication_budget * result.total_events
                + result.final_queue / config.queue_step
                + 1e-12
            )
        else:
            assert result.messages <= config.communication_budget * result.total_events + 1
        assert result.messages <= result.positive_weight_packets


def test_invalid_total_event_count_fails_closed() -> None:
    with pytest.raises(ValueError):
        generate_common_path(
            config=AsyncFactorConfig(total_events=0),
            seed=1,
        )
