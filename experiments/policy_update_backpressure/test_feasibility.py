from __future__ import annotations

import numpy as np

from experiments.policy_update_backpressure.feasibility import (
    completion_schedule,
    declared_initial_policies,
    deterministic_optimum,
    integrate_regret,
    make_role_switch_game,
    nonmyopic_beam_schedule,
    simulate_event_rule,
)


def test_game_is_valid_and_optimum_dominates_initials() -> None:
    game = make_role_switch_game(0.8, 0.85, 0.75)
    optimum, policy = deterministic_optimum(game)
    assert policy.shape == (3, 2)
    assert all(optimum >= game.evaluate(initial)[0]-1e-12
               for initial in declared_initial_policies())


def test_completion_schedule_is_sorted_and_agent_complete() -> None:
    traces = ((1,), (2, 3), (5, 1))
    events = completion_schedule(traces, 12)
    assert events == sorted(events)
    assert {event.agent for event in events} == {0, 1, 2}
    assert all(0 < event.time <= 12 for event in events)


def test_regret_integral_matches_constant_trajectory() -> None:
    area, regret = integrate_regret([(0, 2.0)], horizon=5, optimum=3.0)
    assert area == 10.0
    assert regret == 5.0


def test_zero_rule_preserves_initial_policy() -> None:
    game = make_role_switch_game(0.6, 0.9, 1.0)
    initial = declared_initial_policies()[0]
    optimum, _ = deterministic_optimum(game)
    events = completion_schedule(((1,), (3,), (7,)), 12)
    result = simulate_event_rule(
        game, initial, events, 12, optimum, eta=0.0
    )
    assert np.isclose(result["final_return"], game.evaluate(initial)[0])
    assert result["accepted"] == 0


def test_beam_schedule_is_feasible_and_nonnegative_regret() -> None:
    game = make_role_switch_game(0.6, 0.65, 0.75)
    initial = declared_initial_policies()[1]
    optimum, _ = deterministic_optimum(game)
    events = completion_schedule(((1,), (3,), (7,)), 10)
    result = nonmyopic_beam_schedule(
        game, initial, events, 10, optimum, beam_width=32
    )
    assert result["regret"] >= 0
    assert len(result["actions"]) == len(events)
    assert set(result["actions"]).issubset({0.0, 0.25, 0.5, 0.75, 1.0})


def test_larger_beam_does_not_reduce_feasible_value() -> None:
    game = make_role_switch_game(0.85, 0.9, 1.0)
    initial = declared_initial_policies()[2]
    optimum, _ = deterministic_optimum(game)
    events = completion_schedule(((1,), (3,), (7,)), 12)
    small = nonmyopic_beam_schedule(
        game, initial, events, 12, optimum, beam_width=8
    )
    large = nonmyopic_beam_schedule(
        game, initial, events, 12, optimum, beam_width=32
    )
    assert large["area"] >= small["area"]-1e-12
