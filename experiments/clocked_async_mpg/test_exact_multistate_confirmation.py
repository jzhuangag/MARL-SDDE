from __future__ import annotations

import numpy as np
import pytest

from .exact_multistate_confirmation import (
    INITIAL_LOGITS,
    make_game,
    maximum_event_delay,
    potential_and_gradient,
    simulate_asynchronous,
    simulate_shadow_barrier,
    summarize_trajectory,
)


def test_exact_gradient_matches_finite_difference() -> None:
    game = make_game(0.17)
    _, gradient = potential_and_gradient(INITIAL_LOGITS, game)
    finite = np.zeros_like(gradient)
    epsilon = 1e-6
    for index in np.ndindex(INITIAL_LOGITS.shape):
        positive = INITIAL_LOGITS.copy()
        negative = INITIAL_LOGITS.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        finite[index] = (
            potential_and_gradient(positive, game)[0]
            -potential_and_gradient(negative, game)[0]
        )/(2.0*epsilon)
    assert gradient == pytest.approx(finite, abs=2e-9)


def test_global_cross_lipschitz_matrix_covers_random_pairs() -> None:
    rng = np.random.default_rng(91701)
    game = make_game(0.24)
    for _ in range(100):
        left = rng.normal(scale=2.0, size=INITIAL_LOGITS.shape)
        right = rng.normal(scale=2.0, size=INITIAL_LOGITS.shape)
        left_gradient = potential_and_gradient(left, game)[1]
        right_gradient = potential_and_gradient(right, game)[1]
        block_distance = np.linalg.norm(left-right, axis=1)
        for agent in range(2):
            mismatch = np.linalg.norm(left_gradient[agent]-right_gradient[agent])
            assert mismatch <= game.lipschitz[agent]@block_distance+1e-12


def test_registered_delay_bound_covers_bounded_service_schedule() -> None:
    for ratio in (1.0, 2.0, 4.0, 8.0):
        result = simulate_asynchronous(
            0.1, ratio, 3, "static-test", maximum_time=80.0
        )
        assert int(result["max_realized_delay"]) <= maximum_event_delay(ratio)


def test_both_policies_are_finite_and_make_progress() -> None:
    asynchronous = simulate_asynchronous(
        0.08, 4.0, 2, "static-test", maximum_time=60.0
    )
    shadow = simulate_shadow_barrier(
        0.08, 4.0, 2, "static-test", maximum_time=60.0
    )
    for result in (asynchronous, shadow):
        trajectory = result["trajectory"]
        assert trajectory[-1]["normalized_gap"] < trajectory[0]["normalized_gap"]
        assert all(
            np.isfinite(list(row.values())).all() for row in trajectory
        )


def test_summary_reports_first_target_crossing() -> None:
    trajectory = [
        {"gradient_norm": 2.0, "normalized_gap": 1.0, "packets": 0.0, "potential": 0.0, "time": 0.0, "updates": 0.0},
        {"gradient_norm": 1.0, "normalized_gap": 0.4, "packets": 2.0, "potential": 1.0, "time": 2.0, "updates": 1.0},
        {"gradient_norm": 0.5, "normalized_gap": 0.1, "packets": 4.0, "potential": 2.0, "time": 4.0, "updates": 2.0},
    ]
    summary = summarize_trajectory(trajectory, 0.2)
    assert summary["time_to_target"] == pytest.approx(4.0)
