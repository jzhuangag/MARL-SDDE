from __future__ import annotations

import numpy as np
import pytest

from .adaptive_query_separation import (
    expected_async_two_query_time,
    expected_two_barrier_rounds,
    fixed_safe_step_worst_gradient,
    one_round_minimax_gradient,
    separation_certificate,
)


def test_one_round_minimax_formula_matches_direct_grid() -> None:
    low, high = 0.6, 1.5
    grid = np.linspace(0.0, 2.5, 500_001)
    worst = np.maximum(np.abs(low*grid-1.0), np.abs(high*grid-1.0))
    assert float(np.min(worst)) == pytest.approx(
        one_round_minimax_gradient(low, high), abs=2e-6
    )


def test_two_fresh_fixed_safe_queries_beat_any_one_round_output() -> None:
    for ratio in (0.15, 0.3, 0.5, 0.8, 0.95):
        high = 2.0
        low = ratio*high
        sequential = fixed_safe_step_worst_gradient(low, high, fresh_queries=2)
        nonadaptive = one_round_minimax_gradient(low, high)
        assert sequential < nonadaptive


def test_fresh_query_contraction_is_geometric() -> None:
    low, high = 0.5, 1.0
    values = [
        fixed_safe_step_worst_gradient(low, high, fresh_queries=count)
        for count in range(5)
    ]
    assert values == pytest.approx([1.0, 0.5, 0.25, 0.125, 0.0625])


def test_async_completion_formula_matches_monte_carlo() -> None:
    fast_rate, slow_rate = 2.3, 0.7
    rng = np.random.default_rng(91501)
    samples = 500_000
    fast = rng.gamma(shape=2.0, scale=1.0/fast_rate, size=samples)
    slow = rng.exponential(scale=1.0/slow_rate, size=samples)
    empirical = float(np.mean(np.maximum(fast, slow)))
    exact = expected_async_two_query_time(fast_rate, slow_rate)
    assert empirical == pytest.approx(exact, rel=2.5e-3)


def test_async_two_query_time_beats_two_barriers() -> None:
    for fast_rate, slow_rate in ((1.0, 1.0), (3.0, 0.4), (0.5, 2.0)):
        assert expected_async_two_query_time(
            fast_rate, slow_rate
        ) < expected_two_barrier_rounds(fast_rate, slow_rate)


def test_full_separation_certificate_has_nonempty_target_interval() -> None:
    result = separation_certificate(0.5, 1.0, 3.0, 0.5)
    assert result["nonempty_accuracy_interval"] is True
    assert result["strict_time_advantage"] is True
    assert result["async_two_query_worst_gradient"] == pytest.approx(0.25)
    assert result["one_round_shadow_batch_lower_bound"] == pytest.approx(1.0/3.0)
