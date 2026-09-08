import pytest

from .controlled_cache_performance_bounds import (
    proxy_greedy_policy_regret_upper,
    queue_drift_score_error_upper,
)


def test_proxy_greedy_bound_pays_twice_both_uniform_errors() -> None:
    assert proxy_greedy_policy_regret_upper(
        continuation_value_errors=(0.1, 0.2),
        action_score_errors=(0.03, 0.04),
    ) == pytest.approx(0.74)


def test_exact_continuation_and_scores_have_zero_policy_regret_bound() -> None:
    assert proxy_greedy_policy_regret_upper(
        continuation_value_errors=(0.0, 0.0)
    ) == 0.0


def test_queue_drift_error_has_the_scaled_half_square_constant() -> None:
    assert queue_drift_score_error_upper(
        queue_step=0.2,
        maximum_cost_deviation=0.75,
        lyapunov_weight=3.0,
    ) == pytest.approx(0.01875)
