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


@pytest.mark.parametrize("queue", [0.0, 0.03, 0.2, 1.0])
@pytest.mark.parametrize("cost,budget", [(0.0, 0.4), (1.0, 0.4), (0.0, 1.0)])
def test_queue_remainder_bounds_exact_reflected_potential(
    queue: float, cost: float, budget: float
) -> None:
    step = 0.2
    next_queue = max(0.0, queue + step * (cost - budget))
    exact_increment = (next_queue**2 - queue**2) / (2.0 * step)
    first_order = queue * (cost - budget)
    remainder = exact_increment - first_order
    bound = queue_drift_score_error_upper(
        queue_step=step,
        maximum_cost_deviation=abs(cost - budget),
        lyapunov_weight=1.0,
    )
    assert remainder >= -1e-15
    assert remainder <= bound + 1e-15
