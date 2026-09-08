import pytest

from .controlled_cache_performance_bounds import (
    drift_structured_action_score,
    drift_structured_relative_value,
    proxy_greedy_policy_regret_upper,
    queue_drift_score_error_upper,
    residual_shielded_queue_cap,
)


def test_drift_structured_score_is_exact_bellman_score() -> None:
    scale = 4.0
    stage_reward = 0.7
    next_potential = 1.6
    next_optimal_value = 2.3
    residual = drift_structured_relative_value(
        optimal_value=next_optimal_value,
        current_potential=next_potential,
        lyapunov_scale=scale,
    )
    score = drift_structured_action_score(
        stage_reward=stage_reward,
        expected_next_potential=next_potential,
        expected_next_relative_value=residual,
        lyapunov_scale=scale,
    )
    assert score == pytest.approx(stage_reward + next_optimal_value)


def test_zero_residual_recovers_potential_only_score() -> None:
    assert drift_structured_action_score(
        stage_reward=1.0,
        expected_next_potential=0.6,
        expected_next_relative_value=0.0,
        lyapunov_scale=3.0,
    ) == pytest.approx(0.8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimal_value": 1.0, "current_potential": -0.1, "lyapunov_scale": 1.0},
        {"optimal_value": 1.0, "current_potential": 0.1, "lyapunov_scale": 0.0},
    ],
)
def test_relative_value_rejects_invalid_potential_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        drift_structured_relative_value(**kwargs)


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


def test_residual_queue_cap_pays_score_threshold_and_one_step_overshoot() -> None:
    assert residual_shielded_queue_cap(
        nonqueue_score_advantage_upper=3.0,
        minimum_positive_cost=0.5,
        queue_step=0.2,
        maximum_cost=1.5,
    ) == pytest.approx(6.3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "nonqueue_score_advantage_upper": -1.0,
            "minimum_positive_cost": 0.5,
            "queue_step": 0.2,
            "maximum_cost": 1.0,
        },
        {
            "nonqueue_score_advantage_upper": 1.0,
            "minimum_positive_cost": 0.0,
            "queue_step": 0.2,
            "maximum_cost": 1.0,
        },
        {
            "nonqueue_score_advantage_upper": 1.0,
            "minimum_positive_cost": 1.0,
            "queue_step": 0.2,
            "maximum_cost": 0.5,
        },
    ],
)
def test_residual_queue_cap_rejects_invalid_bounds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        residual_shielded_queue_cap(**kwargs)


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
