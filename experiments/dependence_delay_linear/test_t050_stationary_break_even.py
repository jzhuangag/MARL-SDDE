import itertools
import math

import numpy as np

from experiments.dependence_delay_linear.t049_standard_task_exact import (
    exact_gradient_lag_covariances,
)
from experiments.dependence_delay_linear.t050_stationary_break_even import (
    asymptotic_participation_coefficient,
    continuous_information_optimum,
    continuous_pr_optimum,
    contraction_burn_in_horizon,
    controller_beats_baseline_leading_order,
    exact_edge_long_run_covariance,
    independent_schedule_bound,
    information_efficiency,
    optimal_catalogue_q,
    pr_task_constant,
    probe_break_even_budget,
    sequence_information,
)


def test_continuous_information_optimum_matches_dense_grid():
    optimum = continuous_information_optimum(
        minimum_q=1.0,
        maximum_q=64.0,
        overhead=16.0,
        common_variance=0.25,
        private_variance=1.0,
    )
    assert optimum == 8.0
    grid = np.linspace(1.0, 64.0, 100_001)
    efficiencies = [
        information_efficiency(
            q,
            overhead=16.0,
            common_variance=0.25,
            private_variance=1.0,
        )
        for q in grid
    ]
    assert abs(grid[int(np.argmax(efficiencies))] - optimum) < 1e-3


def test_schedule_information_is_bounded_by_best_efficiency():
    budget = 181
    overhead = 7
    candidates = [1, 4, 16]
    bound = independent_schedule_bound(
        budget=budget,
        candidates=candidates,
        overhead=overhead,
        common_variance=0.2,
        private_variance=1.0,
    )
    for length in range(1, 8):
        for schedule in itertools.product(candidates, repeat=length):
            cost = sum(overhead + q for q in schedule)
            if cost <= budget:
                information = sequence_information(
                    schedule, common_variance=0.2, private_variance=1.0
                )
                assert information <= budget * bound["best_efficiency"] + 1e-12


def test_schedule_improvement_cap_covers_exact_knapsack():
    budget = 73
    overhead = 5
    candidates = [1, 4, 9]
    bound = independent_schedule_bound(
        budget=budget,
        candidates=candidates,
        overhead=overhead,
        common_variance=0.3,
        private_variance=1.0,
    )
    values = {
        q: sequence_information(
            [q], common_variance=0.3, private_variance=1.0
        )
        for q in candidates
    }
    dynamic = np.zeros(budget + 1)
    for used in range(budget + 1):
        for q in candidates:
            cost = overhead + q
            if used + cost <= budget:
                dynamic[used + cost] = max(
                    dynamic[used + cost], dynamic[used] + values[q]
                )
    schedule_information = float(np.max(dynamic))
    fixed_information = 1.0 / bound["best_fixed_risk"]
    exact_improvement = 1.0 - fixed_information / schedule_information
    assert exact_improvement <= bound["schedule_relative_improvement_upper_bound"]


def test_continuous_pr_optimum_and_catalogue_direction():
    optimum = continuous_pr_optimum(
        minimum_q=1.0, maximum_q=64.0, overhead=32.0, rho=0.2
    )
    assert abs(optimum - math.sqrt(128.0)) < 1e-12
    low = optimal_catalogue_q([1, 4, 16, 64], overhead=32.0, rho=0.01)
    high = optimal_catalogue_q([1, 4, 16, 64], overhead=32.0, rho=0.9)
    assert low["q"] > high["q"]
    assert low["coefficient"] == min(
        asymptotic_participation_coefficient(q, overhead=32.0, rho=0.01)
        for q in [1, 4, 16, 64]
    )


def test_probe_break_even_is_exact_rearrangement():
    arguments = dict(
        probe_message_cost=400.0,
        baseline_coefficient=20.0,
        oracle_coefficient=12.0,
        wrong_action_coefficient=30.0,
        error_probability=0.05,
    )
    threshold = probe_break_even_budget(**arguments)
    assert threshold > arguments["probe_message_cost"]
    assert not controller_beats_baseline_leading_order(
        total_budget=threshold * (1.0 - 1e-10), **arguments
    )
    assert controller_beats_baseline_leading_order(
        total_budget=threshold * (1.0 + 1e-10), **arguments
    )


def test_break_even_is_infinite_without_certified_gap():
    threshold = probe_break_even_budget(
        probe_message_cost=10.0,
        baseline_coefficient=4.0,
        oracle_coefficient=3.0,
        wrong_action_coefficient=8.0,
        error_probability=0.5,
    )
    assert math.isinf(threshold)


def test_exact_edge_long_run_covariance_matches_truncated_lags():
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    stationary = np.array([0.6, 0.4])
    conditional = np.array([[1.0, -0.5], [-1.5, 0.75]])
    edge = transition[:, :, None] * conditional[:, None, :]
    second = sum(
        stationary[s]
        * transition[s, u]
        * np.outer(conditional[s], conditional[s])
        for s in range(2)
        for u in range(2)
    )
    task = {
        "continuing_transition": transition,
        "stationary": stationary,
        "edge_gradient_sum": edge,
        "conditional_gradient": conditional,
        "gradient_second_moment": second,
    }
    exact = exact_edge_long_run_covariance(
        transition=transition,
        stationary=stationary,
        edge_gradient_sum=edge,
        conditional_gradient=conditional,
        second_moment=second,
    )
    lags = exact_gradient_lag_covariances(task, horizon=200)
    truncated = np.sum(lags, axis=0)
    np.testing.assert_allclose(exact, truncated, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(exact)) >= -1e-12


def test_pr_task_constant_scalar_case():
    value = pr_task_constant(
        drift=np.array([[2.0]]), long_run_covariance=np.array([[3.0]])
    )
    assert value == 0.75


def test_contraction_horizon_satisfies_target_and_is_minimal():
    radius = 0.9965
    target = 1e-3
    horizon = contraction_burn_in_horizon(
        spectral_radius=radius, target=target, averaging_fraction=0.5
    )
    assert radius ** math.floor(0.5 * horizon) <= target
    assert radius ** math.floor(0.5 * (horizon - 2)) > target
