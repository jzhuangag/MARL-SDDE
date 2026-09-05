import numpy as np

from experiments.dependence_delay_linear.run_t070a_exact_nonstationary_graph_scan import (
    unit_schedule,
)
from experiments.dependence_delay_linear.t079_continuous_static_graph import (
    catalogue_graph_to_weights,
    optimize_static_graph,
    simulate_continuous_graph,
    solve_simplex_quadratic,
)


def test_simplex_quadratic_matches_closed_form_diagonal_case():
    matrix = np.diag([1.0, 2.0, 3.0, 4.0])
    weights, residual = solve_simplex_quadratic(matrix)
    expected = 1.0 / np.diag(matrix)
    expected /= np.sum(expected)
    assert np.allclose(weights, expected, atol=1e-12)
    assert residual <= 1e-12


def test_catalogue_graph_conversion_is_row_stochastic():
    weights = catalogue_graph_to_weights([1, 2, 5, 6], agents=4, alpha_grid=[0.5, 1.0])
    assert np.all(weights >= 0.0)
    assert np.allclose(np.sum(weights, axis=1), 1.0)


def _toy_trajectory(weights=None, dynamic=False):
    parent = {
        "model": {
            "patterns": {"A": [-1.0, -1.0, 1.0, 1.0], "B": [-1.0, 1.0, -1.0, 1.0]},
            "blocks": 24,
        }
    }
    schedule = 0.6 * unit_schedule(parent, "single_switch")
    return simulate_continuous_graph(
        target_schedule=schedule,
        initial_parameter=1.5,
        delay=1,
        decision_blocks=[0, 4, 8, 12, 16, 20],
        gain=0.04,
        curvature=1.0,
        local_steps=10,
        noise_scale=0.1,
        spatial_correlation=0.0,
        temporal_correlation=0.0,
        fixed_weights=weights,
        safe_dynamic_oracle=dynamic,
    )


def test_dynamic_oracle_is_finite_safe_and_fully_charged():
    result = _toy_trajectory(dynamic=True)
    assert np.isfinite(result.auc_risk)
    assert result.learning_transitions == 240
    assert result.extra_probe_transitions == 0
    assert result.message_units <= 12
    assert result.maximum_row_kkt_residual <= 1e-8


def test_multistart_static_optimizer_cannot_lose_to_discrete_start():
    discrete = catalogue_graph_to_weights([1, 1, 5, 5], agents=4, alpha_grid=[0.5, 1.0])
    baseline = _toy_trajectory(weights=discrete)
    optimized = optimize_static_graph(
        lambda matrix: _toy_trajectory(weights=matrix),
        agents=4,
        discrete_start=discrete,
        maximum_iterations=30,
    )
    assert optimized.auc_risk <= baseline.auc_risk + 1e-10
    assert optimized.total_starts == 10
    assert optimized.successful_starts >= 1
    assert optimized.row_sum_residual <= 1e-10
