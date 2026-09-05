import numpy as np

from experiments.dependence_delay_linear.t073_continuous_qp_controller import (
    solve_recipient_qp,
)
from experiments.dependence_delay_linear.t075_accelerated_simplex_qp import (
    qp_components, qp_objective, solve_accelerated_qp,
)


def test_accelerated_solver_matches_frozen_cold_solver() -> None:
    rng = np.random.default_rng(20260827)
    for _ in range(64):
        model = rng.normal(size=4)
        factor = rng.normal(size=(4, 4))
        covariance = factor @ factor.T / 20.0
        recipient = int(rng.integers(0, 4))
        target = float(rng.normal())
        debt = float(rng.uniform(0.0, 2.0))
        reference, _ = solve_recipient_qp(
            model_values=model, recipient_target=target,
            covariance_of_mean=covariance, recipient=recipient, debt=debt,
            drift_weight=4.0, variance_weight=1.0,
        )
        accelerated, _, residual = solve_accelerated_qp(
            model_values=model, recipient_target=target,
            covariance_of_mean=covariance, recipient=recipient, debt=debt,
            drift_weight=4.0, variance_weight=1.0,
            initial_weights=rng.dirichlet(np.ones(4)),
        )
        hessian, linear = qp_components(
            model_values=model, recipient_target=target,
            covariance_of_mean=covariance, recipient=recipient, debt=debt,
            drift_weight=4.0, variance_weight=1.0,
        )
        assert qp_objective(accelerated, hessian, linear) <= qp_objective(reference, hessian, linear) + 2e-6
        assert np.linalg.norm(accelerated - reference) <= 3e-3
        assert residual <= 2e-5
        assert np.isclose(np.sum(accelerated), 1.0)
        assert np.all(accelerated >= 0.0)


def test_warm_start_reduces_iterations_on_nearby_qps() -> None:
    model = np.asarray([0.2, -0.1, 0.8, 0.5])
    covariance = np.asarray([
        [0.4, 0.1, 0.0, 0.0], [0.1, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.05], [0.0, 0.0, 0.05, 0.25],
    ])
    first, cold_iterations, _ = solve_accelerated_qp(
        model_values=model, recipient_target=0.45,
        covariance_of_mean=covariance, recipient=0, debt=0.2,
        drift_weight=4.0, variance_weight=1.0,
    )
    _, warm_iterations, _ = solve_accelerated_qp(
        model_values=model + 1e-4, recipient_target=0.4501,
        covariance_of_mean=covariance, recipient=0, debt=0.2,
        drift_weight=4.0, variance_weight=1.0, initial_weights=first,
    )
    assert warm_iterations < cold_iterations
