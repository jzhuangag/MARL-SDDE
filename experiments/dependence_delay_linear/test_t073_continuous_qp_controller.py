import numpy as np

from experiments.dependence_delay_linear.t073_continuous_qp_controller import (
    fingerprint_covariance,
    project_simplex,
    simulate_continuous_qp_controller,
    solve_recipient_qp,
)


def test_simplex_projection_is_feasible() -> None:
    projected = project_simplex(np.asarray([-1.0, 0.2, 2.0, 0.7]))
    assert np.all(projected >= 0.0)
    assert np.isclose(np.sum(projected), 1.0)


def test_covariance_proxy_is_psd_and_mixing_aware() -> None:
    samples = np.asarray([
        [0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [0.5, 0.5], [-0.5, -0.5]
    ])
    covariance, effective = fingerprint_covariance(samples)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12
    assert 1.0 <= effective <= samples.shape[0]


def test_qp_returns_local_under_large_debt() -> None:
    model = np.asarray([0.0, 1.0, -1.0, 2.0])
    covariance = np.eye(4) * 0.01
    low, _ = solve_recipient_qp(
        model_values=model,
        recipient_target=1.0,
        covariance_of_mean=covariance,
        recipient=0,
        debt=0.0,
        drift_weight=4.0,
        variance_weight=1.0,
    )
    high, _ = solve_recipient_qp(
        model_values=model,
        recipient_target=1.0,
        covariance_of_mean=covariance,
        recipient=0,
        debt=1e6,
        drift_weight=4.0,
        variance_weight=1.0,
    )
    assert low[0] < 0.9
    assert high[0] > 0.999
    assert np.isclose(np.sum(low), 1.0)
    assert np.isclose(np.sum(high), 1.0)


def test_controller_uses_every_transition_for_learning() -> None:
    targets = np.repeat(np.asarray([[-1.0, -1.0, 1.0, 1.0]]), 2, axis=0)
    observations = np.repeat(targets[:, None, :], 10, axis=1)
    result = simulate_continuous_qp_controller(
        observations=observations,
        targets=targets,
        initial_parameter=0.0,
        gain=0.1,
        delay=0,
        decision_blocks=[0, 1],
        pre_steps=5,
        drift_weight=4.0,
        variance_weight=1.0,
        safety_slack=0.0,
        rollback_margin=-0.05,
    )
    assert result.learning_transitions == 20
    assert result.fingerprint_transitions == 10
    assert result.message_units <= 4
    assert np.allclose(np.sum(result.proposed_weights, axis=2), 1.0)
