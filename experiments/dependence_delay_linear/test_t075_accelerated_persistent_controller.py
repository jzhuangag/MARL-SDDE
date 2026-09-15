import numpy as np

from experiments.dependence_delay_linear.t074_persistent_certificate_controller import (
    simulate_persistent_qp_controller,
)
from experiments.dependence_delay_linear.t075_accelerated_persistent_controller import (
    simulate_accelerated_persistent_controller,
)


def test_accelerated_path_matches_frozen_controller_on_deterministic_fixture() -> None:
    rng = np.random.default_rng(8173)
    targets = np.repeat(np.asarray([[-0.3, -0.3, 0.3, 0.3]]), 8, axis=0)
    observations = targets[:, None, :] + 0.1 * rng.standard_normal((8, 10, 4))
    common = dict(
        observations=observations, targets=targets, initial_parameter=0.5,
        gain=0.04, delay=1, decision_blocks=[0, 4], pre_steps=5,
        drift_weight=4.0, variance_weight=1.0, safety_slack=0.0,
        rollback_margin=-0.02, certificate_delta=0.1, rho_cap=0.95)
    frozen = simulate_persistent_qp_controller(**common)
    accelerated = simulate_accelerated_persistent_controller(**common)
    assert np.max(np.abs(frozen.risk_path - accelerated.risk_path)) <= 2e-5
    assert np.array_equal(frozen.rollback_states, accelerated.rollback_states)
    # This fixture has a semidefinite first-decision QP with nonunique weights;
    # path/objective equivalence is primary and the residual remains diagnostic.
    assert np.max(accelerated.qp_residual_path) <= 3e-4
    assert accelerated.learning_transitions == 80
    assert accelerated.message_units == frozen.message_units
    assert accelerated.qp_iterations <= frozen.qp_iterations


def test_warm_start_state_is_predictable_and_feasible() -> None:
    targets = np.repeat(np.asarray([[-1.0, -1.0, 1.0, 1.0]]), 5, axis=0)
    observations = np.repeat(targets[:, None, :], 10, axis=1)
    result = simulate_accelerated_persistent_controller(
        observations=observations, targets=targets, initial_parameter=0.0,
        gain=0.04, delay=0, decision_blocks=[0, 4], pre_steps=5,
        drift_weight=4.0, variance_weight=1.0, safety_slack=0.0,
        rollback_margin=-0.02, certificate_delta=0.1, rho_cap=0.95)
    assert np.all(result.accepted_weights >= 0.0)
    assert np.allclose(np.sum(result.accepted_weights, axis=2), 1.0)
    assert result.fingerprint_transitions == 10
