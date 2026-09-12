import numpy as np

from experiments.dependence_delay_linear.t081_end_block_primal_dual_controller import (
    simulate_end_block_controller,
)


def fixture():
    rng = np.random.default_rng(12)
    targets = np.asarray([
        [-0.4, -0.4, 0.4, 0.4],
        [-0.4, 0.4, -0.4, 0.4],
        [-0.4, 0.4, -0.4, 0.4],
    ])
    observations = targets[:, None, :] + 0.1 * rng.normal(size=(3, 10, 4))
    return observations, targets


def test_end_block_controller_is_causal_under_future_perturbation():
    observations, targets = fixture()
    common = dict(
        targets=targets,
        initial_parameter=1.0,
        gain=0.04,
        delay=1,
        decision_blocks=[0, 1, 2],
        drift_weight=4.0,
        variance_weight=1.0,
        safety_slack=0.0,
        certificate_delta=0.1,
        rho_cap=0.95,
    )
    first = simulate_end_block_controller(observations=observations, **common)
    changed = observations.copy()
    changed[1:] += 100.0
    second = simulate_end_block_controller(observations=changed, **common)
    assert np.array_equal(first.accepted_weights[0], second.accepted_weights[0])


def test_all_samples_are_learning_updates_and_messages_are_charged():
    observations, targets = fixture()
    result = simulate_end_block_controller(
        observations=observations,
        targets=targets,
        initial_parameter=1.0,
        gain=0.04,
        delay=0,
        decision_blocks=[0, 2],
        drift_weight=4.0,
        variance_weight=1.0,
        safety_slack=0.0,
        certificate_delta=0.1,
        rho_cap=0.95,
    )
    assert result.learning_transitions == 30
    assert result.extra_probe_transitions == 0
    assert 2 <= result.message_units <= 4
    assert np.all(result.debt_path >= 0.0)


def test_identity_fixed_graph_matches_local_recursion():
    observations, targets = fixture()
    result = simulate_end_block_controller(
        observations=observations,
        targets=targets,
        initial_parameter=1.0,
        gain=0.04,
        delay=2,
        decision_blocks=[0, 1, 2],
        drift_weight=4.0,
        variance_weight=1.0,
        safety_slack=0.0,
        certificate_delta=0.1,
        rho_cap=0.95,
        fixed_weights=np.eye(4),
    )
    parameter = np.repeat(1.0, 4)
    expected = []
    for block in range(3):
        for sample in observations[block]:
            parameter = parameter + 0.04 * (sample - parameter)
        expected.append(float(np.mean(np.square(parameter - targets[block]))))
    assert np.allclose(result.risk_path, expected)


def test_fixed_graph_validation_rejects_nonstochastic_matrix():
    observations, targets = fixture()
    bad = np.eye(4)
    bad[0, 0] = 2.0
    try:
        simulate_end_block_controller(
            observations=observations,
            targets=targets,
            initial_parameter=1.0,
            gain=0.04,
            delay=0,
            decision_blocks=[0],
            drift_weight=4.0,
            variance_weight=1.0,
            safety_slack=0.0,
            certificate_delta=0.1,
            rho_cap=0.95,
            fixed_weights=bad,
        )
    except ValueError:
        return
    raise AssertionError("nonstochastic fixed graph was accepted")
