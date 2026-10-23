import numpy as np
import pytest

from experiments.dependence_delay_linear.t072_dual_use_graph_controller import (
    _select_predictable_actions,
    simulate_dual_use_controller,
)


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray([
        [-1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ])
    observations = np.repeat(targets[:, None, :], 10, axis=1)
    return observations, targets


def test_all_transitions_are_learning_and_fingerprints_are_reuse() -> None:
    observations, targets = _inputs()
    result = simulate_dual_use_controller(
        observations=observations,
        targets=targets,
        initial_parameter=0.0,
        gain=0.1,
        delay=0,
        decision_blocks=[0, 1],
        pre_steps=5,
        selection_steps=2,
        alpha_grid=[0.5, 1.0],
        drift_weight=1.0,
        safety_slack=0.0,
        rollback_margin=0.0,
    )
    assert result.learning_transitions == 20
    assert result.fingerprint_transitions == 10
    assert result.message_units <= 4
    assert result.candidate_scores == 2 * 4 * 7


def test_second_half_can_rollback_a_bad_predictable_proposal() -> None:
    observations, targets = _inputs()
    observations[0, :5, 0] = 1.0
    observations[0, 5:, 0] = -1.0
    result = simulate_dual_use_controller(
        observations=observations,
        targets=targets,
        initial_parameter=0.0,
        gain=0.1,
        delay=0,
        decision_blocks=[0],
        pre_steps=5,
        selection_steps=2,
        alpha_grid=[0.5, 1.0],
        drift_weight=1.0,
        safety_slack=0.0,
        rollback_margin=0.0,
    )
    assert result.rollback_states.shape == (1, 4)
    assert np.all(result.debt_path >= 0.0)


def test_invalid_split_is_rejected() -> None:
    observations, targets = _inputs()
    with pytest.raises(ValueError, match="invalid dual-use split"):
        simulate_dual_use_controller(
            observations=observations,
            targets=targets,
            initial_parameter=0.0,
            gain=0.1,
            delay=0,
            decision_blocks=[0],
            pre_steps=10,
            selection_steps=2,
            alpha_grid=[0.5, 1.0],
            drift_weight=1.0,
            safety_slack=0.0,
            rollback_margin=0.0,
        )


def test_safety_debt_suppresses_large_mixing_displacement() -> None:
    local = np.asarray([0.0, 1.0, -1.0, 2.0])
    samples = np.repeat(np.asarray([[1.0, 1.0, -1.0, 2.0]]), 2, axis=0)
    _, low_debt, _ = _select_predictable_actions(
        local_pre=local,
        shadow_pre=local,
        donor_snapshot=local,
        selection_samples=samples,
        alpha_grid=[0.5, 1.0],
        debt=np.zeros(4),
        drift_weight=10.0,
    )
    _, high_debt, _ = _select_predictable_actions(
        local_pre=local,
        shadow_pre=local,
        donor_snapshot=local,
        selection_samples=samples,
        alpha_grid=[0.5, 1.0],
        debt=np.asarray([10.0, 0.0, 0.0, 0.0]),
        drift_weight=10.0,
    )
    assert low_debt[0] > 0
    assert high_debt[0] == 0
