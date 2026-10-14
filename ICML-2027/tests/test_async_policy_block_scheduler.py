import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from async_policy_block_scheduler import select_policy_blocks


def test_single_agent_step_is_closed_form_truncated_quadratic_optimum():
    action = select_policy_blocks(
        gradient_norms=[2.0], confidence_radii=[0.5],
        curvature_majorant=[[2.0]], message_costs=[0], environment_costs=[0],
        alpha_max=1.0,
    )
    # a = ||g||^2 - eps||g|| = 3; H = L ||g||^2 = 8; alpha*=a/H=3/8.
    assert action.agents == (0,)
    assert action.step_sizes[0] == pytest.approx(3 / 8, abs=1e-7)
    assert action.certified_gain_bound == pytest.approx(9 / 16, abs=1e-7)


def test_joint_interaction_can_make_serial_update_tie_optimal():
    action = select_policy_blocks(
        gradient_norms=[1.0, 1.0], confidence_radii=[0.0, 0.0],
        curvature_majorant=[[1.0, 1.0], [1.0, 1.0]],
        message_costs=[0, 0], environment_costs=[0, 0], alpha_max=1.0,
    )
    # The fully coupled rank-one curvature makes the best joint lower bound
    # equal to a one-agent action; deterministic tie-break prefers fewer packets.
    assert action.agents == (0,)
    assert action.certified_gain_bound == pytest.approx(0.5, abs=1e-7)


def test_queues_can_select_idle_action_when_update_cost_exceeds_certified_gain():
    action = select_policy_blocks(
        gradient_norms=[1.0], confidence_radii=[0.0],
        curvature_majorant=[[1.0]], message_costs=[1.0], environment_costs=[0.0],
        value_weight=1.0, message_pressure=0.6, alpha_max=1.0,
    )
    assert action.agents == ()
    assert action.drift_score == 0.0


def test_uncertain_gradient_is_not_scheduled():
    action = select_policy_blocks(
        gradient_norms=[1.0], confidence_radii=[1.1],
        curvature_majorant=[[1.0]], message_costs=[0], environment_costs=[0],
    )
    assert action.agents == ()


def test_rejects_non_psd_curvature_majorant():
    with pytest.raises(ValueError, match="positive semidefinite"):
        select_policy_blocks(
            gradient_norms=[1.0, 1.0], confidence_radii=[0.0, 0.0],
            curvature_majorant=[[1.0, 2.0], [2.0, 1.0]],
            message_costs=[0, 0], environment_costs=[0, 0],
        )


def test_outer_subset_choice_is_global_for_the_declared_surrogate():
    action = select_policy_blocks(
        gradient_norms=[1.0, 1.2, 0.7], confidence_radii=[0.05, 0.1, 0.03],
        curvature_majorant=np.eye(3).tolist(), message_costs=[0.01, 5.0, 0.0],
        environment_costs=[0.0, 0.0, 0.0], message_pressure=0.2, alpha_max=1.0,
    )
    assert action.agents == (0, 2)
    assert action.drift_score > 0
