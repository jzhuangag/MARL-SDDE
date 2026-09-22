from __future__ import annotations

import numpy as np
import pytest

from .harl_packet_overlay import (
    SingleFlightRegistry,
    categorical_mean_kl,
    decide_harl_packet_scale,
    diagonal_gaussian_mean_kl,
    sample_split_directional_value,
    teammate_tv_drift_upper,
)


def test_single_flight_registry_enforces_ownership_and_charges_work() -> None:
    registry = SingleFlightRegistry(3)
    registry.launch(1, birth_event=4, charged_transitions=256)
    assert registry.active_agents == (1,)
    with pytest.raises(RuntimeError):
        registry.launch(1, birth_event=4, charged_transitions=256)
    completion = registry.complete(1, completion_event=9)
    assert completion.event_delay == 5
    assert completion.charged_transitions == 256
    assert registry.completed_transitions == 256
    assert registry.active_agents == ()


def test_categorical_policy_drift_is_zero_at_birth_and_positive_after_change() -> None:
    birth = np.log(np.asarray([[0.2, 0.8], [0.6, 0.4]]))
    assert categorical_mean_kl(birth, birth) == pytest.approx(0.0, abs=1e-15)
    current = np.log(np.asarray([[0.4, 0.6], [0.5, 0.5]]))
    assert categorical_mean_kl(birth, current) > 0.0


def test_diagonal_gaussian_policy_drift_matches_closed_form() -> None:
    zeros = np.zeros((3, 2))
    assert diagonal_gaussian_mean_kl(zeros, zeros, zeros, zeros) == 0.0
    shifted = np.ones((3, 2))*0.5
    # Unit variance and two coordinates: 0.5*sum_j shift_j^2 = .25.
    assert diagonal_gaussian_mean_kl(zeros, zeros, shifted, zeros) == pytest.approx(
        0.25
    )


def test_pinsker_aggregation_and_directional_value_are_observable() -> None:
    assert teammate_tv_drift_upper(np.asarray([0.0, 0.08])) == pytest.approx(0.2)
    proposal = np.asarray([0.2, -0.1, 0.4])
    validation = np.asarray([1.0, 0.5, -0.25])
    assert sample_split_directional_value(proposal, validation) == pytest.approx(0.05)


def test_harl_packet_statistics_feed_closed_form_controller() -> None:
    decision = decide_harl_packet_scale(
        proposal_step=np.asarray([0.2, -0.1]),
        validation_gradient=np.asarray([1.0, -0.5]),
        teammate_mean_kls=np.asarray([0.02, 0.08]),
        curvature_upper=0.4,
        mixed_drift_coefficient=0.3,
        debt=1.0,
        risk_budget=0.01,
        tradeoff=2.0,
    )
    assert 0.0 < decision.scale <= 1.0
    assert decision.certificate_penalty > 0.0
    assert decision.debt_after >= 0.0
