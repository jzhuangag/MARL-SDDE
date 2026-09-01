from __future__ import annotations

import numpy as np
import pytest

from .clocked_optimism_phase import (
    choose_clocked_optimism,
    heterogeneous_clock_metric,
    heterogeneous_potential_drift_coefficient,
    heterogeneous_rotational_drift_coefficient,
    lifted_mean_square_spectral_radius,
    lifted_rotational_transition,
    potential_coordinate_factors,
    randomized_factor,
    rotational_coordinate_factors,
    rotational_optimism_threshold,
    stale_optimistic_lifted_spectral_radius,
)


@pytest.mark.parametrize("step", [0.1, 0.3, 0.6, 0.9])
def test_plain_coordinate_gradient_expands_a_rotational_game(step: float) -> None:
    factors = rotational_coordinate_factors(step)
    assert factors.plain > 1.0
    assert factors.optimistic < 1.0


@pytest.mark.parametrize("step", [0.1, 0.3, 0.6, 0.9])
def test_optimism_is_strictly_wasteful_in_a_potential_phase(step: float) -> None:
    factors = potential_coordinate_factors(step)
    assert factors.plain < factors.optimistic < 1.0


@pytest.mark.parametrize("step", [0.1, 0.3, 0.6, 0.9])
def test_exact_rotational_phase_boundary(step: float) -> None:
    threshold = rotational_optimism_threshold(step)
    assert threshold == pytest.approx(1.0 / (2.0 - step * step))
    factors = rotational_coordinate_factors(step)
    assert randomized_factor(factors, threshold - 1e-6) > 1.0
    assert randomized_factor(factors, threshold + 1e-6) < 1.0


def test_clocked_controller_uses_optimism_only_for_positive_drift_value() -> None:
    rotation = choose_clocked_optimism(
        energy=1.0,
        factors=rotational_coordinate_factors(0.5),
        resource_debt=0.0,
        average_optimism_budget=0.25,
        lyapunov_tradeoff=10.0,
    )
    potential = choose_clocked_optimism(
        energy=1.0,
        factors=potential_coordinate_factors(0.5),
        resource_debt=0.0,
        average_optimism_budget=0.25,
        lyapunov_tradeoff=10.0,
    )
    assert rotation.use_optimism
    assert not potential.use_optimism


def test_resource_debt_prevents_unbudgeted_always_optimistic_behavior() -> None:
    factors = rotational_coordinate_factors(0.4)
    debt = 0.0
    uses = []
    energy = 1.0
    for _ in range(200):
        decision = choose_clocked_optimism(
            energy=energy,
            factors=factors,
            resource_debt=debt,
            average_optimism_budget=0.25,
            lyapunov_tradeoff=5.0,
        )
        uses.append(decision.use_optimism)
        debt = decision.resource_debt_after
        # Hold energy fixed here to isolate virtual-queue accounting.
    assert np.mean(uses) <= 0.25 + 1.0 / len(uses)


def test_invalid_normalized_step_is_rejected() -> None:
    for step in (0.0, 1.0, np.inf):
        with pytest.raises(ValueError):
            rotational_coordinate_factors(step)


@pytest.mark.parametrize("arrival", [0.05, 0.2, 0.5, 0.8, 0.95])
@pytest.mark.parametrize("step", [0.2, 0.5, 0.8])
def test_clock_balanced_metric_preserves_rotational_phase_boundary(
    arrival: float, step: float
) -> None:
    threshold = rotational_optimism_threshold(step)
    assert heterogeneous_rotational_drift_coefficient(
        step,
        first_agent_probability=arrival,
        optimism_probability=threshold - 1e-8,
    ) > 0.0
    assert heterogeneous_rotational_drift_coefficient(
        step,
        first_agent_probability=arrival,
        optimism_probability=threshold + 1e-8,
    ) < 0.0


@pytest.mark.parametrize("arrival", [0.05, 0.2, 0.5, 0.8, 0.95])
def test_potential_plain_update_has_more_negative_clock_balanced_drift(
    arrival: float,
) -> None:
    plain = heterogeneous_potential_drift_coefficient(
        0.4, first_agent_probability=arrival, use_optimism=False
    )
    optimistic = heterogeneous_potential_drift_coefficient(
        0.4, first_agent_probability=arrival, use_optimism=True
    )
    assert plain < optimistic < 0.0


def test_clock_metric_penalizes_the_rare_agent_coordinate() -> None:
    first, second = heterogeneous_clock_metric(0.1)
    assert first / second == pytest.approx(9.0)


@pytest.mark.parametrize("arrival", [0.1, 0.37, 0.8])
@pytest.mark.parametrize("step", [0.2, 0.7])
@pytest.mark.parametrize("optimism_probability", [0.0, 0.3, 1.0])
def test_heterogeneous_rotational_matrix_identity(
    arrival: float, step: float, optimism_probability: float
) -> None:
    rotation = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    metric = np.diag(heterogeneous_clock_metric(arrival))
    expected_matrices = []
    for use_optimism in (False, True):
        expected = np.zeros((2, 2))
        for probability, agent in ((arrival, 0), (1.0 - arrival, 1)):
            selector = np.zeros((2, 2))
            selector[agent, agent] = 1.0
            matrix = np.eye(2) - step * selector @ rotation
            if use_optimism:
                matrix = (
                    np.eye(2)
                    - step
                    * selector
                    @ rotation
                    @ (np.eye(2) - step * rotation)
                )
            expected += probability * matrix.T @ metric @ matrix
        expected_matrices.append(expected)
    mixed = (
        (1.0 - optimism_probability) * expected_matrices[0]
        + optimism_probability * expected_matrices[1]
    )
    coefficient = heterogeneous_rotational_drift_coefficient(
        step,
        first_agent_probability=arrival,
        optimism_probability=optimism_probability,
    )
    np.testing.assert_allclose(mixed - metric, coefficient * np.eye(2), atol=1e-14)


def test_delay_zero_lifted_radius_matches_closed_form_boundary() -> None:
    step = 0.4
    threshold = rotational_optimism_threshold(step)
    assert lifted_mean_square_spectral_radius(
        step,
        delay=0,
        first_agent_probability=0.5,
        fresh_optimism_probability=threshold - 1e-5,
    ) > 1.0
    assert lifted_mean_square_spectral_radius(
        step,
        delay=0,
        first_agent_probability=0.5,
        fresh_optimism_probability=threshold + 1e-5,
    ) < 1.0


@pytest.mark.parametrize("delay", [1, 2, 4])
def test_fully_stale_extragradient_does_not_stabilize_delay(delay: int) -> None:
    assert stale_optimistic_lifted_spectral_radius(
        0.3, delay=delay, first_agent_probability=0.5
    ) > 1.0


@pytest.mark.parametrize("delay", [1, 2, 4])
def test_fully_fresh_anchor_stabilizes_registered_delay_smoke(delay: int) -> None:
    assert lifted_mean_square_spectral_radius(
        0.3,
        delay=delay,
        first_agent_probability=0.5,
        fresh_optimism_probability=1.0,
    ) < 1.0


def test_lifted_transition_has_exact_shift_register() -> None:
    transition = lifted_rotational_transition(
        0.2, delay=3, agent=0, fresh_optimistic_anchor=False
    )
    assert transition.shape == (8, 8)
    np.testing.assert_array_equal(transition[2:8, :6], np.eye(6))
