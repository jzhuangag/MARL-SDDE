from itertools import product

import numpy as np
import pytest

from experiments.dependence_delay_linear.t035_scalar_phase_theorem import (
    exact_scalar_risk,
)
from experiments.dependence_delay_linear.t036_affine_markov_moments import (
    delayed_scalar_mode_updates,
    propagate_affine_markov_moments,
    symmetric_ar_sign_chain,
    validate_markov_transition,
)


def test_transition_validation() -> None:
    validate_markov_transition(np.array([[0.7, 0.3], [0.2, 0.8]]))
    with pytest.raises(ValueError):
        validate_markov_transition(np.array([[0.8, 0.3], [0.2, 0.8]]))


def test_mode_conditioned_recursion_matches_path_enumeration() -> None:
    transition = np.array([[0.7, 0.3], [0.2, 0.8]])
    updates = delayed_scalar_mode_updates(
        innovations=np.array([-1.0, 2.0]), mu=0.5, step_size=0.2, delay=1
    )
    initial_probability = np.array([0.4, 0.6])
    initial = np.array([1.5, 1.5])
    steps = 3
    result = propagate_affine_markov_moments(
        transition=transition,
        augmented_updates=updates,
        initial_mode_probability=initial_probability,
        initial_state=initial,
        steps=steps,
    )
    expected_first = np.zeros(2)
    expected_second = np.zeros((2, 2))
    expected_mode = np.zeros(2)
    for path in product(range(2), repeat=steps + 1):
        probability = initial_probability[path[0]]
        state = np.append(initial, 1.0)
        for time in range(steps):
            probability *= transition[path[time], path[time + 1]]
            state = updates[path[time], path[time + 1]] @ state
        expected_first += probability * state[:-1]
        expected_second += probability * np.outer(state[:-1], state[:-1])
        expected_mode[path[-1]] += probability
    np.testing.assert_allclose(result["mean"], expected_first)
    np.testing.assert_allclose(result["second_moment"], expected_second)
    np.testing.assert_allclose(result["mode_probability"], expected_mode)


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_two_state_markov_jump_matches_scalar_ar_covariance(delay: int) -> None:
    markov_lambda = 0.7
    variance = 1.8
    steps = 14
    innovations = np.sqrt(variance) * np.array([-1.0, 1.0])
    result = propagate_affine_markov_moments(
        transition=symmetric_ar_sign_chain(markov_lambda),
        augmented_updates=delayed_scalar_mode_updates(
            innovations=innovations, mu=0.8, step_size=0.12, delay=delay
        ),
        initial_mode_probability=np.array([0.5, 0.5]),
        initial_state=np.full(delay + 1, 1.3),
        steps=steps,
    )
    scalar = exact_scalar_risk(
        initial_error=1.3,
        mu=0.8,
        step_size=0.12,
        delay=delay,
        updates=steps,
        single_variance=variance,
        q=1,
        rho=0.0,
        markov_lambda=markov_lambda,
    )
    assert result["second_moment"][0, 0] == pytest.approx(
        scalar["risk"], abs=1e-11
    )


def test_current_mode_and_iterate_are_not_declared_independent() -> None:
    transition = np.array([[0.95, 0.05], [0.05, 0.95]])
    updates = delayed_scalar_mode_updates(
        innovations=np.array([-1.0, 1.0]), mu=1.0, step_size=0.1, delay=0
    )
    result = propagate_affine_markov_moments(
        transition=transition,
        augmented_updates=updates,
        initial_mode_probability=np.array([0.5, 0.5]),
        initial_state=np.array([0.0]),
        steps=5,
    )
    conditional_means = result["mode_first"][:, 0] / result["mode_probability"]
    assert conditional_means[0] < 0.0 < conditional_means[1]
    assert not np.allclose(conditional_means, result["mean"][0])


def test_augmented_coordinate_remains_probability_mass() -> None:
    transition = symmetric_ar_sign_chain(0.4)
    result = propagate_affine_markov_moments(
        transition=transition,
        augmented_updates=delayed_scalar_mode_updates(
            innovations=np.array([-1.0, 1.0]), mu=1.0, step_size=0.1, delay=2
        ),
        initial_mode_probability=np.array([0.5, 0.5]),
        initial_state=np.ones(3),
        steps=20,
    )
    assert result["mode_probability"].sum() == pytest.approx(1.0)
    assert np.min(np.linalg.eigvalsh(result["covariance"])) >= -1e-12
