from itertools import product

import numpy as np
import pytest

from experiments.dependence_delay_linear.t042_poisson_td_remainder import (
    pathwise_poisson_decomposition,
    pathwise_weighted_poisson_decomposition,
    poisson_moment_constants,
    solve_centered_poisson,
    stationary_distribution,
)


def test_stationary_distribution_and_nonergodic_rejection() -> None:
    transition = np.array([[0.8, 0.2], [0.1, 0.9]])
    stationary = stationary_distribution(transition)
    np.testing.assert_allclose(stationary, np.array([1.0 / 3.0, 2.0 / 3.0]))
    with pytest.raises(ValueError):
        stationary_distribution(np.eye(2))


def test_matrix_poisson_solution_has_zero_residual_and_mean() -> None:
    transition = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
    stationary = stationary_distribution(transition)
    raw = np.array(
        [
            [[0.4, -0.2], [0.1, 0.3]],
            [[-0.1, 0.5], [0.2, -0.4]],
            [[0.3, 0.1], [-0.2, 0.2]],
        ]
    )
    field = raw - np.tensordot(stationary, raw, axes=(0, 0))[None, :, :]
    result = solve_centered_poisson(transition=transition, field=field)
    solution = result["solution"]
    predicted = np.tensordot(transition, solution, axes=(1, 0))
    np.testing.assert_allclose(solution - predicted, field, atol=1e-12)
    np.testing.assert_allclose(
        np.tensordot(stationary, solution, axes=(0, 0)), 0.0, atol=1e-12
    )


def test_uncentered_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="centered"):
        solve_centered_poisson(
            transition=np.array([[0.75, 0.25], [0.25, 0.75]]),
            field=np.array([1.0, 2.0]),
        )


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_pathwise_matrix_decomposition_with_delayed_iterates(delay: int) -> None:
    transition = np.array([[0.85, 0.15], [0.25, 0.75]])
    stationary = stationary_distribution(transition)
    base = np.array(
        [
            [[0.4, 0.1], [-0.2, 0.3]],
            [[-0.3, 0.2], [0.5, -0.1]],
        ]
    )
    field = base - np.tensordot(stationary, base, axes=(0, 0))[None, :, :]
    solution = solve_centered_poisson(transition=transition, field=field)[
        "solution"
    ]
    path = np.array([0, 0, 1, 1, 0, 1, 0])
    iterate = np.array(
        [[1.0, -0.4], [0.8, -0.1], [0.5, 0.2], [0.1, 0.3], [-0.2, 0.4], [-0.3, 0.2]]
    )
    history = np.vstack((np.repeat(iterate[[0]], delay, axis=0), iterate))
    delayed = history[: path.size - 1]
    parts = pathwise_poisson_decomposition(
        transition=transition,
        field=field,
        poisson_solution=solution,
        mode_path=path,
        predictable_vectors=delayed,
    )
    np.testing.assert_allclose(parts["target"], parts["reconstructed"], atol=1e-12)
    np.testing.assert_allclose(parts["residual"], 0.0, atol=1e-12)


def test_martingale_isometry_for_history_dependent_predictable_vectors() -> None:
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    stationary = stationary_distribution(transition)
    raw = np.array([[[0.6, -0.2], [0.1, 0.4]], [[-0.3, 0.5], [0.2, -0.1]]])
    field = raw - np.tensordot(stationary, raw, axes=(0, 0))[None, :, :]
    solution = solve_centered_poisson(transition=transition, field=field)["solution"]
    predicted = np.tensordot(transition, solution, axes=(1, 0))
    horizon = 4
    mean_martingale = np.zeros(2)
    martingale_second = 0.0
    predictable_variance = 0.0
    for path_tuple in product(range(2), repeat=horizon + 1):
        path = np.asarray(path_tuple)
        probability = stationary[path[0]]
        for time in range(horizon):
            probability *= transition[path[time], path[time + 1]]
        vectors = np.asarray(
            [
                [1.0 + 0.2 * path[max(time - 1, 0)], -0.3 + 0.1 * time]
                for time in range(horizon)
            ]
        )
        parts = pathwise_poisson_decomposition(
            transition=transition,
            field=field,
            poisson_solution=solution,
            mode_path=path,
            predictable_vectors=vectors,
        )
        mean_martingale += probability * parts["martingale"]
        martingale_second += probability * float(parts["martingale"] @ parts["martingale"])
        for time in range(horizon):
            source = path[time]
            conditional = 0.0
            for target in range(2):
                delta = solution[target] - predicted[source]
                transformed = delta @ vectors[time]
                conditional += transition[source, target] * float(transformed @ transformed)
            predictable_variance += probability * conditional
    np.testing.assert_allclose(mean_martingale, 0.0, atol=1e-12)
    assert martingale_second == pytest.approx(predictable_variance, abs=1e-12)


def test_reported_variance_constant_bounds_every_mode() -> None:
    transition = np.array([[0.65, 0.35], [0.2, 0.8]])
    stationary = stationary_distribution(transition)
    raw = np.array([[[0.2, 0.7], [-0.3, 0.1]], [[-0.5, 0.2], [0.4, -0.2]]])
    field = raw - np.tensordot(stationary, raw, axes=(0, 0))[None, :, :]
    solution = solve_centered_poisson(transition=transition, field=field)["solution"]
    constants = poisson_moment_constants(
        transition=transition, poisson_solution=solution
    )
    for matrix in constants["variance_operator"]:
        assert (
            np.max(np.linalg.eigvalsh(matrix))
            <= constants["variance_constant"] + 1e-12
        )
    assert constants["h_max"] >= max(np.linalg.norm(x, ord=2) for x in solution)


def test_weighted_terminal_response_decomposition_is_pathwise_exact() -> None:
    transition = np.array([[0.72, 0.28], [0.18, 0.82]])
    stationary = stationary_distribution(transition)
    raw = np.array([[[0.5, -0.1], [0.2, 0.3]], [[-0.2, 0.4], [-0.3, 0.1]]])
    field = raw - np.tensordot(stationary, raw, axes=(0, 0))[None, :, :]
    solution = solve_centered_poisson(transition=transition, field=field)["solution"]
    path = np.array([1, 1, 0, 1, 0, 0])
    vectors = np.array(
        [[0.7, -0.2], [0.7, -0.2], [0.4, 0.0], [0.1, 0.3], [-0.2, 0.2]]
    )
    stable = np.array([[0.82, 0.05], [0.0, 0.76]])
    weights = np.asarray(
        [np.linalg.matrix_power(stable, path.size - 2 - time) for time in range(path.size - 1)]
    )
    parts = pathwise_weighted_poisson_decomposition(
        transition=transition,
        field=field,
        poisson_solution=solution,
        mode_path=path,
        predictable_vectors=vectors,
        left_weights=weights,
    )
    np.testing.assert_allclose(parts["target"], parts["reconstructed"], atol=1e-12)
    np.testing.assert_allclose(parts["residual"], 0.0, atol=1e-12)


def test_scalar_poisson_constants_are_supported() -> None:
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    stationary = stationary_distribution(transition)
    field = np.array([1.0, -stationary[0] / stationary[1]])
    solution = solve_centered_poisson(transition=transition, field=field)["solution"]
    constants = poisson_moment_constants(
        transition=transition, poisson_solution=solution
    )
    assert constants["h_max"] > 0.0
    assert constants["variance_constant"] > 0.0
