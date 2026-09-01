from __future__ import annotations

import numpy as np
import pytest

from .finite_time_drift import (
    expected_biased_noisy_quadratic_lyapunov_step,
    expected_noisy_quadratic_lyapunov_step,
    expected_quadratic_lyapunov_step,
    expected_rate_balanced_quadratic_lyapunov_step,
    interaction_history_weights,
    maximum_constant_step,
    rate_balanced_steps,
    weighted_history_energy,
)


def test_interaction_history_weights_match_definition() -> None:
    matrix = np.asarray([[2.0, 0.5], [1.0, 3.0]])
    probabilities = np.asarray([0.25, 0.75])
    expected = sum(
        probabilities[i]*np.sum(matrix[i])*matrix[i]
        for i in range(2)
    )
    assert interaction_history_weights(matrix, probabilities) == pytest.approx(
        expected
    )


def test_maximum_step_saturates_a_block_condition() -> None:
    matrix = np.asarray([[2.0, 0.4], [0.4, 1.5]])
    probabilities = np.asarray([0.35, 0.65])
    delay = 3
    weights = interaction_history_weights(matrix, probabilities)
    alpha = maximum_constant_step(matrix, probabilities, delay)
    conditions = np.diag(matrix)*alpha+delay**2*weights*alpha**2
    assert float(np.max(conditions)) == pytest.approx(1.0, abs=1e-12)
    assert (conditions <= 1.0+1e-12).all()


def test_zero_delay_recovers_block_smooth_step() -> None:
    matrix = np.asarray([[2.0, 0.4], [0.4, 1.5]])
    probabilities = np.asarray([0.35, 0.65])
    assert maximum_constant_step(matrix, probabilities, 0) == pytest.approx(0.5)


def test_history_inflation_tightens_maximum_step() -> None:
    matrix = np.asarray([[2.0, 0.4], [0.4, 1.5]])
    probabilities = np.asarray([0.35, 0.65])
    base = maximum_constant_step(matrix, probabilities, 3)
    inflated = maximum_constant_step(
        matrix, probabilities, 3, history_inflation=1.7
    )
    assert 0.0 < inflated < base


def test_rate_balanced_steps_equalize_expected_descent() -> None:
    matrix = np.asarray(
        [[2.0, 0.2, 0.1], [0.4, 1.5, 0.3], [0.1, 0.2, 2.5]]
    )
    probabilities = np.asarray([0.15, 0.35, 0.5])
    result = rate_balanced_steps(matrix, probabilities, maximum_delay=4)
    steps = np.asarray(result["step_sizes"])
    conditions = np.asarray(result["conditions"])
    scale = float(result["descent_scale"])
    assert probabilities*steps == pytest.approx(np.full(3, scale))
    assert float(np.max(conditions)) == pytest.approx(1.0, abs=1e-12)
    assert (conditions <= 1.0+1e-12).all()


def test_weighted_history_energy_uses_block_and_time_weights() -> None:
    steps = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    weights = np.asarray([0.5, 2.0])
    expected = 1.0*(0.5*1.0**2+2.0*2.0**2)+2.0*(0.5*3.0**2+2.0*4.0**2)
    assert weighted_history_energy(steps, weights) == pytest.approx(expected)


def test_expected_quadratic_drift_bound_on_random_paths() -> None:
    rng = np.random.default_rng(90417)
    for dimension in (2, 4, 7):
        raw = rng.normal(size=(dimension, dimension))
        curvature = raw.T@raw+0.25*np.eye(dimension)
        probabilities = rng.uniform(0.2, 1.0, size=dimension)
        probabilities /= np.sum(probabilities)
        delay = 4
        alpha = 0.9*maximum_constant_step(
            np.abs(curvature), probabilities, delay
        )
        for _ in range(20):
            path = np.cumsum(rng.normal(scale=0.15, size=(delay+1, dimension)), axis=0)
            delays = rng.integers(0, delay+1, size=dimension)
            result = expected_quadratic_lyapunov_step(
                path, delays, curvature, probabilities, alpha
            )
            assert result["expected_next"] <= result["certified_upper"]+1e-10
            assert result["slack"] >= -1e-10


def test_expected_noisy_quadratic_drift_bound_on_random_paths() -> None:
    rng = np.random.default_rng(44017)
    for dimension in (2, 5):
        raw = rng.normal(size=(dimension, dimension))
        curvature = raw.T@raw+0.4*np.eye(dimension)
        probabilities = rng.uniform(0.2, 1.0, size=dimension)
        probabilities /= np.sum(probabilities)
        delay = 3
        alpha = 0.8*maximum_constant_step(
            np.abs(curvature), probabilities, delay
        )
        sigma = rng.uniform(0.0, 0.6, size=dimension)
        for _ in range(20):
            path = np.cumsum(rng.normal(scale=0.1, size=(delay+1, dimension)), axis=0)
            delays = rng.integers(0, delay+1, size=dimension)
            result = expected_noisy_quadratic_lyapunov_step(
                path, delays, curvature, probabilities, alpha, sigma
            )
            assert result["expected_next"] <= result["certified_upper"]+1e-10
            assert result["variance_penalty"] >= 0.0


def test_zero_noise_extension_matches_exact_enumeration() -> None:
    curvature = np.asarray([[2.0, 0.5], [0.5, 1.5]])
    probabilities = np.asarray([0.4, 0.6])
    path = np.asarray([[0.1, 0.2], [0.2, 0.15], [0.3, -0.1]])
    delays = np.asarray([2, 1])
    alpha = 0.7*maximum_constant_step(
        np.abs(curvature), probabilities, maximum_delay=2
    )
    exact = expected_quadratic_lyapunov_step(
        path, delays, curvature, probabilities, alpha
    )
    noisy = expected_noisy_quadratic_lyapunov_step(
        path, delays, curvature, probabilities, alpha, np.zeros(2)
    )
    assert noisy["expected_next"] == pytest.approx(exact["expected_next"])
    assert noisy["certified_upper"] == pytest.approx(exact["certified_upper"])
    assert noisy["variance_penalty"] == pytest.approx(0.0)


def test_biased_noisy_quadratic_drift_bound_on_random_paths() -> None:
    rng = np.random.default_rng(71091)
    delta = 0.6
    for dimension in (2, 5):
        raw = rng.normal(size=(dimension, dimension))
        curvature = raw.T@raw+0.3*np.eye(dimension)
        probabilities = rng.uniform(0.2, 1.0, size=dimension)
        probabilities /= np.sum(probabilities)
        delay = 3
        alpha = 0.8*maximum_constant_step(
            np.abs(curvature),
            probabilities,
            delay,
            history_inflation=1.0+delta,
        )
        bias = rng.normal(scale=0.08, size=dimension)
        sigma = rng.uniform(0.0, 0.3, size=dimension)
        for _ in range(20):
            path = np.cumsum(
                rng.normal(scale=0.1, size=(delay+1, dimension)), axis=0
            )
            delays = rng.integers(0, delay+1, size=dimension)
            result = expected_biased_noisy_quadratic_lyapunov_step(
                path,
                delays,
                curvature,
                probabilities,
                alpha,
                bias,
                sigma,
                delta,
            )
            assert result["expected_next"] <= result["certified_upper"]+1e-10
            assert result["bias_penalty"] >= 0.0
            assert result["variance_penalty"] >= 0.0


def test_rate_balanced_quadratic_drift_bound_on_random_paths() -> None:
    rng = np.random.default_rng(71092)
    for dimension in (2, 4, 7):
        raw = rng.normal(size=(dimension, dimension))
        curvature = raw.T@raw+0.2*np.eye(dimension)
        probabilities = rng.uniform(0.1, 1.0, size=dimension)
        probabilities /= np.sum(probabilities)
        delay = 4
        allocation = rate_balanced_steps(
            np.abs(curvature), probabilities, delay, history_inflation=1.3
        )
        steps = 0.9*np.asarray(allocation["step_sizes"])
        for _ in range(20):
            path = np.cumsum(
                rng.normal(scale=0.1, size=(delay+1, dimension)), axis=0
            )
            delays = rng.integers(0, delay+1, size=dimension)
            result = expected_rate_balanced_quadratic_lyapunov_step(
                path,
                delays,
                curvature,
                probabilities,
                steps,
                history_inflation=1.3,
            )
            assert result["expected_next"] <= result["certified_upper"]+1e-10


def test_input_validation_is_strict() -> None:
    with pytest.raises(ValueError):
        maximum_constant_step(np.eye(2), np.asarray([0.4, 0.4]), 1)
    with pytest.raises(ValueError):
        weighted_history_energy(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError):
        expected_noisy_quadratic_lyapunov_step(
            np.zeros((2, 2)),
            np.zeros(2, dtype=int),
            np.eye(2),
            np.asarray([0.5, 0.5]),
            0.1,
            np.asarray([0.1, np.nan]),
        )
