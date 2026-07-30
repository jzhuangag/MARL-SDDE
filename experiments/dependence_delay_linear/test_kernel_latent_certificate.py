"""Tests for the bounded-kernel latent-sharing certificate."""

import numpy as np

from kernel_latent_certificate import (
    kernel_latent_rho_upper,
    lazy_joint_tv_upper,
    minimum_kernel_gap,
    periodic_rbf,
    periodic_rbf_independent_mean,
    sample_kernel_probe,
)


def test_periodic_kernel_is_bounded_and_translation_invariant():
    assert periodic_rbf(0.2, 0.2, 0.35) == 1.0
    assert np.isclose(
        periodic_rbf(0.1, 0.4, 0.35),
        periodic_rbf(0.6, 0.9, 0.35),
    )
    assert 0.0 < periodic_rbf(0.1, 0.4, 0.35) < 1.0


def test_periodic_kernel_baseline_matches_monte_carlo():
    rng = np.random.RandomState(17)
    first = rng.random_sample(200000)
    second = rng.random_sample(200000)
    values = np.asarray(
        [periodic_rbf(x, y, 0.35) for x, y in zip(first, second)]
    )
    assert abs(values.mean() - periodic_rbf_independent_mean(0.35)) < 0.003


def test_unknown_baseline_certificate_is_bounded():
    result = kernel_latent_rho_upper(
        similarity_sum=70.0,
        similarity_trials=100,
        similarity_bias_sum=1.0,
        control_sum=20.0,
        control_trials=100,
        control_bias_sum=1.0,
        alpha_similarity=0.003,
        alpha_control=0.003,
    )
    assert 0.0 <= result["baseline_lower"] <= 1.0
    assert 0.0 <= result["rho_upper"] <= 1.0


def test_lazy_gap_is_minimal():
    for persistence in (0.0, 0.8, 0.96):
        gap = minimum_kernel_gap(persistence, 0.01)
        assert lazy_joint_tv_upper(persistence, gap) <= 0.01
        if gap > 1:
            assert lazy_joint_tv_upper(persistence, gap - 1) > 0.01


def test_kernel_probe_stationary_similarity_identity():
    rng = np.random.RandomState(33)
    states = rng.random_sample(3)
    previous = None
    similarities = []
    controls = []
    for _ in range(50000):
        result = sample_kernel_probe(
            rng,
            states,
            persistence=0.8,
            rho=0.6,
            gap=30,
            lengthscale=0.35,
            previous_first=previous,
        )
        states = result["states"]
        previous = result["first"]
        similarities.append(result["similarity"])
        if result["control"] is not None:
            controls.append(result["control"])
    baseline = periodic_rbf_independent_mean(0.35)
    expected = baseline + (1.0 - baseline) * 0.6
    assert abs(np.mean(similarities) - expected) < 0.01
    assert abs(np.mean(controls) - baseline) < 0.01
