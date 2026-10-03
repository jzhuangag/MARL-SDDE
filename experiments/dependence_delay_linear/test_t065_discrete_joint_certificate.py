import numpy as np
import pytest

from experiments.dependence_delay_linear.t065_discrete_joint_certificate import (
    ar1_block_mean_variance_factor,
    audit_certificate,
    common_quadratic_certificate,
    delayed_companion_matrix,
    paired_residual_statistics,
)


def _stationary_ar1_block_mean(rng, correlation, length, dimension):
    state = rng.normal(size=dimension)
    samples = []
    innovation_scale = np.sqrt(1.0 - correlation**2)
    for _ in range(length):
        state = correlation * state + innovation_scale * rng.normal(size=dimension)
        samples.append(state.copy())
    return np.mean(samples, axis=0)


def test_ar1_block_variance_formula_matches_monte_carlo():
    rng = np.random.default_rng(65001)
    correlation = 0.7
    length = 12
    draws = np.asarray(
        [_stationary_ar1_block_mean(rng, correlation, length, 1)[0] for _ in range(40_000)]
    )
    expected = ar1_block_mean_variance_factor(correlation, length)
    assert np.var(draws) == pytest.approx(expected, rel=0.035)


def test_paired_markov_blocks_separate_signal_and_noise():
    rng = np.random.default_rng(65002)
    mean_field = np.asarray([0.4, -0.7, 0.2])
    correlation = 0.6
    length = 16
    statistics = []
    for _ in range(30_000):
        first = mean_field + _stationary_ar1_block_mean(
            rng, correlation, length, mean_field.size
        )
        second = mean_field + _stationary_ar1_block_mean(
            rng, correlation, length, mean_field.size
        )
        statistics.append(paired_residual_statistics(first, second))
    expected_signal = float(mean_field @ mean_field)
    expected_noise = mean_field.size * ar1_block_mean_variance_factor(
        correlation, length
    )
    assert np.mean([row.signal for row in statistics]) == pytest.approx(
        expected_signal, abs=0.012
    )
    assert np.mean([row.noise for row in statistics]) == pytest.approx(
        expected_noise, rel=0.025
    )


def test_delayed_companion_matches_explicit_scalar_update():
    companion = delayed_companion_matrix(np.asarray([[2.0]]), 0.1, delay=2)
    lifted = np.asarray([3.0, 2.0, 1.0])
    assert companion @ lifted == pytest.approx([2.8, 3.0, 2.0])


@pytest.mark.parametrize("delay", [0, 1, 2, 4, 8])
def test_common_certificate_covers_full_gain_interval(delay):
    drift = np.diag([0.5, 1.0])
    certificate = common_quadratic_certificate(
        drift, eta_min=0.005, eta_max=0.02, delay=delay
    )
    audit = audit_certificate(drift, certificate, grid_size=151)
    assert certificate.margin > 1e-5
    assert audit["smallest_p_eigenvalue"] > 0.0
    assert audit["worst_drift_eigenvalue"] < -0.9 * certificate.margin
    assert audit["worst_spectral_radius"] < 1.0


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="same-length"):
        paired_residual_statistics(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="strictly"):
        ar1_block_mean_variance_factor(1.0, 4)
    with pytest.raises(ValueError, match="square"):
        delayed_companion_matrix(np.ones((2, 3)), 0.1, 1)
