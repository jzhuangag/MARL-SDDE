import numpy as np
import pytest

from experiments.dependence_delay_linear.t033_factor_correction_audit import (
    aggregation_risk,
    cancellation_identifiable,
    delay_expanded_loadings,
    delayed_scalar_spectral_radius,
    factor_cancel_weights,
    low_rank_gls_weights,
    run_audit,
)


def test_cancellation_identifiability_boundary() -> None:
    assert cancellation_identifiable(np.array([0.5, 1.5]))
    assert not cancellation_identifiable(np.ones(4))


def test_factor_cancel_constraints() -> None:
    loading = np.array([0.5, 1.5, 2.0])
    weight = factor_cancel_weights(loading, np.array([1.0, 2.0, 3.0]))
    assert np.sum(weight) == pytest.approx(1.0)
    assert weight @ loading == pytest.approx(0.0, abs=1e-12)


def test_low_rank_gls_matches_dense_solve() -> None:
    loading = np.array([[0.5], [1.5], [-0.2]])
    variance = np.array([1.0, 2.0, 0.7])
    omega = np.array([[3.0]])
    low_rank = low_rank_gls_weights(loading, variance, omega)
    covariance = np.diag(variance) + loading @ omega @ loading.T
    dense = np.linalg.solve(covariance, np.ones(3))
    dense /= np.sum(dense)
    np.testing.assert_allclose(low_rank, dense, atol=1e-12)


def test_gls_dominates_exact_cancellation() -> None:
    loading = np.array([0.5, 1.5])
    variance = np.ones(2)
    omega = np.array([[10.0]])
    cancel = factor_cancel_weights(loading, variance)
    gls = low_rank_gls_weights(loading, variance, omega)
    assert aggregation_risk(gls, loading, variance, omega) <= aggregation_risk(
        cancel, loading, variance, omega
    )


def test_delay_cohorts_destroy_two_agent_pathwise_cancellation() -> None:
    loading = np.array([1.0, 2.0])
    assert cancellation_identifiable(delay_expanded_loadings(loading, np.array([0, 0])))
    assert not cancellation_identifiable(
        delay_expanded_loadings(loading, np.array([0, 1]))
    )


def test_negative_weights_can_shrink_delay_stability_region() -> None:
    weight = factor_cancel_weights(np.array([1.0, 2.0]), np.ones(2))
    assert delayed_scalar_spectral_radius(0.9, weight, np.array([0, 0])) < 1.0
    assert delayed_scalar_spectral_radius(0.9, weight, np.array([0, 1])) > 1.0


def test_audit_is_outcome_free_and_internally_consistent() -> None:
    result = run_audit()
    assert result["scientific_trajectories"] == 0
    assert result["known_factor"]["gls_weakly_dominates"]
    assert not result["identifiability"]["equal_loading_feasible"]
    assert not result["delay"]["different_time_pathwise_feasible"]
