import numpy as np

from experiments.nonlinear_markov_td.t030_weighted_msve_audit import (
    AUDIT_CONFIG,
    config_sha256,
    run_audit,
    transformed_second_jacobian,
)
from experiments.nonlinear_markov_td.run_exp019a_blackjack_pilot import exact_value
from experiments.nonlinear_markov_td.t029_blackjack_static_scan import (
    continuing_transition_matrix,
)


def test_config_hash_is_frozen() -> None:
    assert config_sha256() == (
        "84cc7fde97b1dea73cf23d579f55734fcb6237979451ff1b4cc8009e463d4ebf"
    )


def test_theta_zero_recovers_euclidean_constants() -> None:
    _value, stationary, states, _reset = exact_value()
    transition = continuing_transition_matrix()[0]
    diagonal, lipschitz = transformed_second_jacobian(
        transition, stationary, np.ones(len(states)), 0.99
    )
    assert diagonal.shape == (280, 280)
    assert np.allclose(diagonal, diagonal.T, atol=1e-12)
    assert np.min(np.diag(diagonal)) >= 0.0
    assert lipschitz >= 1.0


def test_audit_is_finite_and_complete() -> None:
    result = run_audit()
    assert len(result["metrics"]) == len(AUDIT_CONFIG["theta_values"])
    for row in result["metrics"]:
        assert np.isfinite(row["monotonicity"])
        assert np.isfinite(row["minimum_curvature"])
        assert row["minimum_curvature"] > 0.0
        assert 0.0 <= row["optimistic_maximum_improvement"] <= 1.0
