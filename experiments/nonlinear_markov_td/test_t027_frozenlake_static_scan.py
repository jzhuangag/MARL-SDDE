import numpy as np

from experiments.nonlinear_markov_td.t027_frozenlake_static_scan import (
    CONFIG,
    config_sha256,
    continuing_transition_matrix,
    message_cost,
    stationary_distribution,
    usable_horizon,
    variance_factor,
)


def test_config_hash_is_stable() -> None:
    assert config_sha256() == "f9251599e0382309e5d08d115bf04def6feffb00bff2109de548543643269442"


def test_exact_transition_is_stochastic_and_regenerative() -> None:
    transition, start, terminals = continuing_transition_matrix()
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-12)
    assert transition.shape == (64, 64)
    assert len(terminals) > 0
    for state in terminals:
        assert transition[state, start] == 1.0
        assert np.count_nonzero(transition[state]) == 1


def test_stationary_distribution_is_valid() -> None:
    transition, _, _ = continuing_transition_matrix()
    stationary = stationary_distribution(transition)
    np.testing.assert_allclose(stationary @ transition, stationary, atol=1e-10)
    np.testing.assert_allclose(stationary.sum(), 1.0, atol=1e-12)
    assert np.all(stationary >= 0.0)


def test_hidden_share_preserves_marginal_variance_factor() -> None:
    for rho in CONFIG["rho_values"]:
        assert variance_factor(1, rho) == 1.0
        values = [variance_factor(q, rho) for q in CONFIG["q_values"]]
        assert all(left >= right for left, right in zip(values, values[1:]))


def test_message_cost_and_horizon_charge_full_participation() -> None:
    assert message_cost(32) > message_cost(16) > message_cost(4) > message_cost(1)
    stride = 10
    h1, *_ = usable_horizon(1, 512, "message", 0.0, stride)
    h32, *_ = usable_horizon(32, 512, "message", 0.0, stride)
    assert h1 > h32
    e1, *_ = usable_horizon(1, 512, "environment", 0.0, stride)
    e32, *_ = usable_horizon(32, 512, "environment", 0.0, stride)
    assert e1 == e32 == 512


def test_delay_reduces_usable_horizon() -> None:
    stride = 10
    fresh, *_ = usable_horizon(4, 512, "message", 0.0, stride)
    delayed, *_ = usable_horizon(4, 512, "message", 0.2, stride)
    assert delayed < fresh
