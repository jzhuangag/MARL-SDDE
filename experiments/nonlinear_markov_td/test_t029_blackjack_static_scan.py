import numpy as np

from experiments.nonlinear_markov_td.t029_blackjack_static_scan import (
    CONFIG,
    add_card,
    card_probabilities,
    config_sha256,
    continuing_transition_matrix,
    hit_probability,
    message_cost,
    reset_distribution,
    stationary_distribution,
    usable_horizon,
    variance_factor,
)


def test_config_hash_is_frozen() -> None:
    assert config_sha256() == (
        "c2f0001a2144e93d3ed37983e3fc8baf702b0d1e114891e837f0ae6d9c289e7b"
    )


def test_card_and_reset_distributions_are_exactly_normalized() -> None:
    np.testing.assert_allclose(sum(card_probabilities().values()), 1.0, atol=1e-15)
    np.testing.assert_allclose(sum(reset_distribution().values()), 1.0, atol=1e-15)


def test_ace_update_matches_blackjack_semantics() -> None:
    assert add_card(20, True, 10) == (20, False)
    assert add_card(10, False, 1) == (21, True)
    assert add_card(20, False, 10) == (30, False)


def test_exact_transition_is_stochastic_and_reset_minorized() -> None:
    transition, states, reset = continuing_transition_matrix()
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-12)
    epsilon = float(CONFIG["policy_epsilon"])
    for row in transition:
        assert np.all(row + 1e-15 >= epsilon * reset)
    assert min(1.0 - hit_probability(state) for state in states) >= epsilon - 1e-15


def test_stationary_distribution_is_valid() -> None:
    transition, _, _ = continuing_transition_matrix()
    stationary = stationary_distribution(transition)
    np.testing.assert_allclose(stationary @ transition, stationary, atol=1e-10)
    np.testing.assert_allclose(stationary.sum(), 1.0, atol=1e-12)
    assert np.all(stationary >= 0.0)


def test_hidden_share_variance_factor_is_monotone_in_q() -> None:
    for rho in CONFIG["rho_values"]:
        values = [variance_factor(q, rho) for q in CONFIG["q_values"]]
        assert values[0] == 1.0
        assert all(left >= right for left, right in zip(values, values[1:]))


def test_message_cost_and_dual_budget_geometry() -> None:
    costs = [message_cost(q) for q in CONFIG["q_values"]]
    assert all(left < right for left, right in zip(costs, costs[1:]))
    stride = 10
    message_horizons = [
        usable_horizon(q, 512, "message", 0.0, stride)[0]
        for q in CONFIG["q_values"]
    ]
    assert message_horizons[0] > message_horizons[-1]
    environment_horizons = [
        usable_horizon(q, 512, "environment", 0.0, stride)[0]
        for q in CONFIG["q_values"]
    ]
    assert len(set(environment_horizons)) == 1


def test_delay_reduces_usable_horizon() -> None:
    stride = 10
    fresh = usable_horizon(4, 512, "message", 0.0, stride)[0]
    delayed = usable_horizon(4, 512, "message", 0.2, stride)[0]
    assert delayed < fresh
