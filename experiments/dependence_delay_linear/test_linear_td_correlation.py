"""Implementation tests for EXP-007A."""

import numpy as np

from linear_td_correlation import (
    LinearTDConfig,
    build_mrp,
    effective_participation_rows,
    generate_base_paths,
    observed_transition_pairs,
    simulate_td_budget,
    simulate_td_eta_grid,
    td_noise_gradients,
)


def test_registered_mrp_is_stochastic_and_features_are_orthonormal() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    assert np.allclose(mrp["transition"].sum(axis=1), 1.0)
    assert np.all(mrp["transition"] >= 0.0)
    weighted_gram = (
        mrp["features"].T
        @ np.diag(mrp["stationary"])
        @ mrp["features"]
    )
    assert np.allclose(weighted_gram, np.eye(config.num_features))
    assert np.allclose(
        mrp["a_matrix"] @ mrp["theta_star"], mrp["b_vector"]
    )


def test_full_correlation_shares_every_transition_pair() -> None:
    config = LinearTDConfig(max_budget=1000)
    mrp = build_mrp(config)
    paths = generate_base_paths(20260930, mrp, config)
    current, following = observed_transition_pairs(paths, 1.0)
    assert np.all(current == current[0])
    assert np.all(following == following[0])


def test_full_correlation_has_unit_effective_participation() -> None:
    config = LinearTDConfig(max_budget=2000, lrv_batch_size=20)
    mrp = build_mrp(config)
    paths = generate_base_paths(20260931, mrp, config)
    current, following = observed_transition_pairs(paths, 1.0)
    gradients = td_noise_gradients(current, following, mrp, config)
    rows = effective_participation_rows(
        gradients, 1.0, 20260931, config
    )
    assert all(
        np.isclose(row["effective_participation"], 1.0) for row in rows
    )


def test_td_budget_accounting_and_finiteness() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    paths = generate_base_paths(20260932, mrp, config)
    current, following = observed_transition_pairs(paths, 0.5)
    result = simulate_td_budget(
        current,
        following,
        mrp,
        max_delay=8,
        num_agents=4,
        eta=0.01,
        config=config,
    )
    assert np.all(result["finite"])
    assert np.all(result["charged_budgets"] <= np.array([2000, 8000, 32000]))
    assert np.all(result["updates"] * 8 == result["charged_budgets"])
    assert np.all(result["errors"] >= 0.0)


def test_batched_eta_grid_matches_scalar_kernel() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    paths = generate_base_paths(20260933, mrp, config)
    current, following = observed_transition_pairs(paths, 0.25)
    batched = simulate_td_eta_grid(
        current,
        following,
        mrp,
        max_delay=8,
        num_agents=8,
        config=config,
    )
    eta_index = 5
    scalar = simulate_td_budget(
        current,
        following,
        mrp,
        max_delay=8,
        num_agents=8,
        eta=float(config.eta_grid[eta_index]),
        config=config,
    )
    assert np.array_equal(
        batched["updates"][eta_index], scalar["updates"]
    )
    assert np.array_equal(
        batched["charged_budgets"][eta_index],
        scalar["charged_budgets"],
    )
    assert np.allclose(
        batched["errors"][eta_index], scalar["errors"], rtol=0.0, atol=0.0
    )
