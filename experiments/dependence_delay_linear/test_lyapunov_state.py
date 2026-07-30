"""Implementation tests for EXP-006C."""

import numpy as np
import pytest

from linear_model import make_agent_delays
from lyapunov_state import (
    LyapunovStateConfig,
    propagate_lyapunov_surrogate,
    simulate_lyapunov_state_policy,
)
from online_participation import FiniteBudgetProxyCache, generate_factor_paths
from state_correlation import build_noise_table_components


def small_config() -> LyapunovStateConfig:
    return LyapunovStateConfig(
        total_budget=800,
        checkpoint_count=9,
        block_budget=200,
        num_blocks=4,
        probe_updates_per_block=2,
        rolling_probe_vectors=8,
    )


def test_surrogate_risk_increases_with_lrv() -> None:
    config = small_config()
    delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=4,
        exponent=config.delay_exponent,
    )
    cache = FiniteBudgetProxyCache(config)
    low = propagate_lyapunov_surrogate(
        0.09, 8, 0.02, 5, 0.1, delays, cache
    )
    high = propagate_lyapunov_surrogate(
        0.09, 8, 0.02, 5, 10.0, delays, cache
    )
    assert np.isfinite(low)
    assert high > low


def test_lyapunov_surrogate_is_recursive_and_fully_charged() -> None:
    config = small_config()
    paths = generate_factor_paths(
        seed=20260830, maximum_delay=4, config=config
    )
    noise = build_noise_table_components(
        rho_global=0.4,
        rho_cluster=0.4,
        max_delay=4,
        paths=paths,
        config=config,
    )
    result = simulate_lyapunov_state_policy(
        policy="lyapunov_state_adaptive",
        scenario="balanced_08",
        max_delay=4,
        noise_table=noise,
        config=config,
    )
    actions = result["actions"]
    assert len(actions) == config.num_blocks
    assert actions[0]["lyapunov_surrogate_before"] == pytest.approx(
        config.initial_error**2
    )
    for previous, current in zip(actions[:-1], actions[1:]):
        assert current["lyapunov_surrogate_before"] == pytest.approx(
            previous["lyapunov_surrogate_after_block"]
        )
    assert result["total_probe_cost"] == config.probe_cost
    assert result["charged_budget"] <= config.total_budget
    assert result["within_budget"]
    assert result["finite"]


def test_true_error_is_audit_only_for_lyapunov_decision() -> None:
    config = small_config()
    paths = generate_factor_paths(
        seed=20260831, maximum_delay=4, config=config
    )
    noise = build_noise_table_components(
        rho_global=0.8,
        rho_cluster=0.0,
        max_delay=4,
        paths=paths,
        config=config,
    )
    result = simulate_lyapunov_state_policy(
        policy="lyapunov_state_adaptive",
        scenario="global_08",
        max_delay=4,
        noise_table=noise,
        config=config,
    )
    for action in result["actions"]:
        assert action["decision_state_proxy"] == pytest.approx(
            np.sqrt(action["lyapunov_surrogate_after_probe"])
        )


def test_unknown_policy_is_rejected() -> None:
    config = small_config()
    noise = np.zeros((config.num_agents, config.maximum_updates))
    with pytest.raises(ValueError, match="unknown policy"):
        simulate_lyapunov_state_policy(
            policy="not_registered",
            scenario="independent",
            max_delay=4,
            noise_table=noise,
            config=config,
        )
