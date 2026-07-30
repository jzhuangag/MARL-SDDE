"""Deterministic tests for EXP-006B observable control."""

import unittest

import numpy as np

from online_participation import generate_factor_paths
from state_correlation import (
    StateCorrelationConfig,
    build_noise_table_components,
    estimate_observable_components,
    observable_state_proxy,
    simulate_state_correlation_policy,
)


class StateCorrelationTest(unittest.TestCase):
    def test_probe_budget_is_exactly_4_8_percent(self) -> None:
        config = StateCorrelationConfig()
        self.assertEqual(config.probe_cost, 768)
        self.assertAlmostEqual(
            config.probe_cost / config.total_budget, 0.048
        )

    def test_observable_moment_estimator_detects_independent_noise(self) -> None:
        config = StateCorrelationConfig()
        rng = np.random.RandomState(123)
        gradients = [0.2 + rng.normal(size=8) for _ in range(2000)]
        estimate = estimate_observable_components(
            gradients, np.arange(8), config
        )
        self.assertGreater(estimate["rho_idiosyncratic"], 0.9)

    def test_state_proxy_is_clipped(self) -> None:
        config = StateCorrelationConfig()
        low = observable_state_proxy(
            [np.zeros(8) for _ in range(8)], config
        )
        high = observable_state_proxy(
            [np.full(8, 10.0) for _ in range(8)], config
        )
        self.assertEqual(low, config.state_proxy_min)
        self.assertEqual(high, config.state_proxy_max)

    def test_single_policy_run_is_finite_and_charged(self) -> None:
        config = StateCorrelationConfig()
        paths = generate_factor_paths(
            seed=20260730,
            maximum_delay=4,
            config=config,
        )
        noise = build_noise_table_components(
            rho_global=0.8,
            rho_cluster=0.0,
            max_delay=4,
            paths=paths,
            config=config,
        )
        result = simulate_state_correlation_policy(
            policy="state_correlation_adaptive",
            scenario="global_08",
            max_delay=4,
            noise_table=noise,
            config=config,
        )
        self.assertTrue(result["finite"])
        self.assertTrue(result["within_budget"])
        self.assertEqual(result["total_probe_cost"], config.probe_cost)


if __name__ == "__main__":
    unittest.main()
