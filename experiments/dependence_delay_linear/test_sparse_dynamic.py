"""Deterministic tests for EXP-005C sparse dynamic control."""

import unittest

import numpy as np

from online_participation import build_noise_table, generate_factor_paths
from sparse_dynamic import (
    DynamicConfig,
    POLICIES,
    REGIME_SEQUENCE,
    estimate_dependence_components,
    regime_for_block,
    simulate_dynamic_policy,
)


class SparseDynamicTest(unittest.TestCase):
    def test_regime_sequence(self) -> None:
        config = DynamicConfig()
        observed = [regime_for_block(block, config) for block in [0, 4, 8, 12]]
        self.assertEqual(
            observed, ["independent", "clustered", "global", "mixed"]
        )

    def test_sparse_probe_budget_is_below_five_percent(self) -> None:
        config = DynamicConfig()
        self.assertEqual(config.sparse_probe_cost, 768)
        self.assertLessEqual(
            config.sparse_probe_cost / config.total_budget, 0.05
        )

    def test_moment_estimator_detects_independent_noise(self) -> None:
        config = DynamicConfig()
        rng = np.random.RandomState(123)
        snapshots = [rng.normal(size=8) for _ in range(2000)]
        estimate = estimate_dependence_components(
            snapshots, np.arange(8), config
        )
        self.assertGreater(estimate["rho_idiosyncratic"], 0.9)

    def test_numba_v2_matches_numpy_reference(self) -> None:
        config = DynamicConfig()
        max_delay = 4
        paths = generate_factor_paths(
            seed=20260729,
            maximum_delay=max_delay,
            config=config,
        )
        noise_tables = {
            scenario: build_noise_table(
                scenario=scenario,
                max_delay=max_delay,
                paths=paths,
                config=config,
            )
            for scenario in REGIME_SEQUENCE
        }
        for policy in POLICIES:
            reference = simulate_dynamic_policy(
                policy=policy,
                max_delay=max_delay,
                noise_tables=noise_tables,
                config=config,
                execution_engine="numpy_reference",
            )
            optimized = simulate_dynamic_policy(
                policy=policy,
                max_delay=max_delay,
                noise_tables=noise_tables,
                config=config,
                execution_engine="numba_block_v2",
            )
            np.testing.assert_allclose(
                optimized["checkpoint_errors"],
                reference["checkpoint_errors"],
                rtol=1e-12,
                atol=1e-14,
            )
            self.assertEqual(
                [
                    (row["selected_num_agents"], row["selected_eta"])
                    for row in optimized["actions"]
                ],
                [
                    (row["selected_num_agents"], row["selected_eta"])
                    for row in reference["actions"]
                ],
            )
            for key in (
                "charged_budget",
                "observed_messages",
                "total_probe_cost",
                "total_updates",
                "finite",
                "within_budget",
            ):
                self.assertEqual(optimized[key], reference[key])


if __name__ == "__main__":
    unittest.main()
