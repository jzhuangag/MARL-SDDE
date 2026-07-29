"""Deterministic tests for EXP-005C sparse dynamic control."""

import unittest

import numpy as np

from sparse_dynamic import (
    DynamicConfig,
    estimate_dependence_components,
    regime_for_block,
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


if __name__ == "__main__":
    unittest.main()
