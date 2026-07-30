"""Deterministic tests for the EXP-005B online controller."""

import unittest

import numpy as np

from online_participation import (
    OnlineConfig,
    agent_metadata,
    generate_factor_paths,
    simulate_policy,
    true_aggregate_lrv,
)


class OnlineParticipationTest(unittest.TestCase):
    def test_clusters_are_interleaved(self) -> None:
        config = OnlineConfig()
        clusters = agent_metadata(config)["clusters"]
        np.testing.assert_array_equal(clusters[:8], [0, 1, 2, 3, 0, 1, 2, 3])

    def test_independent_lrv_has_linear_averaging(self) -> None:
        config = OnlineConfig()
        one = true_aggregate_lrv(np.arange(1), 0.0, 0.0, config)
        thirty_two = true_aggregate_lrv(
            np.arange(32), 0.0, 0.0, config
        )
        self.assertAlmostEqual(one / thirty_two, 32.0)

    def test_clustered_lrv_saturates(self) -> None:
        config = OnlineConfig()
        one = true_aggregate_lrv(np.arange(1), 0.0, 0.6, config)
        thirty_two = true_aggregate_lrv(
            np.arange(32), 0.0, 0.6, config
        )
        self.assertLess(one / thirty_two, 10.0)

    def test_fixed_q1_observation_accounting(self) -> None:
        config = OnlineConfig()
        paths = generate_factor_paths(123, 4, config)
        result = simulate_policy(
            policy="fixed_q1_adaptive_eta",
            scenario="independent",
            max_delay=4,
            paths=paths,
            config=config,
        )
        self.assertEqual(result["observed_messages"], result["total_updates"])
        self.assertLessEqual(result["budget_used"], config.total_budget)
        self.assertTrue(result["finite"])


if __name__ == "__main__":
    unittest.main()

