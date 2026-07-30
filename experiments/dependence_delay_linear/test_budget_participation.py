"""Deterministic tests for EXP-005A resource accounting."""

import unittest

import numpy as np

from budget_participation import (
    BudgetConfig,
    budget_horizon,
    per_update_cost,
    selected_delays,
    selected_indices,
)


class BudgetParticipationTest(unittest.TestCase):
    def test_fastest_rule_is_prefix(self) -> None:
        indices = selected_indices(32, 8, "fastest")
        np.testing.assert_array_equal(indices, np.arange(8))

    def test_uniform_rank_reaches_slowest_agent(self) -> None:
        indices = selected_indices(32, 8, "uniform_rank")
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 31)
        self.assertEqual(len(np.unique(indices)), 8)

    def test_message_budget_charges_only_selected_agents(self) -> None:
        spec = {"kind": "message", "budget": 6400.0, "overhead": 4.0}
        delays = np.asarray([0, 1, 2, 3])
        self.assertAlmostEqual(per_update_cost(4, delays, spec), 8.0)
        self.assertEqual(budget_horizon(4, delays, spec), 800)

    def test_smaller_fastest_subset_has_no_larger_delay(self) -> None:
        config = BudgetConfig()
        delay_4 = selected_delays(16, 4, "fastest", config)
        delay_32 = selected_delays(16, 32, "fastest", config)
        self.assertLessEqual(np.max(delay_4), np.max(delay_32))


if __name__ == "__main__":
    unittest.main()

