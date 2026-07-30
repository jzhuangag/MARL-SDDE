"""Deterministic tests for EXP-006A oracle participation phase diagram."""

import unittest

import numpy as np

from online_participation import FiniteBudgetProxyCache, OnlineConfig
from oracle_phase import (
    build_surface,
    oracle_action,
    summarize_delay,
    summarize_tracks,
)


class OraclePhaseTest(unittest.TestCase):
    def test_oracle_action_is_finite_and_feasible(self) -> None:
        config = OnlineConfig()
        action = oracle_action(
            rho_global=0.6,
            rho_cluster=0.0,
            budget=2000,
            error_amplitude=0.1,
            max_delay=16,
            config=config,
            cache=FiniteBudgetProxyCache(config),
        )
        self.assertIn(int(action["selected_q"]), (1, 2, 4, 8, 16, 32))
        self.assertTrue(np.isfinite(action["best_risk"]))
        self.assertGreaterEqual(action["relative_margin"], 0.0)
        self.assertNotEqual(action["selected_q"], action["runner_up_q"])

    def test_small_surface_has_exact_shape(self) -> None:
        surface = build_surface(
            paths=("global",),
            strengths=(0.0, 0.8),
            budgets=(500,),
            errors=(0.1,),
            max_delays=(4, 16),
            overheads=(4,),
        )
        self.assertEqual(len(surface), 4)
        tracks = summarize_tracks(surface)
        delays = summarize_delay(surface)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(delays), 2)
        self.assertTrue(np.isfinite(surface["best_risk"]).all())


if __name__ == "__main__":
    unittest.main()
