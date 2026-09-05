"""Deterministic tests for EXP-013B analysis."""

import unittest

import pandas as pd

from run_realizable_td_confirmation import (
    delay_ratios,
    oracle_choices,
    paired_cluster_ratio,
)


class RealizableConfirmationTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for seed in (1, 2, 3):
            for rho in (0.0, 0.25, 0.5, 0.9):
                for delay in (0, 8):
                    for q in (1, 4, 16, 32):
                        if rho == 0.0:
                            mse = 1.0 / q
                        elif rho == 0.9:
                            mse = 1.0 + abs(q - 4) / 32.0
                        else:
                            mse = 1.0
                        rows.append(
                            {
                                "seed": seed,
                                "rho": rho,
                                "delay": delay,
                                "num_agents": q,
                                "teacher_mse": mse,
                                "finite": True,
                            }
                        )
        self.metrics = pd.DataFrame(rows)

    def test_cluster_ratio_is_deterministic(self) -> None:
        first = paired_cluster_ratio(
            self.metrics, 0.0, 32, 1, 100, 7
        )
        second = paired_cluster_ratio(
            self.metrics, 0.0, 32, 1, 100, 7
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["ratio"], 1.0 / 32.0)

    def test_delay_ratios_preserve_pairing(self) -> None:
        ratios = delay_ratios(self.metrics, 0.0, 32, 1)
        self.assertAlmostEqual(ratios["0"], 1.0 / 32.0)
        self.assertAlmostEqual(ratios["8"], 1.0 / 32.0)

    def test_oracle_choices_follow_constructed_optima(self) -> None:
        _, medians = oracle_choices(self.metrics)
        self.assertEqual(medians["0.0"], 32.0)
        self.assertEqual(medians["0.9"], 4.0)


if __name__ == "__main__":
    unittest.main()
