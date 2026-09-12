"""Deterministic tests for nonlinear gradient-variance smoke helpers."""

import unittest

import numpy as np
import pandas as pd

from run_gradient_variance_smoke import add_theory_columns


class GradientVarianceSmokeTests(unittest.TestCase):
    def test_theory_ratio_is_attached_per_correlation(self) -> None:
        metrics = pd.DataFrame(
            [
                {"rho": 0.0, "num_agents": 1, "trace_variance": 4.0},
                {"rho": 0.0, "num_agents": 4, "trace_variance": 1.0},
                {"rho": 0.5, "num_agents": 1, "trace_variance": 4.0},
                {"rho": 0.5, "num_agents": 4, "trace_variance": 2.5},
            ]
        )
        augmented = add_theory_columns(metrics)
        np.testing.assert_allclose(
            augmented["normalized_trace_variance"],
            augmented["theory_ratio"],
        )
        np.testing.assert_allclose(
            augmented["absolute_ratio_error"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
