"""Deterministic checks for the exact augmented-system implementation."""

import unittest

from linear_model import ModelConfig, exact_risk


class LinearModelTest(unittest.TestCase):
    def test_no_delay_white_noise_matches_closed_form(self) -> None:
        eta = 0.1
        curvature = 1.3
        rho = 0.4
        num_agents = 5
        config = ModelConfig(
            curvature=curvature,
            common_ar=0.0,
            idiosyncratic_ar=0.0,
            horizon=100,
        )
        result = exact_risk(
            eta=eta,
            rho=rho,
            num_agents=num_agents,
            delays=[0] * num_agents,
            config=config,
        )
        noise_variance = rho + (1.0 - rho) / num_agents
        expected = eta * noise_variance / (
            curvature * (2.0 - eta * curvature)
        )
        self.assertTrue(result["stable"])
        self.assertAlmostEqual(result["stationary_mse"], expected, places=11)

    def test_alignment_is_identical_without_delay(self) -> None:
        sample_time = exact_risk(
            eta=0.05,
            rho=0.8,
            num_agents=8,
            delays=[0] * 8,
            config=ModelConfig(common_noise_alignment="sample_time"),
        )
        server_time = exact_risk(
            eta=0.05,
            rho=0.8,
            num_agents=8,
            delays=[0] * 8,
            config=ModelConfig(common_noise_alignment="server_time"),
        )
        self.assertAlmostEqual(
            sample_time["finite_mse"], server_time["finite_mse"], places=12
        )
        self.assertAlmostEqual(
            sample_time["stationary_mse"],
            server_time["stationary_mse"],
            places=12,
        )

    def test_unstable_no_delay_step_is_detected(self) -> None:
        result = exact_risk(
            eta=2.1,
            rho=0.0,
            num_agents=1,
            delays=[0],
            config=ModelConfig(curvature=1.0),
        )
        self.assertFalse(result["stable"])

    def test_invalid_alignment_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            exact_risk(
                eta=0.1,
                rho=0.5,
                num_agents=1,
                delays=[0],
                config=ModelConfig(common_noise_alignment="invalid"),
            )


if __name__ == "__main__":
    unittest.main()
