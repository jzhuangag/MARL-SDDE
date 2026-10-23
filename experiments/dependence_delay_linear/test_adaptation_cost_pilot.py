"""Symbolic, numerical, and behavioral tests for EXP-015A."""

import inspect
import math
import unittest

import numpy as np

from run_adaptation_cost_pilot import (
    Action,
    DELTA,
    ProbeDesign,
    ar1_correlation,
    ar1_mean_factor,
    available_updates,
    bhattacharyya_distance,
    candidate_actions,
    common_direction_eigenvalues,
    gaussian_covariance_kl,
    identification_threshold,
    kalman_log_likelihood,
    oracle_action,
    policy_outcome,
    select_probe_design,
    simulate_common_direction,
    theta_to_rho,
)


class AdaptationCostTests(unittest.TestCase):
    def test_rho_transform_matches_common_factor(self) -> None:
        self.assertAlmostEqual(theta_to_rho(9.0), 0.9)

    def test_ar1_correlation_is_positive_definite(self) -> None:
        matrix = ar1_correlation(20, 0.95)
        self.assertGreater(np.linalg.eigvalsh(matrix).min(), 0.0)

    def test_spectral_kl_matches_dense_formula(self) -> None:
        theta0, theta1, q, samples, coefficient = 0.05, 2.0, 8, 12, 0.8
        matrix = ar1_correlation(samples, coefficient)
        first = np.eye(samples) + q * theta0 * matrix
        second = np.eye(samples) + q * theta1 * matrix
        dense = 0.5 * (
            np.trace(np.linalg.solve(second, first))
            - samples
            + np.linalg.slogdet(second)[1]
            - np.linalg.slogdet(first)[1]
        )
        self.assertAlmostEqual(
            gaussian_covariance_kl(
                theta0, theta1, q, samples, coefficient
            ),
            dense,
            places=10,
        )

    def test_kl_zero_for_equal_instances(self) -> None:
        self.assertEqual(
            gaussian_covariance_kl(1.0, 1.0, 4, 16, 0.8),
            0.0,
        )

    def test_kl_accumulates_with_probe_count(self) -> None:
        small = gaussian_covariance_kl(0.05, 2.0, 4, 8, 0.8)
        large = gaussian_covariance_kl(0.05, 2.0, 4, 32, 0.8)
        self.assertGreater(large, small)

    def test_mixing_changes_identification_cost(self) -> None:
        fast, _ = identification_threshold(0.05, 0.5, 4, 1, 0.0)
        slow, _ = identification_threshold(0.05, 0.5, 4, 1, 0.95)
        self.assertGreater(slow, fast)

    def test_bhattacharyya_distance_is_symmetric(self) -> None:
        left = bhattacharyya_distance(0.05, 2.0, 8, 10, 0.7)
        right = bhattacharyya_distance(2.0, 0.05, 8, 10, 0.7)
        self.assertAlmostEqual(left, right)

    def test_q1_is_rejected_for_cross_agent_identification(self) -> None:
        with self.assertRaises(ValueError):
            identification_threshold(0.05, 2.0, 1, 1, 0.8)

    def test_dual_budget_horizon(self) -> None:
        action = Action(4, 2)
        updates = available_updates(action, 1000, 100, 16, 8)
        self.assertLessEqual((updates + 8) * (16 + 4), 1000)
        self.assertLessEqual((updates + 8) * 2, 100)

    def test_delay_reduces_completed_updates(self) -> None:
        action = Action(4, 1)
        self.assertGreater(
            available_updates(action, 1000, 100, 16, 0),
            available_updates(action, 1000, 100, 16, 8),
        )

    def test_markov_mean_factor_reduces_with_stride(self) -> None:
        self.assertGreater(
            ar1_mean_factor(100, 0.95),
            ar1_mean_factor(100, 0.95**8),
        )

    def test_closed_form_mean_factor_matches_direct_sum(self) -> None:
        samples = 37
        coefficient = 0.83
        direct = (
            samples
            + 2.0
            * sum(
                (samples - lag) * coefficient**lag
                for lag in range(1, samples)
            )
        ) / samples**2
        self.assertAlmostEqual(
            ar1_mean_factor(samples, coefficient), direct, places=12
        )

    def test_oracle_participation_changes_with_correlation(self) -> None:
        low, _ = oracle_action(0.0, 0.0, 100000, 100000, 4, 0, 32)
        high, _ = oracle_action(8.0, 0.0, 100000, 100000, 4, 0, 32)
        self.assertGreater(low.q, high.q)

    def test_horizon_aware_probe_can_fallback(self) -> None:
        design = ProbeDesign(4, 1, 20, 160, 20, 10)
        selected = select_probe_design(
            [design], 100, 10, 4, 8, 0, True, True
        )
        self.assertIsNone(selected)

    def test_decision_path_has_no_regime_argument(self) -> None:
        parameters = inspect.signature(select_probe_design).parameters
        self.assertNotIn("regime", parameters)
        self.assertNotIn("theta_true", parameters)

    def test_kalman_likelihood_prefers_generating_instance_on_average(self) -> None:
        wins = 0
        for seed in range(100):
            rng = np.random.RandomState(seed)
            observations = simulate_common_direction(
                rng, 2.0, 8, 30, 0.8
            )
            low = kalman_log_likelihood(observations, 0.05, 8, 0.8)
            high = kalman_log_likelihood(observations, 2.0, 8, 0.8)
            wins += int(high > low)
        self.assertGreaterEqual(wins / 100.0, 1.0 - DELTA)

    def test_same_seed_configuration_is_deterministic(self) -> None:
        scenario = {
            "scenario": "unit",
            "theta_low": 0.05,
            "theta_high": 2.0,
            "mixing": 0.8,
            "delay": 0,
            "overhead": 4,
            "maximum_agents": 8,
            "budget_name": "long",
            "budget_multiplier": 3.0,
            "message_budget": 5000,
            "environment_budget": 500,
            "reference_probe_q": 4,
            "reference_probe_b": 1,
            "reference_probe_samples": 20,
            "reference_lower_bound_samples": 10,
        }
        first = policy_outcome(20271101, scenario, "high", "paid_etc")
        second = policy_outcome(20271101, scenario, "high", "paid_etc")
        self.assertEqual(first, second)
        self.assertTrue(math.isfinite(first["squared_error"]))


if __name__ == "__main__":
    unittest.main()
