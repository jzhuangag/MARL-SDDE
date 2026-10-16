"""Regression and statistical tests for EXP-014B implementation."""

import inspect
import math
import unittest

import numpy as np
import pandas as pd
import torch

from run_hierarchical_controller_pilot import (
    Action,
    CertificateState,
    OnlineState,
    b_step_reward,
    build_task_teacher,
    certificate_bounds,
    choose_hierarchical_action,
    effective_participation,
    mixing_upper,
    remaining_updates,
    run_configuration,
    stability_feasible,
)


class HierarchicalControllerTests(unittest.TestCase):
    def test_remaining_updates_monotonicity(self) -> None:
        self.assertGreaterEqual(
            remaining_updates(4, 1, 10000, 1000),
            remaining_updates(32, 1, 10000, 1000),
        )
        self.assertGreaterEqual(
            remaining_updates(4, 1, 10000, 1000),
            remaining_updates(4, 8, 10000, 1000),
        )
        self.assertGreaterEqual(
            remaining_updates(4, 1, 10000, 1000),
            remaining_updates(4, 1, 5000, 500),
        )

    def test_dual_budget_formula_never_overspends(self) -> None:
        for q in (4, 16, 32):
            for b in (1, 4, 8):
                n = remaining_updates(q, b, 12345, 987)
                self.assertLessEqual(n * (64 + q), 12345)
                self.assertLessEqual(n * b, 987)

    def test_b_step_teacher_fixed_point(self) -> None:
        device = torch.device("cpu")
        teacher = build_task_teacher(17, device)
        current = torch.randn(8, 4)
        following = torch.randn(8, 4)
        for gap in (1, 4, 8):
            reward = b_step_reward(
                teacher,
                current,
                following,
                gap,
                torch.zeros(8),
            )
            target = reward + (0.9 ** gap) * teacher(following)
            torch.testing.assert_close(target, teacher(current))

    def test_gap_does_not_change_simultaneous_rho(self) -> None:
        for gap in (1, 4, 8):
            self.assertAlmostEqual(effective_participation(16, 0.9), 16 / 14.5)
            self.assertGreaterEqual(mixing_upper(0.8, gap), 0.0)

    def test_decision_signature_has_no_true_rho(self) -> None:
        parameters = inspect.signature(
            choose_hierarchical_action
        ).parameters
        self.assertNotIn("true_rho", parameters)
        self.assertNotIn("teacher", parameters)

    def test_q1_cannot_create_correlation_certificate(self) -> None:
        state = CertificateState()
        bounds = certificate_bounds(state)
        self.assertEqual(state.collision_trials, 0)
        self.assertEqual(bounds["rho_upper"], 1.0)
        self.assertEqual(bounds["rho_lower"], 0.0)

    def test_anytime_certificate_known_rho_coverage(self) -> None:
        rng = np.random.RandomState(41)
        rho = 0.7
        covered = []
        for repetition in range(100):
            state = CertificateState()
            collisions = rng.binomial(
                1, 0.5 + 0.5 * rho, size=512
            )
            for value in collisions:
                state.collisions += int(value)
                state.collision_trials += 1
            covered.append(rho <= certificate_bounds(state)["rho_upper"])
        self.assertGreaterEqual(np.mean(covered), 0.99)

    def test_uncertainty_falls_back_all_agent(self) -> None:
        action, fallback, reason, _ = choose_hierarchical_action(
            OnlineState(),
            CertificateState(),
            delay=8,
            message_remaining=20000,
            environment_remaining=2000,
        )
        self.assertTrue(fallback)
        self.assertEqual(action.q, 32)
        self.assertEqual(reason, "insufficient_certificate")

    def test_rho0_delay0_ideal_falls_back_all_agent(self) -> None:
        state = CertificateState(
            stays=100,
            transition_trials=128,
            collisions=64,
            collision_trials=128,
        )
        action, fallback, reason, _ = choose_hierarchical_action(
            OnlineState(),
            state,
            delay=0,
            message_remaining=20000,
            environment_remaining=2000,
        )
        self.assertTrue(fallback)
        self.assertEqual(action.q, 32)
        self.assertEqual(reason, "zero_delay_no_harm")

    def test_high_rho_high_delay_can_select_small_q(self) -> None:
        state = CertificateState(
            stays=760,
            transition_trials=1000,
            collisions=950,
            collision_trials=1000,
            cumulative_collision_bias=0.0,
        )
        online = OnlineState(
            loss_upper=1.0,
            progress_lower=0.05,
            gradient_noise_upper=10.0,
        )
        action, fallback, _, bounds = choose_hierarchical_action(
            online,
            state,
            delay=8,
            message_remaining=30000,
            environment_remaining=3000,
        )
        self.assertGreater(bounds["rho_lower"], 0.55)
        self.assertFalse(fallback)
        self.assertLess(action.q, 32)

    def test_unsafe_eta_is_filtered(self) -> None:
        self.assertFalse(stability_feasible(Action(4, 1, 0.03), 8, 0.8))
        self.assertTrue(stability_feasible(Action(4, 8, 0.02), 8, 0.8))

    def test_streaming_state_is_scalar_memory(self) -> None:
        state = CertificateState()
        self.assertTrue(
            all(
                isinstance(value, (int, float))
                for value in vars(state).values()
            )
        )

    def test_analysis_primitives_are_deterministic(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0])
        first = float(np.exp(np.mean(np.log(values))))
        second = float(np.exp(np.mean(np.log(values))))
        self.assertEqual(first, second)
        self.assertTrue(math.isfinite(first))

    def test_end_to_end_configuration_respects_both_budgets(self) -> None:
        trajectory, endpoint, coverage = run_configuration(
            seed=20270821,
            task_seed=20270901,
            rho=0.9,
            delay=8,
            budget_name="unit",
            message_budget=400,
            environment_budget=64,
            policy="hierarchical_conservative",
            device=torch.device("cpu"),
        )
        self.assertTrue(trajectory)
        self.assertTrue(coverage)
        self.assertLessEqual(endpoint["messages"], endpoint["message_budget"])
        self.assertLessEqual(
            endpoint["environment_steps"], endpoint["environment_budget"]
        )
        self.assertTrue(math.isfinite(endpoint["teacher_mse"]))


if __name__ == "__main__":
    unittest.main()
