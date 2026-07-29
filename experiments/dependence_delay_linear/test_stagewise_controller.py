"""Deterministic tests for the EXP-004 stagewise controller."""

import unittest

import numpy as np

from stagewise_controller import (
    DelayProxyCache,
    StagewiseConfig,
    batch_means_lrv,
    build_delay_transition,
    choose_action,
    true_long_run_variance,
)


class StagewiseControllerTest(unittest.TestCase):
    def test_no_delay_transition(self) -> None:
        transition = build_delay_transition(
            eta=0.1, curvature=1.5, delays=[0, 0, 0]
        )
        self.assertEqual(transition.shape, (1, 1))
        self.assertAlmostEqual(transition[0, 0], 0.85)

    def test_batch_means_white_noise_scale(self) -> None:
        rng = np.random.RandomState(123)
        values = rng.normal(size=20000)
        estimate = batch_means_lrv(values, batch_size=20)
        self.assertLess(abs(estimate - 1.0), 0.15)

    def test_true_lrv_saturates_with_common_noise(self) -> None:
        config = StagewiseConfig()
        one = true_long_run_variance(0.9, 1, config)
        thirty_two = true_long_run_variance(0.9, 32, config)
        independent_one = true_long_run_variance(0.0, 1, config)
        independent_thirty_two = true_long_run_variance(0.0, 32, config)
        self.assertLess(one / thirty_two, 1.02)
        self.assertGreater(independent_one / independent_thirty_two, 30.0)

    def test_first_stage_action_is_fixed_and_predictable(self) -> None:
        config = StagewiseConfig()
        maximum_delay = max(config.max_delay_schedule)
        x_buffer = np.ones(config.total_steps + maximum_delay + 1)
        action = choose_action(
            policy="adaptive_joint",
            stage=0,
            x_buffer=x_buffer,
            current_index=maximum_delay,
            previous_statistics=None,
            previous_delays=None,
            current_delays=config.delay_profiles()[0],
            current_rho=0.0,
            cache=DelayProxyCache(config),
            config=config,
        )
        self.assertEqual(int(action["num_agents"]), 32)
        self.assertAlmostEqual(action["eta"], config.default_eta)


if __name__ == "__main__":
    unittest.main()
