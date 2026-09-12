"""Focused tests for the baseline-as-sensing anytime switch prototype."""

import math
import unittest

from adaptive_change_of_measure import AdaptiveAction
from baseline_as_sensing_switch import (
    BaselineAsSensingSwitch,
    anytime_threshold,
    bretagnolle_huber_safety_slack_lower_bound,
    exact_cumulative_log_likelihood_ratio,
    high_regime_chernoff_risk_bound,
)


class BaselineAsSensingSwitchTests(unittest.TestCase):
    def test_threshold_and_zero_safety_boundary(self):
        self.assertAlmostEqual(
            anytime_threshold(switch_loss_upper=4.0, safety_budget=1.0), math.log(4.0)
        )
        self.assertTrue(math.isinf(anytime_threshold(switch_loss_upper=4.0, safety_budget=0.0)))
        for loss, epsilon in ((0.0, 0.0), (1.0, -0.1), (1.0, 1.0), (1.0, 2.0)):
            with self.assertRaises(ValueError):
                anytime_threshold(switch_loss_upper=loss, safety_budget=epsilon)

    def test_packetwise_log_lr_matches_exact_batch_identity(self):
        actions = [AdaptiveAction(2, 1), AdaptiveAction(5, 3), AdaptiveAction(1, 2)]
        observations = [0.35, -0.7, 1.1]
        controller = BaselineAsSensingSwitch(
            theta_low=0.1, theta_high=1.3, mixing=0.8,
            threshold=math.inf, cutoff_packets=len(actions),
        )
        records = [
            controller.observe_baseline_packet(observation=y, action=a)
            for y, a in zip(observations, actions)
        ]
        self.assertFalse(controller.switched)
        self.assertAlmostEqual(
            controller.cumulative_log_likelihood_ratio,
            exact_cumulative_log_likelihood_ratio(
                observations=observations, actions=actions, theta_low=0.1,
                theta_high=1.3, mixing=0.8,
            ), places=12,
        )
        self.assertAlmostEqual(sum(record.increment for record in records), controller.cumulative_log_likelihood_ratio, places=14)

    def test_first_crossing_switches_once_and_cutoff_is_strict(self):
        action = AdaptiveAction(4, 1)
        controller = BaselineAsSensingSwitch(
            theta_low=0.05, theta_high=2.0, mixing=0.6,
            threshold=0.0, cutoff_packets=2,
        )
        first = controller.observe_baseline_packet(observation=0.0, action=action)
        second = controller.observe_baseline_packet(observation=9.0, action=action)
        self.assertFalse(first.crossed)
        self.assertFalse(first.switched)
        self.assertTrue(second.switched)
        self.assertEqual(controller.switched_at, 2)
        with self.assertRaises(RuntimeError):
            controller.observe_baseline_packet(observation=0.2, action=action)

    def test_high_regime_chernoff_tail_sum_bound(self):
        bound = high_regime_chernoff_risk_bound(
            threshold=math.log(10.0), chernoff_s=0.5, information_rate=0.4,
            initialization_constant=0.2, cutoff_packets=20,
            delay_and_inflight_loss=0.1, per_packet_opportunity_loss=0.03,
            no_switch_loss=2.0,
        )
        expected_n = math.ceil((0.5 * math.log(10.0) + 0.2) / 0.4)
        self.assertEqual(bound.detection_scale, expected_n)
        self.assertLessEqual(bound.cutoff_miss_probability_upper, 1.0)
        self.assertGreaterEqual(bound.regime_one_regret_upper, 0.1)
        infinite = high_regime_chernoff_risk_bound(
            threshold=math.inf, chernoff_s=0.5, information_rate=0.4,
            initialization_constant=0.0, cutoff_packets=20,
            delay_and_inflight_loss=0.1, per_packet_opportunity_loss=0.03,
            no_switch_loss=2.0,
        )
        self.assertTrue(math.isinf(infinite.regime_one_regret_upper))

    def test_bretagnolle_huber_safety_slack_lower_bound(self):
        expected = 3.0 * max(0.0, 0.5 * math.exp(-math.log(2.0)) - 0.5 / 4.0)
        self.assertAlmostEqual(
            bretagnolle_huber_safety_slack_lower_bound(
                low_regime_wrong_deployment_gap=3.0,
                high_regime_wrong_deployment_gap=4.0,
                maximum_kl=math.log(2.0), high_regime_regret=0.5,
            ), expected,
        )
        self.assertEqual(
            bretagnolle_huber_safety_slack_lower_bound(
                low_regime_wrong_deployment_gap=3.0,
                high_regime_wrong_deployment_gap=4.0,
                maximum_kl=0.0, high_regime_regret=4.0,
            ), 0.0,
        )
        with self.assertRaises(ValueError):
            bretagnolle_huber_safety_slack_lower_bound(
                low_regime_wrong_deployment_gap=0.0,
                high_regime_wrong_deployment_gap=1.0,
                maximum_kl=0.0, high_regime_regret=0.0,
            )


if __name__ == "__main__":
    unittest.main()
