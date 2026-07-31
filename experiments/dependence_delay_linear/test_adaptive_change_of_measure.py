"""Numerical and boundary audits for the T-016 likelihood theorems."""

import math
import unittest

import mpmath
import numpy as np

from adaptive_change_of_measure import (
    AdaptiveAction,
    adaptive_log_likelihood,
    adaptive_log_likelihood_ratio,
    conditional_gaussian_kl,
    dense_common_covariance,
    dense_log_likelihood,
    dual_budget_feasible,
    innovation_information_trace,
    simulate_adaptive_path,
    theorem_derived_fallback,
    usable_commit_updates,
)


def threshold_rule(history, _actions):
    if not history:
        return AdaptiveAction(2, 1)
    if history[-1] >= 0.0:
        return AdaptiveAction(5, 3)
    return AdaptiveAction(3, 1)


class AdaptiveChangeOfMeasureTests(unittest.TestCase):
    def test_conditional_kl_matches_high_precision_scalar_identity(self):
        got = conditional_gaussian_kl(0.3, 0.7, -0.2, 1.4, 5)
        mpmath.mp.dps = 70
        s0 = 1 + mpmath.mpf(5) * mpmath.mpf("0.7")
        s1 = 1 + mpmath.mpf(5) * mpmath.mpf("1.4")
        dm = mpmath.sqrt(5) * mpmath.mpf("0.5")
        expected = (mpmath.log(s1 / s0) + (s0 + dm**2) / s1 - 1) / 2
        self.assertAlmostEqual(got, float(expected), places=14)

    def test_irregular_dimension_changing_innovations_match_dense(self):
        actions = [
            AdaptiveAction(2, 1),
            AdaptiveAction(5, 3),
            AdaptiveAction(1, 2),
            AdaptiveAction(8, 4),
        ]
        observations = np.asarray([0.4, -1.2, 0.7, 2.1])
        for theta in (0.05, 2.0):
            self.assertAlmostEqual(
                adaptive_log_likelihood(observations, actions, theta, 0.83),
                dense_log_likelihood(observations, actions, theta, 0.83),
                places=12,
            )

    def test_adaptive_selection_pathwise_lr_matches_brute_dense(self):
        rng = np.random.RandomState(17)
        observations, actions = simulate_adaptive_path(
            rng, 0.5, 0.8, threshold_rule, 5
        )
        innovation = adaptive_log_likelihood_ratio(
            observations, actions, 2.0, 0.05, 0.8
        )
        brute = dense_log_likelihood(
            observations, actions, 2.0, 0.8
        ) - dense_log_likelihood(observations, actions, 0.05, 0.8)
        self.assertAlmostEqual(innovation, brute, places=11)

    def test_brute_force_enumerates_every_small_threshold_branch(self):
        for first in (-0.4, 0.4):
            for second in (-0.7, 0.7):
                observations = np.asarray([first, second, 0.2])
                actions = []
                history = []
                for observation in observations:
                    actions.append(threshold_rule(tuple(history), tuple(actions)))
                    history.append(float(observation))
                innovation = adaptive_log_likelihood_ratio(
                    observations, actions, 0.5, 0.05, 0.9
                )
                brute = dense_log_likelihood(
                    observations, actions, 0.5, 0.9
                ) - dense_log_likelihood(
                    observations, actions, 0.05, 0.9
                )
                self.assertAlmostEqual(innovation, brute, places=11)

    def test_full_dimension_lr_reduces_to_common_directions(self):
        actions = [AdaptiveAction(2, 1), AdaptiveAction(3, 2)]
        blocks = [np.asarray([0.2, -0.6]), np.asarray([1.0, 0.4, -0.1])]
        common = np.asarray(
            [block.sum() / math.sqrt(len(block)) for block in blocks]
        )
        times = (0, 2)
        full = np.concatenate(blocks)

        def full_log_density(theta):
            covariance = np.eye(5)
            offsets = (0, 2)
            for t, action_t in enumerate(actions):
                for s, action_s in enumerate(actions):
                    covariance[
                        offsets[t] : offsets[t] + action_t.q,
                        offsets[s] : offsets[s] + action_s.q,
                    ] += theta * 0.8 ** abs(times[t] - times[s])
            return -0.5 * (
                5 * math.log(2 * math.pi)
                + np.linalg.slogdet(covariance)[1]
                + full @ np.linalg.solve(covariance, full)
            )

        full_lr = full_log_density(0.5) - full_log_density(0.05)
        reduced_lr = adaptive_log_likelihood_ratio(
            common, actions, 0.5, 0.05, 0.8
        )
        self.assertAlmostEqual(full_lr, reduced_lr, places=11)

    def test_both_directional_kl_equal_expected_log_lr(self):
        for source, target in ((0.05, 0.5), (0.5, 0.05)):
            rng = np.random.RandomState(902)
            log_ratios = []
            informations = []
            for _ in range(6000):
                observations, actions = simulate_adaptive_path(
                    rng, source, 0.7, threshold_rule, 3
                )
                log_ratios.append(
                    adaptive_log_likelihood_ratio(
                        observations, actions, source, target, 0.7
                    )
                )
                informations.append(
                    sum(
                        innovation_information_trace(
                            observations,
                            actions,
                            source,
                            target,
                            0.7,
                        )
                    )
                )
            self.assertLess(
                abs(np.mean(log_ratios) - np.mean(informations)), 0.035
            )
            self.assertGreater(np.mean(informations), 0.0)

    def test_likelihood_ratio_martingale_mean(self):
        rng = np.random.RandomState(44)
        ratios = []
        for _ in range(15000):
            observations, actions = simulate_adaptive_path(
                rng, 0.2, 0.6, threshold_rule, 3
            )
            ratios.append(
                math.exp(
                    adaptive_log_likelihood_ratio(
                        observations, actions, 0.3, 0.2, 0.6
                    )
                )
            )
        self.assertLess(abs(float(np.mean(ratios)) - 1.0), 0.025)

    def test_ville_optional_stopping_error_bound(self):
        rng = np.random.RandomState(190)
        crossings = 0
        delta = 0.1
        for _ in range(12000):
            observations, actions = simulate_adaptive_path(
                rng, 0.2, 0.6, threshold_rule, 6
            )
            crossed = False
            for stop in range(1, 7):
                log_ratio = adaptive_log_likelihood_ratio(
                    observations[:stop],
                    actions[:stop],
                    0.35,
                    0.2,
                    0.6,
                )
                if log_ratio >= math.log(1.0 / delta):
                    crossed = True
                    break
            crossings += int(crossed)
        self.assertLessEqual(crossings / 12000.0, delta + 0.012)

    def test_lambda_zero_covariance_is_diagonal(self):
        actions = [AdaptiveAction(2, 1), AdaptiveAction(7, 4)]
        covariance = dense_common_covariance(actions, 0.8, 0.0)
        self.assertEqual(covariance[0, 1], 0.0)

    def test_lambda_one_information_saturates(self):
        short = dense_common_covariance(
            [AdaptiveAction(4, 1)] * 20, 1.0, 1.0
        )
        long = dense_common_covariance(
            [AdaptiveAction(4, 1)] * 200, 1.0, 1.0
        )
        info_short = 0.5 * (
            np.trace(np.linalg.solve(short, 4.0 * np.ones_like(short))) ** 2
        )
        info_long = 0.5 * (
            np.trace(np.linalg.solve(long, 4.0 * np.ones_like(long))) ** 2
        )
        self.assertLess(info_long / info_short, 1.03)

    def test_q_one_is_valid_variance_data_but_not_spatial_certificate(self):
        value = conditional_gaussian_kl(0.0, 0.05, 0.0, 0.5, 1)
        self.assertGreater(value, 0.0)

    def test_exact_dual_budget_exhaustion(self):
        actions = [AdaptiveAction(2, 2), AdaptiveAction(2, 2)]
        self.assertTrue(dual_budget_feasible(actions, 3, 10, 5, 1))
        self.assertFalse(dual_budget_feasible(actions, 3, 9, 5, 1))
        self.assertFalse(dual_budget_feasible(actions, 3, 10, 4, 1))

    def test_delay_exceeding_horizon_leaves_no_commit(self):
        self.assertEqual(usable_commit_updates(4, 8), 0)

    def test_fallback_is_strict_theorem_inequality(self):
        explore, epsilon = theorem_derived_fallback(1.0, 0.2, 0.3, 1.6)
        self.assertTrue(explore)
        self.assertAlmostEqual(epsilon, 1.5)
        tied, _ = theorem_derived_fallback(1.0, 0.2, 0.3, 1.5)
        self.assertFalse(tied)


if __name__ == "__main__":
    unittest.main()
