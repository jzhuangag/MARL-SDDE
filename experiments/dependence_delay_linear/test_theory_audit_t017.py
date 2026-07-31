import math
import unittest

import numpy as np

from adaptive_change_of_measure import AdaptiveAction
from theory_audit_t017 import (
    amortization_scale_lower,
    asymptotic_risk_coefficient,
    centered_normal_scale_tv,
    le_cam_error_floor,
    mixing_boundary_kl,
    terminal_mean_risk,
    zero_mean_gaussian_kl,
)


class T017TheoryAuditTests(unittest.TestCase):
    def test_exact_scale_tv_is_symmetric_and_strictly_below_one(self):
        forward = centered_normal_scale_tv(0.25, 2.0)
        reverse = centered_normal_scale_tv(2.0, 0.25)
        self.assertAlmostEqual(forward, reverse, places=15)
        self.assertGreater(forward, 0.0)
        self.assertLess(forward, 1.0)
        self.assertGreater(le_cam_error_floor(0.25, 2.0), 0.0)

    def test_covariance_kl_identity(self):
        source = np.asarray([[2.0, 0.3], [0.3, 1.0]])
        self.assertAlmostEqual(zero_mean_gaussian_kl(source, source), 0.0, places=14)
        target = np.eye(2)
        expected = 0.5 * (np.trace(source) - 2.0 - np.linalg.slogdet(source)[1])
        self.assertAlmostEqual(zero_mean_gaussian_kl(source, target), expected)

    def test_every_finite_irregular_design_converges_to_lambda_one(self):
        actions = [
            AdaptiveAction(q=1, b=1),
            AdaptiveAction(q=4, b=3),
            AdaptiveAction(q=2, b=7),
            AdaptiveAction(q=5, b=2),
        ]
        values = [mixing_boundary_kl(actions, 0.7, 1.0 - 10.0 ** (-k)) for k in range(2, 7)]
        self.assertTrue(all(left > right for left, right in zip(values, values[1:])))
        self.assertLess(values[-1], 1.0e-8)

    def test_lambda_one_information_is_bounded_by_one_latent_draw(self):
        theta0, theta1 = 0.4, 1.3
        actions = [AdaptiveAction(q=8, b=1) for _ in range(250)]
        from adaptive_change_of_measure import dense_common_covariance

        data_kl = zero_mean_gaussian_kl(
            dense_common_covariance(actions, theta0, 1.0),
            dense_common_covariance(actions, theta1, 1.0),
        )
        latent_kl = 0.5 * (theta0 / theta1 - 1.0 - math.log(theta0 / theta1))
        self.assertLessEqual(data_kl, latent_kl + 1.0e-12)
        self.assertAlmostEqual(data_kl, latent_kl, delta=2.0e-3)

    def test_exact_risk_has_claimed_asymptotic_coefficient(self):
        theta, q, coefficient = 0.6, 5, 0.8
        target = asymptotic_risk_coefficient(theta, q, coefficient)
        self.assertAlmostEqual(
            200000 * terminal_mean_risk(theta, q, coefficient, 200000),
            target,
            delta=2.0e-4,
        )

    def test_probe_amortization_threshold_diverges_as_oracle_gap_closes(self):
        values = [amortization_scale_lower(20, 3.0, gap) for gap in (0.2, 0.02, 0.002)]
        self.assertEqual(values, [300.0, 3000.0, 30000.0])


if __name__ == "__main__":
    unittest.main()
