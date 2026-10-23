import unittest
import numpy as np
from finite_horizon_return_control import MonteCarloPotential
from shared_reference_return import contrasts,exact_identity_audit


class MonteCarloReturnTests(unittest.TestCase):
    def test_exact_expectation_with_wrong_potential(self):
        self.assertLess(exact_identity_audit()["maximum_absolute_error"],1e-12)

    def test_identity_zero_for_every_reported_estimator(self):
        rng=np.random.default_rng(78109997)
        x,_=contrasts(np.zeros((8,25)),rng.normal(size=(8,25)),rng.normal(size=(8,25)))
        for sample in x.values():np.testing.assert_array_equal(sample,np.zeros(8))

    def test_finite_time_dependent_training_only_fit(self):
        rng=np.random.default_rng(78109998)
        block={"states":rng.normal(size=(8*25,4)),"rewards":-np.ones((8,25))}
        fit=MonteCarloPotential(ridge=.001).fit([block])
        pred=fit.predict(block["states"])
        self.assertEqual(pred.shape,(8,25))
        self.assertTrue(np.isfinite(pred).all())
        self.assertGreater(float(pred[:,-1].mean()),float(pred[:,0].mean()))


if __name__=="__main__":unittest.main()
