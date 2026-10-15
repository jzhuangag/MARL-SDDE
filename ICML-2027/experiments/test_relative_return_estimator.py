import unittest
import numpy as np
from relative_return_estimator import finite_game_audit, maximal_pair, disagreeing_joint, first_disagreement, empirical_bernstein


class RelativeReturnTests(unittest.TestCase):
    def test_exact_markov_identity(self):
        self.assertLess(finite_game_audit()["absolute_error"],1e-12)

    def test_maximal_marginals_and_disagreement(self):
        rng=np.random.default_rng(88201)
        p=np.array([.1,.3,.6]); q=np.array([.5,.2,.3])
        counts=np.zeros((3,3))
        for _ in range(20000):
            a,b=maximal_pair(p,q,rng); counts[a,b]+=1
        counts/=counts.sum()
        np.testing.assert_allclose(counts.sum(1),p,atol=.015)
        np.testing.assert_allclose(counts.sum(0),q,atol=.015)
        self.assertAlmostEqual(float(np.trace(counts)),.6,delta=.015)

    def test_conditioned_joint_not_independent_residuals(self):
        rng=np.random.default_rng(88202)
        p=[np.array([.3,.7]),np.array([.6,.4])]
        q=[np.array([.5,.5]),np.array([.2,.8])]
        first=0
        for _ in range(10000):
            a,b=disagreeing_joint(p,q,rng)
            self.assertNotEqual(a,b)
            first+=int(a[0]!=b[0])
        self.assertAlmostEqual(first/10000,.2/(1-.8*.6),delta=.02)

    def test_zero_identity_and_known_weight(self):
        rng=np.random.default_rng(88203)
        x,aux=first_disagreement(lambda t:([0],1),lambda t,p:999,4,rng)
        self.assertEqual(x,0)
        x,aux=first_disagreement(lambda t:([0],.9),lambda t,p:2,4,rng)
        self.assertAlmostEqual(x,2*(1-.9**4))

    def test_no_sample_range_confidence(self):
        low,rad=empirical_bernstein([0]*16,20,.05)
        self.assertGreater(rad,0)
        self.assertLess(low,0)


if __name__=="__main__":
    unittest.main()
