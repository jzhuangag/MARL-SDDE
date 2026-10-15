import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from relative_return_estimator import first_disagreement
import relative_return_estimator as estimator
from stable_overlap import stable_overlap


class StableIdentityTests(unittest.TestCase):
    def test_identity_normalization_roundoff(self):
        rng=np.random.default_rng(75109998)
        for n in (5,50):
            for _ in range(1000):
                p=rng.random(n)
                self.assertEqual(stable_overlap(p,p)[3],1.0)

    def test_do_not_clip_real_small_difference(self):
        p=np.array([.5,.5]); q=p+np.array([1e-10,-1e-10])
        self.assertLess(stable_overlap(p,q)[3],1.0)

    def test_identity_never_calls_branch(self):
        rng=np.random.default_rng(75109997)
        p=rng.random(50)
        def common(t):
            return [0],stable_overlap(p,p)[3]**3
        def branch(t,actions):
            raise AssertionError("phantom identity branch")
        self.assertEqual(first_disagreement(common,branch,25,rng)[0],0.0)


if __name__=="__main__":
    unittest.main()
