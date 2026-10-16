"""Outcome-free independent-validation/precision/accounting identities."""
import itertools
import unittest
import numpy as np


class DevelopmentProtocol(unittest.TestCase):
    def test_selected_training_score_biased_but_independent_validation_unbiased(self):
        mu=np.array([0.,.1,-.2]);noise=[np.array([0.,s*.5,t*.5]) for s,t in itertools.product((-1,1),repeat=2)]
        train=[];heldout=[];truth=[]
        for a,b in itertools.product(noise,repeat=2):
            k=int(np.argmax(mu+a));train.append((mu+a)[k]);heldout.append((mu+b)[k]);truth.append(mu[k])
        self.assertGreater(np.mean(train),np.mean(truth))
        self.assertAlmostEqual(np.mean(heldout),np.mean(truth))

    def test_predictable_diagnostic_cost_is_action_dependent_and_not_refunded(self):
        dimensions=[18,18,18];m=4;h=25;policy=[100,100,100];action=[.5,0,1.]
        active=[i for i,a in enumerate(action) if a]
        diagnostic=m*h*sum(dimensions[i]*4+16 for i in active)
        self.assertEqual(diagnostic,17600)
        whole_program=20*h*sum(d*4+20 for d in dimensions)
        self.assertEqual(whole_program+diagnostic,155600)
        self.assertEqual(sum(policy[i] for i in active),200)

    def test_training_and_validation_slices_are_disjoint_for_entire_probe_catalogue(self):
        ids=list(range(20))
        for m in (4,8,16):
            self.assertFalse(set(ids[:m])&set(ids[-4:]))


if __name__=="__main__":unittest.main()
