"""Algebra/protocol regressions; these are not learning-performance evidence."""
import unittest
import numpy as np
from sealed_bridge.core import (bridge_weights, transported_direction, risk_feedback,
                               Option, RiskQueueController, SealedAnchor)


class BridgeTests(unittest.TestCase):
    def test_joint_trajectory_mixture_and_gradient_identity(self):
        # Exhaustive two-agent, two-step action words with reward interaction.
        words = np.asarray([[int(k >> j & 1) for j in range(4)] for k in range(16)])
        source = np.asarray([.25,.7,.25,.7]); target = np.asarray([.6,.4,.6,.4])
        p = np.prod(np.where(words, target, 1-target), axis=1)
        mu = np.prod(np.where(words, source, 1-source), axis=1)
        q = (p+mu)/2
        reward = (words[:,0]==words[:,1]).astype(float) + 2*(words[:,2]!=words[:,3])
        ht = reward[:,None]*np.stack([words[:,[0,2]].sum(1)-2*target[0],
                                    words[:,[1,3]].sum(1)-2*target[1]],axis=1)
        hm = reward[:,None]*np.stack([words[:,[0,2]].sum(1)-2*source[0],
                                    words[:,[1,3]].sum(1)-2*source[1]],axis=1)
        wt, wm = bridge_weights(np.log(p),np.log(mu))
        g = (p[:,None]*ht).sum(0); anchor = (mu[:,None]*hm).sum(0)
        for beta in [0,.5,1]:
            expectation = (q[:,None]*(wt[:,None]*ht-beta*wm[:,None]*hm)).sum(0)+beta*anchor
            np.testing.assert_allclose(expectation,g,rtol=0,atol=1e-14)
        # Per-agent mixing is a different experiment and MUST NOT be substituted.
        wrong = np.prod(np.where(words, (source+target)/2,1-(source+target)/2),axis=1)
        self.assertGreater(float(np.max(abs(q-wrong))), .001)

    def test_weight_bounds_do_not_depend_on_log_gap(self):
        wt, wm = bridge_weights(np.array([-1.,-10000.]),np.array([-10000.,-1.]))
        np.testing.assert_allclose(wt,[2.,0.]);np.testing.assert_allclose(wm,[0.,2.])

    def test_equal_policy_pooling_limit(self):
        rng=np.random.default_rng(4501)
        bridge=rng.normal(size=(8,5));anchor=rng.normal(size=(4,5))
        d=transported_direction(bridge,bridge,anchor,4/12)
        np.testing.assert_allclose(d,np.concatenate([bridge,anchor]).mean(0),rtol=0,atol=2e-16)

    def test_controller_is_predata_and_cost_exact(self):
        options=[Option(4,0),Option(8,1)]
        c=RiskQueueController(options,step=.1,variance_weight=.2,targets=[1.,1.])
        costs=np.array([[.5,.7],[1.5,1.4]])
        decision=c.begin(costs,np.random.default_rng(1),source_id='s',current_hash='theta')
        a=SealedAnchor('s',np.arange(6.).reshape(3,2))
        x=a.open_once(decision)
        with self.assertRaises(RuntimeError):a.open_once(decision)
        risks=risk_feedback(x,x,x,options)
        report=c.close(risks,costs[decision['index']])
        self.assertFalse(report['variance_upper_certificate'])
        self.assertFalse(report['current_return_used'])
        self.assertAlmostEqual(sum(report['probabilities_after']),1.)

    def test_bad_roles_or_shapes_fail_closed(self):
        a=SealedAnchor('s',np.ones((2,2)))
        with self.assertRaises(RuntimeError):a.open_once({'source_id':'s'})
        with self.assertRaises(ValueError):risk_feedback(np.ones((1,2)),np.ones((1,2)),np.ones((2,2)),[Option(2,1)])


if __name__ == '__main__':unittest.main()
