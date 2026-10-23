"""Algebra/software regression only: no environment, scientific seeds or returns."""
import itertools
import unittest
import numpy as np
import torch
from sealed_bridge_learning.control import AdaptiveRiskQueue, NativeBudget
from sealed_bridge_learning.prefix import prefix_objective, prefix_weights


class Interface(unittest.TestCase):
    def test_exact_prefix_mean_by_enumeration(self):
        theta = torch.tensor([.3, -.5], dtype=torch.float64, requires_grad=True)
        mu = torch.tensor([-.4, .6], dtype=torch.float64, requires_grad=True)
        expected = torch.zeros(2, dtype=torch.float64)
        control = torch.zeros(2, dtype=torch.float64)
        objective = torch.zeros((), dtype=torch.float64)
        source_objective = torch.zeros((), dtype=torch.float64)
        gamma = .91
        for flat in itertools.product((0., 1.), repeat=6):
            actions = torch.tensor(flat, dtype=torch.float64).reshape(3, 2)
            tl = torch.distributions.Bernoulli(logits=theta).log_prob(actions).sum(-1)
            sl = torch.distributions.Bernoulli(logits=mu).log_prob(actions).sum(-1)
            # A state baseline may depend on earlier actions, never on current action.
            baseline = torch.tensor([.2, flat[0]-.1, flat[2]+flat[1]], dtype=torch.float64)
            rewards = torch.tensor([flat[0]*flat[1], flat[2]-flat[3],
                                    flat[4]*flat[0]+flat[5]*flat[2]], dtype=torch.float64)
            wt, ws = prefix_weights(tl, sl)
            density = .5*(tl.detach().sum().exp()+sl.detach().sum().exp())
            expected += density*torch.autograd.grad(prefix_objective(tl, rewards, baseline, wt, gamma), theta, retain_graph=True)[0]
            control += density*torch.autograd.grad(prefix_objective(sl, rewards, baseline, ws, gamma), mu, retain_graph=True)[0]
            total = (rewards*gamma**torch.arange(3, dtype=torch.float64)).sum()
            objective = objective + tl.sum().exp()*total
            source_objective = source_objective + sl.sum().exp()*total
        np.testing.assert_allclose(expected, torch.autograd.grad(objective, theta)[0], atol=2e-15)
        np.testing.assert_allclose(control, torch.autograd.grad(source_objective, mu)[0], atol=2e-15)

    def test_late_drift_does_not_reweight_early_prefix(self):
        t = torch.tensor([-.5, -.6, -1.0], dtype=torch.float64)
        a, _ = prefix_weights(t, torch.tensor([-.5, -.6, -1.0], dtype=torch.float64))
        b, _ = prefix_weights(t, torch.tensor([-.5, -.6, -100.0], dtype=torch.float64))
        np.testing.assert_array_equal(a[:2], b[:2])
        self.assertGreater(float(b[-1]), float(a[-1]))

    def test_adaptive_regret_and_constant_invariance(self):
        controller = AdaptiveRiskQueue(range(3), [1.], 1.)
        shifted = AdaptiveRiskQueue(range(3), [1.], 1.)
        rng, other = np.random.default_rng(12), np.random.default_rng(12)
        losses = np.random.default_rng(19).uniform(0, 20, (80, 3))
        total, totals = 0., np.zeros(3)
        for k, loss in enumerate(losses):
            total += float(controller.probabilities @ loss)
            totals += loss
            controller.begin(np.zeros((3, 1)), rng, k, str(k))
            shifted.begin(np.zeros((3, 1)), other, k, str(k))
            controller.close(loss, [0.])
            shifted.close(loss+111., [0.])
            np.testing.assert_allclose(controller.probabilities, shifted.probabilities, atol=1e-14)
        bound = 1.5*np.sqrt(2)*np.sqrt(controller.epsilon+controller.loss_energy)
        self.assertLessEqual(total-totals.min(), bound+1e-12)

    def test_full_budget_inflight_drain_and_no_refund(self):
        ledger = NativeBudget(250, 25)
        ledger.launch('a', 2, 2)
        ledger.launch('b', 2, 2)
        self.assertEqual(ledger.free, 50)
        ledger.start_source('a'); ledger.finish_source('a')
        ledger.start_source('b'); ledger.finish_source('b')
        self.assertFalse(ledger.affordable_bridge('a', 5))
        with self.assertRaises(RuntimeError):
            ledger.terminal_fresh(1)
        ledger.start_receipt('a', 3)
        ledger.start_receipt('b', 2)
        ledger.terminal_fresh(1)
        ledger.charge_evaluation(100, 'benchmark')
        ledger.charge_evaluation(100, 'frozen_policy')
        ledger.assert_exhausted()
        self.assertEqual(ledger.evaluation_paid, 200)
        with self.assertRaises(RuntimeError):
            ledger.launch('a', 2, 2)


if __name__ == '__main__':
    unittest.main()
