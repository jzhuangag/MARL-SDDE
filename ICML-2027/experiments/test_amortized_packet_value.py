"""Finite-distribution/protocol identities only; no native MARL evaluation."""
import copy
import itertools
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from amortized_packet_value import value_geometry, expected_risk_feedback, interpolate_server_proposal


class GeometryTests(unittest.TestCase):
    def test_additive_basis_and_explicit_remainder(self):
        actions = np.array(list(itertools.product((0., .5, 1., 2.), repeat=2)))
        basis = [np.flatnonzero(np.all(actions == a, axis=1))[0] for a in ((1.,0.),(0.,1.))]
        z = np.array([[1.,-2.],[-1.,2.],[2.,1.],[-2.,-1.]])
        x = z @ actions.T
        linear = value_geometry(x,actions,basis)
        self.assertEqual(linear["response_basis"]["remainder_statistic"],0.)
        remainder = .1*actions[:,0]*actions[:,1]
        nonlinear = value_geometry(x+remainder,actions,basis)
        self.assertAlmostEqual(nonlinear["response_basis"]["remainder_statistic"],1.6)
        centered = value_geometry(x+remainder,actions,basis,predictable_offset=remainder)
        self.assertLess(centered["response_basis"]["remainder_statistic"],1e-14)
        idle = value_geometry(np.zeros((4,1)),np.empty((1,0)),())
        self.assertEqual(idle["response_basis"]["variance_statistic"],0.)
        self.assertEqual(idle["response_basis"]["remainder_statistic"],0.)

    def test_expected_selection_gap_not_pointwise_certificate(self):
        actions = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        mean = np.array([.1,-.2]); nonlinear = np.array([0.,0.,0.,-.3])
        outcomes = [np.array([a*.4,b*.3])+mean for a,b in itertools.product((-1.,1.),repeat=2)]
        truth = actions @ mean + nonlinear
        gaps, penalties = [], []
        for u,v in itertools.product(outcomes,repeat=2):
            x = np.stack((u,v)) @ actions.T + nonlinear
            g = value_geometry(x,actions,(1,2))
            chosen = int(np.argmax(x.mean(axis=0)))
            gaps.append(float(truth.max()-truth[chosen]))
            penalties.append(expected_risk_feedback(g,"response_basis",2,.2))
        self.assertLessEqual(np.mean(gaps),np.mean(penalties))

    def test_line_search_owns_server_moments_and_idle_preserves_them(self):
        torch.manual_seed(7)
        module = torch.nn.Linear(2,1)
        server = SimpleNamespace(actor=module,actor_optimizer=torch.optim.Adam(module.parameters(),lr=.01))
        unit = copy.deepcopy(server)
        unit.actor_optimizer.zero_grad();unit.actor(torch.ones(2)).square().sum().backward();unit.actor_optimizer.step()
        old = [p.clone() for p in server.actor.parameters()]
        endpoint = [p.clone() for p in unit.actor.parameters()]
        half = interpolate_server_proposal(server,unit,.5)
        for p,a,b in zip(half.actor.parameters(),old,endpoint):
            torch.testing.assert_close(p,a+.5*(b-a))
        self.assertEqual(len(server.actor_optimizer.state),0)
        self.assertEqual(len(half.actor_optimizer.state),len(unit.actor_optimizer.state))
        self.assertEqual(len(interpolate_server_proposal(server,unit,0).actor_optimizer.state),0)
        exact = interpolate_server_proposal(server,unit,1)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(exact.actor.parameters(),unit.actor.parameters())))
        for state in half.actor_optimizer.state.values():
            self.assertEqual(int(state["step"]),1)


if __name__ == "__main__":
    unittest.main()
