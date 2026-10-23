import unittest
import numpy as np
import torch
from control import simplex,RecycleController
from packet import NetworkPackets,adam_variant,digest
from native import base


class InterfaceTests(unittest.TestCase):
    def test_simplex_and_projection_regret(self):
        rng=np.random.default_rng(99)
        for _ in range(100):
            w=simplex(rng.normal(size=9));u=simplex(rng.normal(size=9));hint=rng.normal(size=9);y=rng.normal(size=9);eta=.7
            p=simplex(w+eta*hint);following=simplex(w+eta*y)
            lhs=np.dot(u-p,y)
            rhs=(sum((u-w)**2)-sum((u-following)**2))/(2*eta)+eta/2*sum((y-hint)**2)
            self.assertLessEqual(lhs,rhs+1e-10)

    def test_predictable_cost_recurrence(self):
        c=RecycleController(2,batch_episodes=16,horizon=25,frames=40)
        rng=np.random.default_rng(71);cost=np.asarray([sum(x>0 for x in a)/2 for a in c.actions])
        a,m=c.decide(np.zeros(9),cost,.1,rng,.5)
        with self.assertRaises(RuntimeError):c.decide(np.zeros(9),cost,.1,rng)
        record=c.feedback(np.zeros((m,9)),cost[a],.1+.5*m/16)
        self.assertEqual(record["sample_variance_trace"],0)
        self.assertEqual(c.round,1)
        with self.assertRaises(RuntimeError):c.feedback(np.zeros((m,9)),cost[a],.1)

    def test_service_independent_of_selection(self):
        a=NetworkPackets(99,2);b=NetworkPackets(99,2);g=[torch.ones(2)]
        for t in range(20):
            a.commit([1,0]);b.commit([0,1])
            for i in range(2):
                ja=a.launch(t,i,tuple(a.versions),g,1,8);jb=b.launch(t,i,tuple(b.versions),g,1,8)
                self.assertEqual(ja["due"],jb["due"])
            a.receive_latest(t);b.receive_latest(t)
        a.drain();b.drain();self.assertEqual(a.transmitted_gradient_bytes,320)

    def test_counterfactual_keeps_server_moments(self):
        p=base.config();p["actor_optimizer_steps_per_packet"]=1
        actors,_,_=base.initialize({"name":"reference","scenario":"simple_reference_v3"},79219990,p)
        server=actors[0];g=[torch.ones_like(v)*.01 for v in server.actor.parameters()]
        first=adam_variant(server,g,1.);before=digest(list(first.actor.parameters()))
        next_actor=adam_variant(first,g,.5)
        self.assertEqual(before,digest(list(first.actor.parameters())))
        self.assertIs(adam_variant(first,g,0),first)
        self.assertTrue(all(float(x["step"])==2 for x in next_actor.actor_optimizer.state.values()))
        self.assertTrue(all(float(x["step"])==1 for x in first.actor_optimizer.state.values()))


if __name__=="__main__":
    torch.set_num_threads(1);unittest.main()
