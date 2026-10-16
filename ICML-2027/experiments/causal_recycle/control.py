"""Predictable optimistic mirror descent with Lyapunov resource prices.

Both action and probe count are chosen BEFORE the new bank is opened. Full-
information contrasts and a sample variance feed only subsequent decisions.
No empirical variance is treated as a population upper confidence bound.
"""
import itertools
import math
import numpy as np


def simplex(v):
    v=np.asarray(v,dtype=float)
    if not np.isfinite(v).all():raise FloatingPointError("nonfinite simplex input")
    u=np.sort(v)[::-1];cs=np.cumsum(u)-1
    indices=np.arange(1,len(v)+1)
    rho=np.flatnonzero(u-cs/indices>0)[-1]
    out=np.maximum(v-cs[rho]/(rho+1),0)
    return out/out.sum()


class RecycleController:
    def __init__(self,agents,*,batch_episodes,horizon,frames,alphas=(0,.5,1),
                 probes=(4,8,16),outbound_target=.5,forward_target=.75,
                 eta_scale=25.,meta_scale=1.,use_dual=True):
        self.actions=list(itertools.product(alphas,repeat=agents))
        self.probes=np.asarray(probes,dtype=int)
        if min(probes)<2 or max(probes)>batch_episodes:raise ValueError("probe catalogue must fit training bank")
        self.batch=batch_episodes;self.horizon=horizon
        self.eta=eta_scale/math.sqrt(frames);self.zeta=meta_scale/math.sqrt(frames)
        self.V=math.sqrt(frames)
        self.outbound_target=outbound_target;self.forward_target=forward_target
        self.QC=self.QF=0.;self.use_dual=use_dual
        self.w=np.zeros(len(self.actions));self.w[self.actions.index(tuple([1]*agents))]=1.
        self.z=np.ones(len(self.probes))/len(self.probes)
        self.round=0;self._decision=None

    def decide(self,hints,costs,hint_fraction,rng,availability_fraction=1.):
        if self._decision is not None:raise RuntimeError("previous bank has not closed")
        hints=np.asarray(hints,dtype=float);costs=np.asarray(costs,dtype=float)
        qc=self.QC if self.use_dual else 0.;qf=self.QF if self.use_dual else 0.
        action_prob=simplex(self.w+self.eta*(hints-qc/self.V*costs))
        probe_cost=hint_fraction+availability_fraction*self.probes/self.batch
        probe_prob=simplex(self.z-self.zeta*qf*probe_cost)
        action=int(rng.choice(len(self.actions),p=action_prob))
        probe=int(rng.choice(len(self.probes),p=probe_prob))
        self._decision={"action":action,"probe":probe,"action_prob":action_prob,
                        "probe_prob":probe_prob,"costs":costs,"probe_cost":probe_cost,
                        "QC":qc,"QF":qf,"QC_actual":self.QC,"QF_actual":self.QF}
        return action,int(self.probes[probe])

    def feedback(self,contrast_samples,actual_outbound_fraction,actual_forward_fraction):
        if self._decision is None:raise RuntimeError("decision must precede feedback")
        x=np.asarray(contrast_samples,dtype=float)
        d=self._decision
        if abs(actual_outbound_fraction-d["costs"][d["action"]])>1e-10:
            raise AssertionError("action cost differs from predictable charge")
        if abs(actual_forward_fraction-d["probe_cost"][d["probe"]])>1e-10:
            raise AssertionError("probe cost differs from predictable charge")
        if x.shape!=(self.probes[d["probe"]],len(self.actions)) or not np.isfinite(x).all():
            raise ValueError("finite independent episode contrasts with registered shape required")
        normalized=x/self.horizon
        gain=normalized.mean(axis=0)
        trace=float(normalized.var(axis=0,ddof=1).sum())
        # The forecast hint can be biased; this fresh feedback may not be.
        self.w=simplex(self.w+self.eta*(gain-d["QC"]/self.V*d["costs"]))
        moment_loss=.5*self.V*self.eta*trace/self.probes
        self.z=simplex(self.z-self.zeta*(d["QF"]*d["probe_cost"]+moment_loss))
        self.QC=max(0.,self.QC+actual_outbound_fraction-self.outbound_target)
        self.QF=max(0.,self.QF+actual_forward_fraction-self.forward_target)
        record={"round":self.round,"action_index":d["action"],"probe_episodes":int(self.probes[d["probe"]]),
                "action":list(self.actions[d["action"]]),"action_probability":d["action_prob"].tolist(),
                "probe_probability":d["probe_prob"].tolist(),"sample_variance_trace":trace,
                "selected_gain_estimate":float(gain[d["action"]]),"QC_before":d["QC_actual"],"QF_before":d["QF_actual"],
                "QC_price":d["QC"],"QF_price":d["QF"],
                "QC_after":self.QC,"QF_after":self.QF,
                "outbound_fraction":actual_outbound_fraction,"forward_fraction":actual_forward_fraction}
        self.round+=1;self._decision=None
        return record
