"""Frozen, training-only finite-horizon Monte Carlo control variate.

Unlike the learner's discounted critic, this fits undiscounted return-to-go
and includes remaining episode time. Its errors affect only estimator variance.
"""
import numpy as np


class MonteCarloPotential:
    def __init__(self,horizon=25,ridge=1.0):
        self.horizon=horizon;self.ridge=ridge

    def features(self,states):
        x=np.tanh((states-self.location)/self.scale)
        tau=np.tile(np.arange(self.horizon,0,-1)/self.horizon,len(states)//self.horizon)[:,None]
        return np.concatenate([tau,tau*tau,tau*x,tau*x*x],axis=1)

    def fit(self,blocks):
        states=np.concatenate([b["states"] for b in blocks]).astype(float)
        rewards=np.concatenate([b["rewards"] for b in blocks]).astype(float)
        target=np.cumsum(rewards[:,::-1],axis=1)[:,::-1].reshape(-1)
        self.location=states.mean(axis=0)
        self.scale=np.maximum(states.std(axis=0),.1)
        phi=self.features(states)
        self.coef=np.linalg.solve(phi.T@phi+self.ridge*np.eye(phi.shape[1]),phi.T@target)
        if not np.isfinite(self.coef).all():raise FloatingPointError("nonfinite potential fit")
        return self

    def predict(self,states):
        return (self.features(np.asarray(states,dtype=float))@self.coef).reshape(-1,self.horizon)
