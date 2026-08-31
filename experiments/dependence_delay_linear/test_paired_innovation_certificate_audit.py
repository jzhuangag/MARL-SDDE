"""Deterministic moment identities, not empirical coverage or efficacy tests."""
import math
import numpy as np
import pytest


@pytest.mark.parametrize('lam',[0.,.5,.9])
@pytest.mark.parametrize('m',[2,8])
def test_innovation_estimator_covariance(lam,m):
    coefficients=np.zeros(m)
    for s in range(1,m):
        coefficients[s]+=1
        coefficients[s-1]-=lam
    coefficients/=(m-1)*(1-lam)
    np.testing.assert_allclose(coefficients.sum(),1.)
    R=lam**np.abs(np.subtract.outer(np.arange(m),np.arange(m)))
    np.testing.assert_allclose(coefficients@R@coefficients,(1+lam)/((m-1)*(1-lam)),rtol=1e-12)


def test_boundary_state_cancels():
    lam,theta=.7,2.
    innovations=np.array([.2,-.3,.4,.1])
    for boundary in [-10.,0.,10.]:
        state=boundary
        obs=[]
        for u in innovations:
            state=lam*state+u
            obs.append(theta+state)
        h=sum(obs[s]-lam*obs[s-1] for s in range(1,4))/(3*(1-lam))
        np.testing.assert_allclose(h-theta,innovations[1:].sum()/(3*(1-lam)),atol=1e-14)


def test_normal_mixture_boundary_equation():
    alpha,v0=.05,1.3
    for V in [0.,.1,1.,100.]:
        b=math.sqrt((V+v0)*(2*math.log(1/alpha)+math.log1p(V/v0)))
        logM=.5*math.log(v0/(V+v0))+b*b/(2*(V+v0))
        np.testing.assert_allclose(logM,math.log(1/alpha))


def test_endogenous_displacement_is_not_centered():
    # A symmetric two-point witness is enough to disprove algebraic centering.
    noise=np.array([-1.,1.])
    predictable=.3
    assert np.mean(-2*predictable*noise)==0
    assert np.mean(-2*noise*noise)==-2
