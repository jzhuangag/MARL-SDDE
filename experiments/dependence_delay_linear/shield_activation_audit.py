"""Public-model Gaussian activation diagnostic; no learning trajectories."""
import math
import numpy as np


def ar1_quadratic(coefficients, lam):
    total, past = 0., 0.
    for c in coefficients:
        total += c*c+2*c*past
        past = lam*(past+c)
    return max(0.,float(total))


def activation_path(targets, m, eta, lam, variance, initial, n=4, delta=.05):
    targets=np.asarray(targets,dtype=float)
    if targets.ndim!=1 or not targets.size or not np.all(np.isfinite(targets)):
        raise ValueError('finite scalar target schedule required')
    if m<1 or int(m)!=m or not 0<eta<1 or not 0<=lam<1 or variance<0:
        raise ValueError('invalid model parameters')
    if not np.isfinite(variance) or not np.isfinite(initial) or n<1 or int(n)!=n or not 0<delta<1:
        raise ValueError('invalid confidence parameters')
    T=len(targets)
    H=(m+2*sum((m-h)*lam**h for h in range(1,m)))/m**2
    radius=math.sqrt(2*variance*H*math.log(2*n*T/delta))
    mu=float(initial)
    out=[]
    for k,theta in enumerate(targets):
        mu=(1-eta)**m*mu+(1-(1-eta)**m)*theta
        N=(k+1)*m
        coefficients=eta*(1-eta)**np.arange(N-1,-1,-1,dtype=float)
        coefficients[-m:] -= 1/m
        var=variance*ar1_quadratic(coefficients,lam)
        bias=mu-theta
        if var==0:
            probability=float(abs(bias)>radius)
        else:
            sd=math.sqrt(var)
            probability=.5*math.erfc((radius+bias)/(sd*math.sqrt(2)))+.5*math.erfc((radius-bias)/(sd*math.sqrt(2)))
        out.append(dict(block=k,bias=float(bias),variance=var,radius=radius,probability=probability))
    return out
