"""Audit-only fixed-graph moments, matching frozen T081 history semantics."""
import numpy as np
from experiments.dependence_delay_linear.continuous_ar1_covariance_audit import covariance_path


def fixed_moments(*, targets, steps_per_block, gain, delay, weights,
                  decision_blocks, initial_parameter, spatial_covariance,
                  temporal_correlation):
    target = np.asarray(targets, dtype=float)
    w = np.asarray(weights, dtype=float)
    if target.ndim != 2 or not np.all(np.isfinite(target)):
        raise ValueError('targets must be finite blocks by agents')
    blocks,n = target.shape
    if blocks < 1 or n < 1 or w.shape != (n,n) or not np.all(np.isfinite(w)):
        raise ValueError('invalid dimensions or weights')
    if np.min(w)<0 or not np.allclose(w.sum(1),1):
        raise ValueError('weights must be row stochastic')
    if delay < 0 or int(delay)!=delay or steps_per_block<1 or int(steps_per_block)!=steps_per_block:
        raise ValueError('invalid delay or block length')
    if not 0<gain<1 or not np.isfinite(initial_parameter):
        raise ValueError('invalid learner parameters')
    decisions = set(decision_blocks)
    if any(int(k)!=k or k<0 or k>=blocks for k in decisions):
        raise ValueError('invalid decision block')
    dim = n*(delay+2)  # current, oldest-to-newest history, shadow
    shadow = slice(dim-n,dim)
    maps, offsets = [],[]
    for k in range(blocks):
        for step in range(steps_per_block):
            a,b = np.eye(dim),np.zeros((dim,n))
            a[:n,:n] *= 1-gain
            a[shadow,shadow] *= 1-gain
            b[:n] = gain*np.eye(n)
            b[shadow] = gain*np.eye(n)
            if step == steps_per_block-1:
                mix = np.eye(dim)
                if k in decisions:
                    mix[:n] = 0
                    if delay == 0:
                        mix[:n,:n] = w
                    else:
                        mix[:n,:n] = np.diag(np.diag(w))
                        mix[:n,n:2*n] = w-np.diag(np.diag(w))
                shift = np.eye(dim)
                if delay:
                    shift[n:(delay+1)*n] = 0
                    if delay>1:
                        shift[n:delay*n,2*n:(delay+1)*n] = np.eye((delay-1)*n)
                    shift[delay*n:(delay+1)*n,:n] = np.eye(n)
                post = shift @ mix
                a,b = post @ a, post @ b
            maps.append((a,b))
            offsets.append(b @ target[k])
    covariances = covariance_path(maps,spatial_covariance,temporal_correlation)
    mean = np.full(dim,float(initial_parameter))
    means = []
    for (a,_),offset in zip(maps,offsets):
        mean = a @ mean + offset
        means.append(mean.copy())
    end = np.arange(steps_per_block-1,len(maps),steps_per_block)
    means = np.asarray(means)[end]
    covariances = covariances[end]
    risk = ((means[:,:n]-target)**2+np.diagonal(covariances[:,:n,:n],axis1=1,axis2=2)).mean(1)
    return {'risk_path':risk,'mean_path':means,'covariance_path':covariances}
