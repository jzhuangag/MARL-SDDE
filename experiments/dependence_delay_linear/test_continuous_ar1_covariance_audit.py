"""Independent full Toeplitz covariance checks, without sampled outcomes."""
import numpy as np
import pytest
from experiments.dependence_delay_linear.continuous_ar1_covariance_audit import covariance_path


def direct_path(maps, sigma, lam):
    dimension, n = maps[0][1].shape
    coefficients = np.zeros((dimension,0))
    result = []
    for k,(a,b) in enumerate(maps,1):
        coefficients = np.column_stack([a @ coefficients, b])
        temporal = lam**np.abs(np.subtract.outer(np.arange(k),np.arange(k)))
        result.append(coefficients @ np.kron(temporal,sigma) @ coefficients.T)
    return np.asarray(result)


@pytest.mark.parametrize('lam',[0.,.4,.9])
@pytest.mark.parametrize('delay',[0,1,3])
def test_augmented_matches_full_toeplitz_with_delay(lam,delay):
    n, dimension = 2, 2*(delay+1)
    sigma = np.array([[1.,.3],[.3,.7]])
    maps = []
    for t in range(8):
        a = np.zeros((dimension,dimension))
        b = np.zeros((dimension,n))
        w = np.array([[.8,.2],[.1,.9]]) if t%2 else np.eye(n)
        if delay:
            # Deterministic delayed linear collaboration, not the frozen block wrapper.
            a[:n,:n] = .65*np.eye(n)
            a[:n,-n:] += .15*w
            a[n:,:-n] = np.eye(dimension-n)
        else:
            a[:n,:n] = .8*w
        b[:n] = .2*np.eye(n)
        maps.append((a,b))
    observed = covariance_path(maps,sigma,lam)
    np.testing.assert_allclose(observed,direct_path(maps,sigma,lam),atol=1e-13,rtol=1e-12)
    assert min(np.linalg.eigvalsh(c).min() for c in observed) >= -1e-12


def test_two_step_closed_form():
    maps = [(np.array([[.8]]),np.array([[.2]]))]*2
    np.testing.assert_allclose(covariance_path(maps,[[1.]],.8)[-1,0,0],.1168)


@pytest.mark.parametrize('sigma,lam',[([[1.,2.],[2.,1.]],.5),([[1.]],1.),([[float('nan')]],.5)])
def test_reject_invalid_law(sigma,lam):
    with pytest.raises(ValueError):
        covariance_path([(np.eye(1),np.ones((1,len(sigma))))],sigma,lam)
