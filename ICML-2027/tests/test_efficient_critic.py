import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from efficient_critic import efficient_estimate, information_factor, innovation_gain


@pytest.mark.parametrize("n,r",[(1,0.),(8,.6),(17,.9),(45,0.)])
def test_information_matches_full_covariance_inverse(n,r):
    covariance = r**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    exact = np.ones(n)@np.linalg.solve(covariance,np.ones(n))
    assert information_factor(n,r) == pytest.approx(exact)


def test_causal_gain_estimator_matches_gls():
    r, n = .83, 27
    data = np.random.default_rng(17).normal(size=(n,9))
    covariance = r**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    weights = np.linalg.solve(covariance,np.ones(n))
    weights /= weights.sum()
    estimate = efficient_estimate(data,r)
    assert np.allclose(estimate[-1],weights@data,rtol=1e-12,atol=1e-12)
    assert np.allclose(efficient_estimate(data[:11],r)[-1],estimate[10])


def test_gain_minimizes_exact_lyapunov_risk():
    r, n = .7, 18
    v = 1/information_factor(n-1,r)
    w = (1+r)/(1-r)
    gain = innovation_gain(n,r)
    assert gain == pytest.approx(v/(v+w))
    best = (1-gain)**2*v+gain**2*w
    assert best == pytest.approx(1/information_factor(n,r))
    assert all(best <= (1-k)**2*v+k*k*w+1e-12 for k in np.linspace(0,1,101))
