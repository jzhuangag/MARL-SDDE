"""Outcome-free regression witnesses for law alignment and variance bounds."""
import numpy as np
import pytest

from experiments.dependence_delay_linear.t068_safe_personalized_mixing import initial_moment_state
from experiments.dependence_delay_linear.t079_continuous_static_graph import propagate_continuous_graph_block


@pytest.mark.parametrize('lam', [0., .8])
def test_two_block_identity_graph_exposes_missing_cross_covariance(lam):
    state = initial_moment_state([0.,0.], [0.,0.], delay=0)
    for _ in range(2):
        result = propagate_continuous_graph_block(
            state, targets=[0.,0.], gain=.2, curvature=1., local_steps=1,
            noise_scale=1., spatial_correlation=0., temporal_correlation=lam,
            fixed_weights=np.eye(2))
        state = result.state
    legacy = result.personalized_risk[0]
    continuous = .2**2*(.8**2+1+2*.8*lam)
    np.testing.assert_allclose(legacy, .0656)
    np.testing.assert_allclose(continuous-legacy, 2*.2**2*.8*lam, atol=1e-14)
    if lam:
        assert continuous > legacy


@pytest.mark.parametrize('m', [1,2,8,32])
def test_block_mean_variance_formula_and_monotone_upper_bound(m):
    lam, upper = .7, .9
    lags = np.abs(np.subtract.outer(np.arange(m),np.arange(m)))
    exact = np.sum(lam**lags)/m**2
    formula = (m+2*sum((m-h)*lam**h for h in range(1,m)))/m**2
    bound = (m+2*sum((m-h)*upper**h for h in range(1,m)))/m**2
    np.testing.assert_allclose(exact,formula)
    assert exact <= bound+1e-14
    assert bound <= min(1., (1+upper)/(m*(1-upper)))+1e-14
