"""Exact symmetric linear-response integration through the frozen simulator."""
import numpy as np
import pytest
from experiments.dependence_delay_linear.fixed_end_block_moments_audit import fixed_moments
from experiments.dependence_delay_linear.t081_end_block_primal_dual_controller import simulate_end_block_controller


@pytest.mark.parametrize('delay',[0,1,3])
@pytest.mark.parametrize('lam',[0.,.8])
@pytest.mark.parametrize('decisions',[[0,1,2],[0,2]])
def test_moments_match_frozen_quadratic_response(delay,lam,decisions):
    targets=np.array([[.2,-.3],[.5,.1],[-.1,.4]])
    w=np.array([[.7,.3],[.2,.8]])
    sigma=np.array([[.6,.2],[.2,1.]])
    blocks,m,n=3,2,2
    center=np.repeat(targets[:,None,:],m,axis=1)
    def risks(obs):
        return simulate_end_block_controller(
            observations=obs,targets=targets,initial_parameter=.4,gain=.2,
            delay=delay,decision_blocks=decisions,drift_weight=1.,variance_weight=1.,
            safety_slack=0.,certificate_delta=.05,rho_cap=.95,fixed_weights=w).risk_path
    base=risks(center)
    temporal=lam**np.abs(np.subtract.outer(np.arange(blocks*m),np.arange(blocks*m)))
    root=np.linalg.cholesky(np.kron(temporal,sigma))
    expected=base.copy()
    for column in root.T:
        delta=column.reshape(blocks,m,n)
        expected += (risks(center+delta)+risks(center-delta))/2-base
    actual=fixed_moments(targets=targets,steps_per_block=m,gain=.2,delay=delay,
                         weights=w,decision_blocks=decisions,initial_parameter=.4,
                         spatial_covariance=sigma,temporal_correlation=lam)
    np.testing.assert_allclose(actual['risk_path'],expected,atol=1e-12,rtol=1e-11)
