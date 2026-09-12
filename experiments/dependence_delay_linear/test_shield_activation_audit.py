import numpy as np
import pytest
from experiments.dependence_delay_linear.shield_activation_audit import ar1_quadratic,activation_path


@pytest.mark.parametrize('lam',[0.,.5,.95])
def test_quadratic_matches_dense(lam):
    c=np.array([.2,-.5,.1,.9,-.3])
    R=lam**np.abs(np.subtract.outer(np.arange(5),np.arange(5)))
    np.testing.assert_allclose(ar1_quadratic(c,lam),c@R@c,atol=1e-14)


def test_zero_noise_and_sign_symmetry():
    a=activation_path([1.,-1.],2,.2,.5,0.,0.)
    assert all(row['probability']==1 for row in a)
    b=activation_path([1.,-1.],2,.2,.5,1.,.2)
    c=activation_path([-1.,1.],2,.2,.5,1.,-.2)
    np.testing.assert_allclose([r['probability'] for r in b],[r['probability'] for r in c])


def test_identical_shadow_and_mean_cannot_activate():
    # One sample, deterministic initial zero, eta<1: independent direct scalar check.
    result=activation_path([0.],1,.5,0.,1.,0.)[0]
    np.testing.assert_allclose(result['variance'],.25)
    assert 0<=result['probability']<.05
