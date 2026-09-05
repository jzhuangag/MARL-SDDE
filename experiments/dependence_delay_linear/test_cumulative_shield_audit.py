"""Algebraic witnesses, not controller efficacy or frozen protocol changes."""
import math
import pytest


def robust_excess(z,s,h,r):
    return (z-h)**2-(s-h)**2+2*r*abs(z-s)


def test_credit_can_spend_only_certified_improvement():
    credit=0.
    total=0.
    # Exact target h=0: earn by correcting a shadow at 1, then spend on shadow 0.
    for z,s in [(0.,1.),(.5,0.)]:
        u=robust_excess(z,s,0.,0.)
        assert u<=credit
        credit-=u
        total+=z*z-s*s
        assert total<=0
    assert credit==.75


@pytest.mark.parametrize('c',[-2.,0.,2.])
@pytest.mark.parametrize('allowance',[0.,.1,3.])
def test_segment_root_is_feasible_and_maximal(c,allowance):
    v=1.3
    disc=math.sqrt(c*c+v*v*allowance)
    root=allowance/(disc+c) if c>0 else (-c+disc)/(v*v)
    beta=min(1.,root)
    assert v*v*beta*beta+2*c*beta <= allowance+1e-12
    if beta<1:
        other=beta+1e-5
        assert v*v*other*other+2*c*other > allowance


def test_no_free_credit_inside_interval():
    s,h,r=.2,0.,1.
    for z in [-10.,-1.,0.,.1,.3,1.,10.]:
        assert robust_excess(z,s,h,r)>0
    assert robust_excess(s,s,h,r)==0
