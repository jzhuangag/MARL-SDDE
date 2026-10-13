import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from acquisition_gate import gated_select, perfect_information_is_futile, public_fallback
from efficient_critic import efficient_actions, efficient_true_risk


def test_futility_endpoints_imply_the_entire_correlation_continuum():
    config={"calibration_workers":8,"calibration_innovations":256,
            "q_catalogue":[1,2,4,8],"spacing_catalogue":[1,4,16],"message_budget":65536}
    cell={"overhead":4,"actor_budget":8192,"persistence":.9}
    paid, unpaid = efficient_actions(config,cell,paid=True), efficient_actions(config,cell,paid=False)
    fallback=public_fallback(unpaid,[0.,.2,.5,.8,1.])
    assert perfect_information_is_futile(paid,fallback)
    def forbidden_observation():
        raise AssertionError("a skipped acquisition must not read its sensor")
    selected,skipped=gated_select(paid,unpaid,[0.,.2,.5,.8,1.],forbidden_observation)
    assert skipped and selected==fallback
    for rho in np.linspace(0,1,101):
        assert all(efficient_true_risk(a,rho)>=efficient_true_risk(fallback,rho)-1e-12 for a in paid)


def test_full_catalogue_is_retained_when_probing_can_be_valuable():
    config={"calibration_workers":8,"calibration_innovations":256,
            "q_catalogue":[1,2,4,8],"spacing_catalogue":[1,4,16],"message_budget":65536}
    cell={"overhead":16,"actor_budget":131072,"persistence":0.}
    paid, unpaid=efficient_actions(config,cell,paid=True),efficient_actions(config,cell,paid=False)
    fallback=public_fallback(unpaid,[0.,.2,.5,.8,1.])
    assert not perfect_information_is_futile(paid,fallback)
    assert {a.q for a in paid}=={1,2,4,8}
