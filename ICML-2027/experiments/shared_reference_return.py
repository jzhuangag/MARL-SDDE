"""Unclipped, reference-centered actual-return importance estimators.

State-value control variates are frozen before the reference bank. Their
accuracy affects variance, not the expectation identity. No critic loss is
used as the learning-value target.
"""
import numpy as np


def contrasts(log_ratios,rewards,values):
    lr=np.asarray(log_ratios,dtype=float)
    r=np.asarray(rewards,dtype=float)
    v=np.asarray(values,dtype=float)
    if lr.shape!=r.shape or v.shape!=r.shape:
        raise ValueError("episode/time matrices must have matching shapes")
    w=np.exp(np.cumsum(lr,axis=1))
    if not np.isfinite(w).all():
        raise FloatingPointError("unclipped importance weights overflow")
    following=np.concatenate([v[:,1:],np.zeros((len(v),1))],axis=1)
    g=r.sum(axis=1)
    return {"trajectory_is":(w[:,-1]-1)*g,
            "centered_trajectory_is":(w[:,-1]-1)*(g-v[:,0]),
            "per_decision_is":((w-1)*r).sum(axis=1),
            "td_control_variate":((w-1)*(r+following-v)).sum(axis=1)},w


def exact_identity_audit():
    """Enumerate a finite two-step MDP with intentionally wrong frozen V."""
    import itertools
    p=[.7,.4];q=[.3,.6]
    totals={m:0.0 for m in contrasts(np.zeros((1,2)),np.zeros((1,2)),np.zeros((1,2)))[0]}
    target=0.0
    for actions in itertools.product(range(2),repeat=2):
        pp=np.array([p[t] if a else 1-p[t] for t,a in enumerate(actions)])
        qq=np.array([q[t] if a else 1-q[t] for t,a in enumerate(actions)])
        reward=np.array([2*actions[0]-.4,actions[1]+3*actions[0]*actions[1]])
        # V_t can depend on the observed pre-action state, but not the current
        # action. Deliberately inaccurate values exercise the unbiasedness.
        value=np.array([17.0,-11.0+5*actions[0]])
        xx,_=contrasts(np.log(pp/qq)[None],reward[None],value[None])
        target+=(pp.prod()-qq.prod())*reward.sum()
        for method,x in xx.items():
            totals[method]+=qq.prod()*float(x[0])
    if any(abs(x-target)>1e-12 for x in totals.values()):
        raise AssertionError("relative importance/control-variate identity failed")
    return {"exact_return_difference":float(target),"expectations":totals,
            "maximum_absolute_error":max(abs(x-target) for x in totals.values())}
