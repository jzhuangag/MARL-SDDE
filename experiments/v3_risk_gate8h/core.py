"""Fixed-design Gaussian gate and direct learners; no hidden regime in controllers."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import itertools
import math
from pathlib import Path
import sys

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TSP-V3-revised-20261018/manuscript/tools"))
from risk_gate_core import InstanceRisk, admit, risk_upper

POLICIES = ("risk_corrected_gate", "no_carryover_gate", "information_only",
            "always_all", "fixed_q2_b1", "known_hypothesis_oracle_evaluation_only")
LEARNERS = ("running_mean", "averaged_delayed_affine")


@dataclass(frozen=True)
class Family:
    Q: int
    theta_high: float
    lam: float
    D: int
    theta_low: float = .05
    h: int = 8

    @property
    def key(self):
        return f"Q{self.Q}-th{self.theta_high:g}-l{self.lam:g}-D{self.D}"


@dataclass(frozen=True, order=True)
class Action:
    q: int
    b: int


def families(protocol):
    g = protocol["public_grid"]
    return tuple(Family(q, th, lam, d, g["theta_low"], g["h"])
                 for q, th, lam, d in itertools.product(g["Q"], g["theta_high"], g["lambda"], g["D"]))


def catalogue(family):
    return tuple(Action(2**p, b) for p in range(int(math.log2(family.Q))+1) for b in (1,2,4,8))


def scheduled(action, bm, be, h):
    if min(bm, be) < 0:
        raise ValueError("Negative remaining budget")
    return min(bm//(h+action.q), be//action.b)


@lru_cache(None)
def weights(learner, M, D):
    N = max(0, M-D)
    if N == 0:
        return 1., np.zeros(0)
    if learner == LEARNERS[0]:
        d, w = 1/(N+1), np.full(N, 1/(N+1))
    elif learner == LEARNERS[1]:
        mu, eta = .5, .2/(D+1)
        impulse = np.empty(N)
        impulse[0] = 1.
        for t in range(1, N):
            impulse[t] = impulse[t-1]-(eta*mu*impulse[t-D-1] if t>D else 0.)
        w = eta/N*np.cumsum(impulse)[::-1]
        d = float(1-mu*w.sum())
    else:
        raise ValueError(learner)
    w.flags.writeable = False
    return float(d), w


@lru_cache(None)
def program_risk(learner, action, bm, be, family, theta):
    M = scheduled(action, bm, be, family.h)
    d, w = weights(learner, M, family.D)
    r = family.lam**action.b
    diagonal = float(w@w)
    filtered = lfilter([1.], [1., -r], w) if len(w) else w
    common = 2*float(w@filtered)-diagonal
    k = float(w@(r**np.arange(1, len(w)+1)))
    return float(d*d+theta*common+diagonal/action.q), d, k


def best_action(learner, bm, be, family, theta):
    return min(catalogue(family), key=lambda a: (program_risk(learner,a,bm,be,family,theta)[0],a.q,a.b))


def logdet_prefix(theta, q, lam, count):
    P, value = theta, 0.
    for _ in range(count):
        value += math.log1p(q*P)
        P /= 1+q*P
        P = lam*lam*P+(1-lam*lam)*theta
    return value


@lru_cache(None)
def probe_plan(family):
    """O(n) exact determinant-affinity recursion, smallest n, no observations."""
    thetas = (family.theta_low, family.theta_high, (family.theta_low+family.theta_high)/2)
    P, logdet = list(thetas), [0.,0.,0.]
    for n in range(1,4097):
        for j, theta in enumerate(thetas):
            logdet[j] += math.log1p(family.Q*P[j])
            P[j] /= 1+family.Q*P[j]
            P[j] = family.lam**2*P[j]+(1-family.lam**2)*theta
        affinity = min(1.,math.exp(.25*(logdet[0]+logdet[1])-.5*logdet[2]))
        if affinity <= .025:
            return n, affinity
    return 0, 1.


def posterior_variance(theta, lam, q, n):
    P = theta
    for t in range(n):
        P /= 1+q*P
        if t+1 < n:
            P = lam*lam*P+(1-lam*lam)*theta
    return P


@lru_cache(None)
def design(learner, family, scale):
    """Both hypothetical instances, fixed programs and ledger; no true regime."""
    bm, be = (family.h+family.Q)*scale, 4*scale
    n, delta = probe_plan(family)
    pm, pe = n*(family.h+family.Q), n
    feasible = n>0 and pm<=bm and pe<=be
    fallback = Action(family.Q,1)
    result = dict(bm=bm, be=be, probe_n=n, delta=delta, feasible=feasible,
                  probe_messages=pm, probe_environment=pe, fallback=fallback)
    if not feasible:
        result.update(low=fallback, high=fallback, corrected=False, no_carryover=False,
                      zero_allowance=False, U0=None, U1=None, carry0=None, carry1=None)
        return result
    low = best_action(learner,bm-pm,be-pe,family,family.theta_low)
    high = best_action(learner,bm-pm,be-pe,family,family.theta_high)
    certificates = []
    for theta, correct, wrong in ((family.theta_low,low,high),(family.theta_high,high,low)):
        jc,dc,kc = program_risk(learner,correct,bm-pm,be-pe,family,theta)
        jw,dw,kw = program_risk(learner,wrong,bm-pm,be-pe,family,theta)
        jf = program_risk(learner,fallback,bm,be,family,theta)[0]
        s2 = max(0.,theta-posterior_variance(theta,family.lam,family.Q,n))
        certificates.append(InstanceRisk(jc,jw,dc,dw,kc,kw,s2,delta,jf))
    (u0,c0),(u1,c1) = map(risk_upper,certificates)
    result.update(low=low,high=high,U0=u0,U1=u1,carry0=c0,carry1=c1,
                  corrected=admit(*certificates,.1,True),
                  no_carryover=(u1-c1<certificates[1].fallback_J and u0-c0<=certificates[0].fallback_J+.1),
                  zero_allowance=admit(*certificates,0.,True))
    return result


def likelihood(values, theta, q, lam):
    m, P, value = 0.,theta,0.
    root = math.sqrt(q)
    for y in values:
        S = 1+q*P
        e = float(y)-root*m
        value -= .5*(math.log(2*math.pi*S)+e*e/S)
        m += P*root/S*e
        P /= 1+q*P
        m *= lam
        P = lam*lam*P+(1-lam*lam)*theta
    return value


def identify(observations, family):
    y = math.sqrt(family.Q)*observations
    llr = likelihood(y,family.theta_high,family.Q,family.lam)-likelihood(y,family.theta_low,family.Q,family.lam)
    return ("high" if llr>0 else "low"), float(llr)


def deployment_decision(policy, public_design, identified):
    """No actual regime, empirical risks, future observations or seed as inputs."""
    if policy == "fixed_q2_b1":
        return False,Action(2,1)
    wants = (public_design["feasible"] and
             (policy=="information_only" or
              (policy=="risk_corrected_gate" and public_design["corrected"]) or
              (policy=="no_carryover_gate" and public_design["no_carryover"])))
    if wants:
        if identified not in ("low","high"):
            raise ValueError("Admitted probe requires its observed likelihood decision")
        return True,public_design[identified]
    return False,public_design["fallback"]


def potential_observations(seed, family, regime, learner, length):
    key = f"V3-RISK-GATE-8H:{seed}:{family.key}:{regime}:{learner}".encode()
    digest = hashlib.sha256(key).digest()
    rng_seed = digest.hex()
    rng = np.random.RandomState([int.from_bytes(digest[i:i+4],"big") for i in range(0,32,4)])
    theta = family.theta_low if regime=="low" else family.theta_high
    innovations = rng.normal(size=length)
    common = np.empty(length)
    common[0] = math.sqrt(theta)*innovations[0]
    factor = math.sqrt(theta*(1-family.lam**2))
    for t in range(1,length):
        common[t] = family.lam*common[t-1]+factor*innovations[t]
    private = np.cumsum(rng.normal(size=(family.Q,length)),axis=0)
    return common,private,rng_seed


def observations(world, action, start, M):
    common,private = world[:2]
    indices = start+action.b*np.arange(1,M+1)
    if M and indices[-1]>=len(common):
        raise AssertionError("Charged horizon exceeds potential stream")
    return common[indices]+private[action.q-1,indices]/action.q


def direct_learner(values, learner, D, check=lambda: None):
    N = max(0,len(values)-D)
    if learner == LEARNERS[0]:
        return float((1+values[:N].sum())/(N+1)),N
    if learner != LEARNERS[1]:
        raise ValueError(learner)
    parameter, total, applied = 1.,0.,0
    pending = []
    eta = .2/(D+1)
    for t,y in enumerate(values):
        if t%256==0:
            check()
        pending.append(eta*(float(y)-.5*parameter))
        if len(pending)>D:
            parameter += pending.pop(0)
            total += parameter
            applied += 1
    assert applied==N
    return (parameter if N==0 else total/N),N


def group_rows(seed,family,regime,learner,scales,check=lambda: None):
    """Actual simulation; called only by the admitted native worker after handoff."""
    check()
    world = potential_observations(seed,family,regime,learner,4*max(scales)+1)
    for scale in scales:
        public = design(learner,family,scale)
        n = public["probe_n"]
        decision,llr = identify(observations(world,Action(family.Q,1),0,n),family) if public["feasible"] else (None,None)
        for policy in POLICIES:
            check()
            if policy==POLICIES[-1]:
                theta = family.theta_low if regime=="low" else family.theta_high
                used,action = False,best_action(learner,public["bm"],public["be"],family,theta)
            else:
                used,action = deployment_decision(policy,public,decision)
            pm,pe = (public["probe_messages"],n) if used else (0,0)
            M = scheduled(action,public["bm"]-pm,public["be"]-pe,family.h)
            values = observations(world,action,pe,M)
            estimate,N = direct_learner(values,learner,family.D,check)
            theta = family.theta_low if regime=="low" else family.theta_high
            jf = program_risk(learner,public["fallback"],public["bm"],public["be"],family,theta)[0]
            corrected = policy==POLICIES[0]
            upper = public["U0" if regime=="low" else "U1"] if corrected and used else (jf if corrected else None)
            yield dict(seed=seed,family=family.key,Q=family.Q,theta_high=family.theta_high,
                               lambda_public=family.lam,D=family.D,scale=scale,regime=regime,learner=learner,
                               policy=policy,stream_seed=world[2],terminal_estimate=estimate,
                               terminal_squared_risk=estimate*estimate,probe_used=bool(used),
                               probe_n=n if used else 0,probe_q=family.Q if used else 0,probe_b=1 if used else 0,
                               identified=decision if used else None,log_likelihood_ratio=llr if used else None,
                               direction_error=bool(decision!=regime) if used else None,
                               candidate_feasible=bool(public["feasible"]),
                               corrected_admit=bool(public["corrected"]),zero_allowance_admit=bool(public["zero_allowance"]),
                               q=action.q,b=action.b,scheduled=M,usable=N,
                               message_budget=public["bm"],environment_budget=public["be"],
                               messages_used=pm+M*(family.h+action.q),environment_used=pe+M*action.b,
                               directional_bound=public["delta"] if used else None,
                               certified_population_upper=upper,fallback_population_risk=jf,
                               initialization=1.,physical_startup_charge=0)
