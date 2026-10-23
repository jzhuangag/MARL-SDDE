"""Exact finite-MRP construction checks, not a sampled efficacy experiment.

No random seeds, files, training outcomes or experiment identifier are created.
The online protocol has no access to true transition probabilities. Fractions
and exhaustive enumeration are proof checks, not claimed production complexity.
"""

from collections import deque
from fractions import Fraction as F
from itertools import product
from math import comb
import json


class DelayedFlipPair:
    """Two private value estimates; fixed-lag, chronological paired updates."""

    def __init__(self, gamma, eta, weight, warmup, delay, rewards=(1, 2),
                 *, env_budget, message_budget, header_bits):
        self.gamma, self.eta, self.weight = map(F, (gamma, eta, weight))
        if not 0 < self.gamma < 1:
            raise ValueError("discount must be in (0,1)")
        cap = min(F(1, 2), (1-self.gamma)/(4*self.gamma))
        if not 0 <= self.weight <= cap:
            raise ValueError("weight outside the public pathwise-stability cap")
        if not 0 < self.eta <= 1/(1+self.gamma+2*self.gamma*cap):
            raise ValueError("step outside the public stability cap")
        if any(type(x) is not int for x in (warmup, delay, header_bits,
                                            env_budget, message_budget)):
            raise ValueError("counts and budgets must be integers")
        if warmup < 1 or delay < 0 or header_bits < 0 or min(env_budget, message_budget) < 0:
            raise ValueError("invalid count or budget")
        self.m, self.delay, self.header_bits = warmup, delay, header_bits
        self.env_budget, self.message_budget = env_budget, message_budget
        self.rewards = tuple(map(F, rewards))
        if len(self.rewards) != 2 or min(self.rewards) <= 0:
            raise ValueError("two positive, public reward scales required")
        self.u = [F(0), F(0)]
        self.counts = [0, 0]
        self.processed = self.ticks = self.env_used = self.message_used = 0
        self.pending = deque()

    def step(self, xi, xj):
        if xi not in (0, 1) or xj not in (0, 1):
            raise ValueError("binary flip observations required")
        cost = 2*(self.header_bits+1)
        if self.env_used+2 > self.env_budget or self.message_used+cost > self.message_budget:
            raise ValueError("next global tick would exceed a budget")
        self.env_used += 2
        self.message_used += cost  # Includes messages still pending at deadline.
        self.pending.append((self.ticks, xi, xj))
        self.ticks += 1
        if len(self.pending) <= self.delay:
            return tuple(self.u)
        birth, xi, xj = self.pending.popleft()
        assert birth == self.processed
        if self.processed < self.m:
            self.counts[0] += xi
            self.counts[1] += xj
            z = (F(xi), F(xj))
        else:
            delta = F(self.counts[0]-self.counts[1], self.m)
            w = self.weight
            z = ((1-w)*xi+w*(xj+delta), (1-w)*xj+w*(xi-delta))
        for i in (0, 1):
            A = 1-self.gamma+2*self.gamma*z[i]
            self.u[i] += self.eta*(self.rewards[i]-A*self.u[i])
        self.processed += 1
        return tuple(self.u)


def pair_law(pi, pj):
    return [(x, y, (pi if x else 1-pi)*(pj if y else 1-pj))
            for x, y in product((0, 1), repeat=2)]


def risk_by_moments(pi, pj, gamma, eta, w, m, D, T, reward=F(1)):
    """Exact evaluator. True pi,pj are used here only, NEVER by the protocol."""
    # Each bucket stores probability, unnormalized first and second moments.
    buckets = {(0, 0): (F(1), F(0), F(0))}
    H = max(0, T-D)
    for k in range(min(m, H)):
        nxt = {}
        for (ci, cj), (p, M, Q) in buckets.items():
            for x, y, prob in pair_law(pi, pj):
                f = 1-eta*(1-gamma+2*gamma*x)
                old = nxt.get((ci+x, cj+y), (F(0), F(0), F(0)))
                inc = (p, f*M+eta*reward*p,
                       f*f*Q+2*f*eta*reward*M+eta**2*reward**2*p)
                nxt[ci+x, cj+y] = tuple(o+prob*z for o, z in zip(old, inc))
        buckets = nxt
    for _ in range(max(0, H-m)):
        nxt = {}
        for key, (p, M, Q) in buckets.items():
            dhat = F(key[0]-key[1], m)
            f1 = f2 = F(0)
            for x, y, prob in pair_law(pi, pj):
                f = 1-eta*(1-gamma+2*gamma*((1-w)*x+w*(y+dhat)))
                f1 += prob*f
                f2 += prob*f*f
            nxt[key] = (p, f1*M+eta*reward*p,
                        f2*Q+2*eta*reward*f1*M+eta**2*reward**2*p)
        buckets = nxt
    target = reward/(1-gamma+2*gamma*pi)
    assert sum(z[0] for z in buckets.values()) == 1
    return sum(Q-2*target*M+target**2*p for p, M, Q in buckets.values())


def stationary_risk(pi, pj, gamma, eta, w, m, reward=F(1)):
    a = 1-gamma+2*gamma*pi
    s = 4*gamma**2*((1-w)**2*pi*(1-pi)+w**2*pj*(1-pj))
    risk = F(0)
    for ki, kj in product(range(m+1), repeat=2):
        prob = (comb(m, ki)*pi**ki*(1-pi)**(m-ki)
                *comb(m, kj)*pj**kj*(1-pj)**(m-kj))
        Delta = F(ki-kj, m)-(pi-pj)
        ac = a+2*gamma*w*Delta
        denom = 2*ac-eta*(ac*ac+s)
        assert ac > 0 and denom > 0
        risk += prob*((reward/ac-reward/a)**2+eta*reward**2*s/(ac*ac*denom))
    return risk


def audit():
    # A declared arithmetic witness, not a search over outcome-selected cells.
    pi, pj, gamma, eta, w = F(1, 4), F(3, 8), F(1, 2), F(1, 4), F(1, 100)
    m, D, T, header = 1, 2, 5, 8
    risk = [F(0)]*(T+1)
    for path in product(pair_law(pi, pj), repeat=T):
        learner = DelayedFlipPair(gamma, eta, w, m, D, env_budget=2*T,
                                 message_budget=2*T*(header+1), header_bits=header)
        prob = F(1)
        for _, _, p in path:
            prob *= p
        target = 1/(1-gamma+2*gamma*pi)
        risk[0] += prob*target**2
        for tick, (x, y, _) in enumerate(path, 1):
            val = learner.step(x, y)[0]
            risk[tick] += prob*(val-target)**2
        assert (learner.env_used, learner.message_used) == (2*T, 2*T*(header+1))
        assert learner.processed == T-D and len(learner.pending) == D
        try:
            learner.step(0, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("budget overspend accepted")
    for t in range(T+1):
        assert risk[t] == risk_by_moments(pi, pj, gamma, eta, w, m, D, t)
    # Warm-up value is correlated with calibration error; do not factor it.
    cov = F(0)
    for x, y, prob in pair_law(pi, pj):
        for x2, y2, prob2 in pair_law(pi, pj):
            u2 = (1-eta*(1-gamma+2*gamma*x2))*eta+eta
            Delta = F(x+x2-y-y2, 2)-(pi-pj)
            cov += prob*prob2*u2*Delta
    assert cov != 0
    # All possible counts for an equal-window calibration cancel identically.
    for ki, kj in product(range(4), repeat=2):
        ai, aj = F(ki, 3), F(kj, 3)
        assert (1-w)*ai+w*(aj+(ai-aj)) == ai
    # Exact derivative of stationary risk at w=0, independent of m and D.
    a = 1-gamma+2*gamma*pi
    s0 = 4*gamma**2*pi*(1-pi)
    derivative = -2*s0*eta*(2*a-eta*a*a)/(a*a*(2*a-eta*(a*a+s0))**2)
    assert derivative < 0
    local = stationary_risk(pi, pj, gamma, eta, F(0), m)
    corrected = stationary_risk(pi, pj, gamma, eta, w, m)
    return {
        "kind": "exact_identity_checks_not_efficacy",
        "enumerated_paths": 4**T, "risk_clock_identities": T+1,
        "all_equalities_exact_rational": True,
        "charged_actor_transitions": 2*T, "charged_message_bits": 2*T*(header+1),
        "processed_pairs": T-D, "pending_pairs_charged": D,
        "warmup_value_calibration_covariance": str(cov),
        "stationary_risk_derivative_at_zero": str(derivative),
        "local_constant_step_stationary_risk": str(local),
        "corrected_constant_step_stationary_risk": str(corrected),
        "stationary_difference": str(corrected-local),
        "strong_local_full_state_risk_bound": "L^2*p_i*(1-p_i)/t; L=2*gamma*c_i/(1-gamma)^2",
        "decision": "REJECT_AS_PAPER_CANDIDATE_NO_PILOT",
        "boundary": "Constant-step gain is not superiority to strong local learning; informative flips are iid despite correlated states. No independent novelty qualified.",
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
