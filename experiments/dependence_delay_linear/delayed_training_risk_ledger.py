"""A scoped reference, NOT a new pilot or full-state TD safety theorem.

Certifies cumulative PRE-UPDATE VISITED-STATE prediction risk against the
independently updated shadow. Uses every raw transition once, with overlapping
finite returns and fixed delivery lag. The note states the conditional-law
assumption, martingale proof, and why the result does not solve full-state MSE.
"""

from dataclasses import dataclass
import json
import math

import numpy as np


@dataclass(frozen=True)
class LedgerSpec:
    dimension: int
    reward_bound: float
    discount: float
    return_length: int
    delivery_delay: int
    horizon: int
    delta: float
    initial_allowance: float
    allowance_per_step: float
    law_tag: str

    def __post_init__(self):
        for name in ("dimension", "return_length", "horizon"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.delivery_delay, bool) or not isinstance(self.delivery_delay, int)
                or self.delivery_delay < 0 or not math.isfinite(self.reward_bound)
                or self.reward_bound <= 0 or not 0 <= self.discount < 1
                or not 0 < self.delta < 1 or not self.law_tag):
            raise ValueError("invalid fixed ledger specification")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.initial_allowance, self.allowance_per_step)):
            raise ValueError("allowances must be explicit finite nonnegative numbers")

    @property
    def value_bound(self):
        return self.reward_bound/(1-self.discount)

    @property
    def lag(self):
        return self.return_length+self.delivery_delay


def worst_excess(actual, shadow, bound):
    """sup_{|V|<=B} [(actual-V)^2-(shadow-V)^2], scalar only."""
    if (not all(math.isfinite(v) for v in (actual, shadow, bound)) or bound <= 0
            or abs(actual) > bound+1e-12 or abs(shadow) > bound+1e-12):
        raise ValueError("invalid bounded scalar prediction")
    d = actual-shadow
    return max(0., d*(actual+shadow)+2*bound*abs(d))


def safe_scale(candidate, shadow, bound, available):
    """Real-arithmetic maximal scale; act guards its rounded realized cost."""
    worst_excess(candidate, shadow, bound)
    if not math.isfinite(available) or available < 0:
        raise ValueError("negative/nonfinite available allowance")
    d = candidate-shadow
    a, b = d*d, max(0., 2*shadow*d+2*bound*abs(d))
    if a+b <= available:
        return 1.
    if available == 0:
        return 0.
    return min(1., 2*available/(b+math.sqrt(b*b+4*a*available)))


def score(actual, shadow, finite_return):
    return (actual-shadow)*(actual+shadow-2*finite_return)


def colored_radius(counts, spec):
    n = np.asarray(counts)
    if (n.shape != (spec.lag,) or not np.isfinite(n).all() or np.any(n < 0)
            or np.any(n != np.floor(n)) or n.sum() > spec.horizon):
        raise ValueError("invalid chronological color-prefix counts")
    # Conditional Hoeffding lemma + union over at most lag*horizon prefixes.
    return float(4*spec.value_bound**2
                 *np.sqrt(2*n*math.log(spec.lag*spec.horizon/spec.delta)).sum())


class DelayedRiskLedger:
    """One local streaming ledger; caller executes returned parameters.

    Proposal/own-shadow arrays must be measurable BEFORE the next reward.
    This class never accepts oracle risk, P, V*, externally selected labels or
    cancellation requests. Calling observe does not itself update parameters;
    the same record must also drive both actual and shadow TD updates.
    """
    def __init__(self, spec):
        self.spec = spec
        self.time = 0
        self.upper = 0.
        self.pending_upper = 0.
        self.settled_score_bias = 0.
        self.counts = np.zeros(spec.lag, dtype=int)
        self.pending = []
        self.awaiting_observation = False
        self.next_state = None
        self.actor_transitions = 0
        self.processed_labels = 0
        self.return_reward_reuses = 0

    def act(self, candidate, own_shadow, state, *, law_tag):
        spec, B = self.spec, self.spec.value_bound
        if (self.awaiting_observation or self.time >= spec.horizon or law_tag != spec.law_tag
                or isinstance(state, bool) or not isinstance(state, int)
                or not 0 <= state < spec.dimension
                or (self.next_state is not None and state != self.next_state)):
            raise ValueError("wrong time, state or registered law")
        c, l = np.array(candidate, float, copy=True), np.array(own_shadow, float, copy=True)
        if (c.shape != (spec.dimension,) or l.shape != c.shape
                or not np.isfinite(c).all() or not np.isfinite(l).all()
                or np.max(np.abs(c)) > B or np.max(np.abs(l)) > B):
            raise ValueError("invalid candidate/independently updated shadow")
        allowance = spec.initial_allowance+spec.allowance_per_step*(self.time+1)
        available = max(0., allowance-self.upper)
        alpha = safe_scale(float(c[state]), float(l[state]), B, available)
        actual = l+alpha*(c-l)
        reservation = worst_excess(float(actual[state]), float(l[state]), B)
        # Near-identical predictions can quantize a whole interval of alpha
        # onto one parameter value. Walking ulps could take ~10^15 iterations.
        # Use bounded conservative backoff, then exact shadow restoration.
        if reservation > available or self.upper+reservation > allowance:
            alpha *= .5
            actual = l+alpha*(c-l)
            reservation = worst_excess(float(actual[state]), float(l[state]), B)
        if reservation > available or self.upper+reservation > allowance:
            alpha, actual, reservation = 0., l.copy(), 0.
        self.upper += reservation
        if actual[state] != l[state]:
            self.pending.append({"birth": self.time, "age": 0, "return": 0.,
                                 "actual": float(actual[state]), "shadow": float(l[state]),
                                 "reservation": reservation})
            self.pending_upper += reservation
        self.awaiting_observation = True
        self._current_state = state
        actual.flags.writeable = False
        return {"parameters": actual, "alpha": alpha, "reservation": reservation,
                "upper": self.upper, "allowance": allowance}

    def observe(self, state, reward, next_state, *, law_tag):
        spec = self.spec
        if (not self.awaiting_observation or law_tag != spec.law_tag
                or state != self._current_state or not math.isfinite(reward)
                or abs(reward) > spec.reward_bound
                or isinstance(next_state, bool) or not isinstance(next_state, int)
                or not 0 <= next_state < spec.dimension):
            raise ValueError("invalid or duplicate raw transition")
        remaining = []
        matured = []
        for item in self.pending:
            if item["age"] < spec.return_length:
                item["return"] += spec.discount**item["age"]*reward
                self.return_reward_reuses += 1
            item["age"] += 1
            if item["age"] == spec.lag:
                bias = 2*abs(item["actual"]-item["shadow"])*spec.value_bound*spec.discount**spec.return_length
                self.settled_score_bias += score(item["actual"], item["shadow"], item["return"])+bias
                self.pending_upper -= item["reservation"]
                self.counts[item["birth"] % spec.lag] += 1
                self.processed_labels += 1
                matured.append(dict(item))
            else:
                remaining.append(item)
        self.pending = remaining
        # Numerical cancellation can produce a tiny negative pending sum.
        self.pending_upper = sum(item["reservation"] for item in self.pending)
        refreshed = self.settled_score_bias+colored_radius(self.counts, spec)+self.pending_upper
        # Crucial: maturity may INCREASE the noisy bound; it must not revoke an
        # already valid bound on past risk and consume unreserved allowance.
        self.upper = min(self.upper, refreshed)
        self.time += 1
        self.actor_transitions += 1
        self.next_state = next_state
        self.awaiting_observation = False
        return matured

    def final_status(self):
        if self.awaiting_observation:
            raise ValueError("last executed action still needs its transition")
        return {"time": self.time, "upper": self.upper,
                "allowance": self.spec.initial_allowance+self.spec.allowance_per_step*self.time,
                "pending_labels": len(self.pending), "pending_upper": self.pending_upper,
                "processed_labels": self.processed_labels,
                "actor_transitions": self.actor_transitions,
                "extra_environment_transitions": 0,
                "return_reward_reuses": self.return_reward_reuses,
                "risk_scope": "pre-update visited-state prediction risk only"}


def qualification_report():
    B = 1.
    alpha = safe_scale(.5, 0., B, .2)
    # State zero is all that is observed; the hidden state's true value can
    # be either +1 or -1. Both laws produce exactly the same observed path.
    c, l = np.array([0., 1.]), np.zeros(2)
    full = [float(np.mean((c-v)**2-(l-v)**2))
            for v in (np.array([0., 1.]), np.array([0., -1.]))]
    # Cumulative credit does not bound the reflected drawdown/debt queue.
    q, reflected = 0., []
    for g in (-1., 1.):
        q = max(0., q+g)
        reflected.append(q)
    return {"kind": "deterministic_contract_qualification_not_efficacy",
            "zero_allowance_nonzero_direction_scale": safe_scale(.5, 0., B, 0.),
            "positive_allowance_example": {"available": .2, "alpha": alpha,
                                           "reservation": worst_excess(alpha*.5, 0., B)},
            "unvisited_state_counterexample": {"visited_risk_in_both_laws": 0.,
                                                "full_state_contrasts": full},
            "cumulative_vs_reflected_counterexample": {"prefixes": [-1., 0.], "reflected_queue": reflected},
            "decision": {"visited_risk_interface": "qualified_under_stated_fixed_law_and_timing",
                         "full_state_MSE_interface": "not_qualified",
                         "independent_novelty_established": False,
                         "general_Lyapunov_convergence_proved": False,
                         "efficacy_or_regret_gain_proved": False,
                         "pilot_authorized": False, "formal_authorized": False}}


if __name__ == "__main__":
    print(json.dumps(qualification_report(), indent=2, sort_keys=True, allow_nan=False))
