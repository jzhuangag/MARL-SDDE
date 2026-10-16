"""Causal gradient transport and prospective variance/resource control.

The bridge is a mixture of COMPLETE joint-policy trajectory distributions,
not a product of independently mixed agent/time-step policies. Importance
weights are coefficients, detached before multiplying policy-gradient samples.
An anchor cohort is consumed once; adaptive reuse would invalidate the stated
conditional-unbiasedness interface.
"""
from dataclasses import dataclass
import hashlib
import numpy as np


def bridge_weights(current_logp, source_logp, source_fraction=.5):
    current_logp, source_logp = np.broadcast_arrays(
        np.asarray(current_logp, float), np.asarray(source_logp, float))
    if not 0 < source_fraction < 1:
        raise ValueError("A defensive bridge requires two positive mixture masses")
    if not np.isfinite(current_logp).all() or not np.isfinite(source_logp).all():
        raise FloatingPointError("Invalid trajectory log probabilities")
    denominator = np.logaddexp(np.log1p(-source_fraction) + current_logp,
                              np.log(source_fraction) + source_logp)
    target = np.exp(current_logp - denominator)
    source = np.exp(source_logp - denominator)
    assert np.all(target <= 1 / (1-source_fraction) + 1e-12)
    assert np.all(source <= 1 / source_fraction + 1e-12)
    return target, source


def trace_variance(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite independent episode-gradient vectors required")
    return float(np.square(x - x.mean(axis=0)).sum() / (len(x)-1))


def risk_feedback(target, control, anchor, options):
    """All-option UNBIASED variance feedback, not a confidence upper bound.

target/control are paired, weighted episode gradients from the current bridge.
anchor contains independent source-policy episode gradients. The options'
sample counts concern the SAME policies/distributions, not other trajectories.
"""
    target, control, anchor = map(lambda x: np.asarray(x, float), (target, control, anchor))
    if target.shape != control.shape or target.shape[1] != anchor.shape[1]:
        raise ValueError("Gradient geometry mismatch")
    avar = trace_variance(anchor)
    return np.asarray([trace_variance(target-o.beta*control)/o.episodes
                       + o.beta**2*avar/len(anchor) for o in options])


def transported_direction(target, control, anchor, beta):
    if not 0 <= beta <= 1:
        raise ValueError("Frozen beta catalogue is restricted to [0,1]")
    target, control, anchor = map(lambda x: np.asarray(x, float), (target, control, anchor))
    if target.shape != control.shape or target.shape[1] != anchor.shape[1]:
        raise ValueError("Gradient geometry mismatch")
    result = target.mean(axis=0) + beta*(anchor.mean(axis=0)-control.mean(axis=0))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite transported direction")
    return result


def project_simplex(x):
    x = np.asarray(x, float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Invalid simplex vector")
    ordered = np.sort(x)[::-1]
    offsets = (np.cumsum(ordered)-1)/np.arange(1, len(x)+1)
    rho = np.flatnonzero(ordered > offsets)[-1]
    return np.maximum(x-offsets[rho], 0.)


@dataclass(frozen=True)
class Option:
    episodes: int
    beta: float


class RiskQueueController:
    """Full-information projected OGD on a variance/resource drift surrogate.

No return, current anchor vector, current bridge or selected-score maximum
enters begin(). Realized risk feedback is used ONLY for later decisions.
Losses need finite second moments for the finite-time expected OGD inequality;
they are not silently clipped to manufacture a bounded-loss theorem.
"""
    def __init__(self, options, *, step, variance_weight, targets):
        self.options = tuple(options)
        assert self.options and all(o.episodes >= 2 and 0 <= o.beta <= 1 for o in options)
        if step <= 0 or variance_weight < 0:
            raise ValueError("Invalid controller constants")
        self.step, self.variance_weight = float(step), float(variance_weight)
        self.targets = np.asarray(targets, float)
        if self.targets.ndim != 1 or np.any(self.targets <= 0):
            raise ValueError("Invalid resource targets")
        self.p = np.full(len(options), 1/len(options))
        self.q = np.zeros_like(self.targets)
        self.pending = None

    def begin(self, costs, rng, *, source_id, current_hash):
        if self.pending is not None:
            raise RuntimeError("Unclosed control transaction")
        costs = np.asarray(costs, float)
        if costs.shape != (len(self.options), len(self.targets)) or np.any(costs < 0) or not np.isfinite(costs).all():
            raise ValueError("Every option needs prospective, fully charged resource cost")
        index = int(rng.choice(len(self.options), p=self.p))
        self.pending = dict(index=index, costs=costs.copy(), queues=self.q.copy(),
                            probabilities=self.p.copy(), source_id=source_id, current_hash=current_hash)
        return dict(index=index, episodes=self.options[index].episodes,
                    beta=self.options[index].beta, source_id=source_id,
                    current_hash=current_hash, before_anchor_and_bridge=True)

    def close(self, risks, actual_cost):
        if self.pending is None:
            raise RuntimeError("No prospective decision")
        risks, actual_cost = np.asarray(risks, float), np.asarray(actual_cost, float)
        if risks.shape != self.p.shape or not np.isfinite(risks).all() or np.any(risks < 0):
            raise ValueError("Invalid full-information variance feedback")
        s = self.pending
        if not np.array_equal(actual_cost, s['costs'][s['index']]):
            raise AssertionError("Prospective charge does not equal actual ledger")
        losses = self.variance_weight*risks + s['costs'] @ s['queues']
        self.p = project_simplex(s['probabilities'] - self.step*losses)
        self.q = np.maximum(0., s['queues']+actual_cost-self.targets)
        self.pending = None
        return dict(index=s['index'], source_id=s['source_id'], current_hash=s['current_hash'],
                    probabilities_before=s['probabilities'].tolist(), probabilities_after=self.p.tolist(),
                    queues_before=s['queues'].tolist(), queues_after=self.q.tolist(),
                    risk_feedback=risks.tolist(), losses=losses.tolist(), actual_cost=actual_cost.tolist(),
                    variance_upper_certificate=False, current_return_used=False)


class SealedAnchor:
    """Only policy/baseline/shape/service metadata may be opened before begin.

This is an audit contract, not an OS security boundary. A one-use transaction
guards accidental adaptive reuse; callers must also hash immutable payloads.
"""
    def __init__(self, identifier, gradients):
        self.identifier = str(identifier)
        self.__gradients = np.array(gradients, dtype=float, copy=True)
        if self.__gradients.ndim != 2 or len(self.__gradients) < 2 or not np.isfinite(self.__gradients).all():
            raise ValueError("Invalid anchor payload")
        self.__gradients.setflags(write=False)
        self.sha256 = hashlib.sha256(self.__gradients.tobytes()).hexdigest()
        self.consumed = False

    def open_once(self, decision):
        if self.consumed:
            raise RuntimeError("Adaptive reuse of an anchor cohort is prohibited")
        if not decision.get('before_anchor_and_bridge') or decision.get('source_id') != self.identifier:
            raise RuntimeError("Receipt choice must precede the current anchor and bridge")
        self.consumed = True
        return self.__gradients
