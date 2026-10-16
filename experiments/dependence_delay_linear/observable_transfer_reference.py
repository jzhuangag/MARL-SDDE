"""Observable coupled-probe reference; exact qualification, not an efficacy run.

The estimator consumes transitions/returns, not P or V*. It requires fresh
conditional simulation/reset access. The qualification functions alone use a
known finite model to enumerate expectations. No random seed is consumed.
"""

from dataclasses import dataclass
import hashlib
import itertools
import json
import math

import numpy as np


def _positive_int(x, name):
    if isinstance(x, bool) or not isinstance(x, int) or x < 1:
        raise ValueError(name + " must be a positive integer")


@dataclass(frozen=True)
class ProbeSpec:
    value: np.ndarray
    directions: np.ndarray
    start_state: int
    discount: float
    step: float
    horizon: int
    return_length: int
    reward_bound: float

    def __post_init__(self):
        _positive_int(self.horizon, "horizon")
        _positive_int(self.return_length, "return_length")
        v = np.array(self.value, dtype=float, copy=True)
        d = np.array(self.directions, dtype=float, copy=True)
        if (v.ndim != 1 or not len(v) or d.ndim != 2
                or d.shape[0] != len(v) or d.shape[1] < 1
                or not np.isfinite(v).all() or not np.isfinite(d).all()
                or not 0 <= self.discount < 1 or not 0 < self.step <= 1
                or not np.isfinite(self.reward_bound) or self.reward_bound <= 0
                or isinstance(self.start_state, bool)
                or not isinstance(self.start_state, int)
                or not 0 <= self.start_state < len(v)):
            raise ValueError("invalid probe specification")
        vmax = self.reward_bound/(1-self.discount)
        if np.max(np.abs(v)) > vmax or np.max(np.abs(v[:, None]+d)) > vmax:
            raise ValueError("initial local and donor values must lie in bounded TD box")
        v.flags.writeable = False
        d.flags.writeable = False
        object.__setattr__(self, "value", v)
        object.__setattr__(self, "directions", d)

    @property
    def value_bound(self):
        return self.reward_bound/(1-self.discount)

    @property
    def direction_bound(self):
        return float(np.max(np.abs(self.directions)))

    @property
    def context_hash(self):
        payload = {"v": self.value.tolist(), "d": self.directions.tolist(),
                   "s": self.start_state, "gamma": self.discount, "eta": self.step,
                   "H": self.horizon, "L": self.return_length, "R": self.reward_bound}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _state(state, dimension):
    if isinstance(state, bool) or not isinstance(state, (int, np.integer)) or not 0 <= state < dimension:
        raise ValueError("invalid observed state")


def encode_probe(spec, training_transitions, evaluation_state, return_transitions):
    """One random-time sample, with K uniform on 0..H-1 and U uniform on states.

    Training starts from the frozen context. Return trajectory starts from U
    with independent randomness, not the training trajectory's reused suffix.
    These sampling-law requirements cannot be verified from numbers alone.
    Each transition is (state, reward, next_state), charged once even if its
    reward is deterministic. The last return transition is charged as well.
    """
    v, u = spec.value.copy(), spec.directions.copy()
    current, training_count = spec.start_state, 0
    for state, reward, nxt in training_transitions:
        _state(state, len(v))
        _state(nxt, len(v))
        if state != current or not np.isfinite(reward) or abs(reward) > spec.reward_bound:
            raise ValueError("invalid training observation")
        if training_count >= spec.horizon-1:
            raise ValueError("K must be smaller than H")
        # Compute both right-hand sides before overwriting a row (s may equal j).
        new_v = v[state]+spec.step*(reward+spec.discount*v[nxt]-v[state])
        new_u = u[state]+spec.step*(spec.discount*u[nxt]-u[state])
        v[state], u[state] = new_v, new_u
        current = nxt
        training_count += 1
    _state(evaluation_state, len(v))
    current, label, return_count = evaluation_state, 0., 0
    for state, reward, nxt in return_transitions:
        _state(state, len(v))
        _state(nxt, len(v))
        if (state != current or not np.isfinite(reward) or abs(reward) > spec.reward_bound
                or return_count >= spec.return_length):
            raise ValueError("invalid independent return observation")
        label += spec.discount**return_count*reward
        current = nxt
        return_count += 1
    if return_count != spec.return_length:
        raise ValueError("return trajectory has wrong length")
    row = u[evaluation_state]
    gram = spec.horizon*np.outer(row, row)
    linear = spec.horizon*row*(v[evaluation_state]-label)
    return {"G": gram, "g": linear, "training_transitions": training_count,
            "return_transitions": return_count, "reset_requests": 2,
            "context_hash": spec.context_hash}


class ProbeAccumulator:
    """Streaming fixed-context moments; ordered IDs prevent duplicate counting.

    Unique IDs do not prove independence. Fresh replicate sampling is an
    external collection contract, and overlapping old replay is not certified.
    """
    def __init__(self, spec):
        self.spec = spec
        k = spec.directions.shape[1]
        self.G = np.zeros((k, k))
        self.g = np.zeros(k)
        self.n = self.training = self.returns = self.resets = 0

    def add(self, sample_number, sample):
        if sample_number != self.n or isinstance(sample_number, bool):
            raise ValueError("replicate IDs must be ordered and unique")
        if sample["context_hash"] != self.spec.context_hash:
            raise ValueError("cannot reuse moments after changing proposal/context")
        self.G += sample["G"]
        self.g += sample["g"]
        self.n += 1
        self.training += sample["training_transitions"]
        self.returns += sample["return_transitions"]
        self.resets += sample["reset_requests"]

    def means(self):
        if self.n == 0:
            raise ValueError("no evidence")
        return self.G/self.n, self.g/self.n


def confidence_terms(spec, n, delta):
    """Fixed-n replicate Hoeffding union bound, not a within-chain iid bound."""
    _positive_int(n, "replicates")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    k = spec.directions.shape[1]
    entries = k*(k+1)//2+k
    scale = math.sqrt(2*math.log(2*entries/delta)/n)
    h, d, b = spec.horizon, spec.direction_bound, spec.value_bound
    return {"gram_radius": h*d*d*scale,
            "linear_radius": 2*h*d*b*scale,
            "return_bias_radius": h*d*b*spec.discount**spec.return_length,
            "delta": delta, "entries": entries}


def robust_qp(G, g, terms):
    """Coefficients of a convex uniform upper bound over beta>=0,sum beta<=1."""
    g = np.asarray(g, dtype=float)
    G = np.asarray(G, dtype=float)
    if G.shape != (len(g), len(g)) or not np.isfinite(G).all() or not np.isfinite(g).all():
        raise ValueError("invalid moments")
    if not np.allclose(G, G.T) or np.linalg.eigvalsh(G).min() < -1e-10:
        raise ValueError("Gram estimate must be positive semidefinite")
    penalties = [terms[k] for k in ("gram_radius", "linear_radius", "return_bias_radius")]
    if not np.isfinite(penalties).all() or min(penalties) < 0:
        raise ValueError("invalid confidence penalties")
    return (G+terms["gram_radius"]*np.ones_like(G),
            g+terms["linear_radius"]+terms["return_bias_radius"])


def scalar_controller(G, g, terms):
    """Exact one-donor robust QP; positive action only with negative upper cost."""
    matrix, linear = robust_qp(np.array([[G]]), np.array([g]), terms)
    a, b = float(matrix[0, 0]), float(linear[0])
    beta = float(np.clip(-b/a, 0., 1.)) if a > 0 else float(b < 0)
    upper = a*beta*beta+2*b*beta
    if upper >= 0:
        beta, upper = 0., 0.
    return {"beta": beta, "upper_advantage": upper}


def project_transfer_simplex(vector):
    """Euclidean projection onto nonnegative weights with sum at most one."""
    y = np.maximum(np.asarray(vector, dtype=float), 0.)
    if not np.isfinite(y).all():
        raise ValueError("invalid transfer weights")
    if y.sum() <= 1.:
        return y
    ordered = np.sort(y)[::-1]
    thresholds = (np.cumsum(ordered)-1)/np.arange(1, len(y)+1)
    last = np.flatnonzero(ordered > thresholds)[-1]
    projected = np.maximum(y-thresholds[last], 0.)
    return projected/max(1., float(projected.sum()))


def qp_controller(G, g, terms, iterations=100):
    """Bounded-work projected-gradient QP with explicit optimization gap.

    Safety uses the final upper value, not assumed solver optimality. The
    real-arithmetic suboptimality bound is L/(2*iterations), starting at zero.
    Solving a k-dimensional QP does not remove the estimator's horizon cost.
    """
    _positive_int(iterations, "QP iterations")
    matrix, linear = robust_qp(G, g, terms)
    matrix = (matrix+matrix.T)/2
    lipschitz = max(0., 2*float(np.linalg.eigvalsh(matrix)[-1]))
    beta = np.zeros(len(linear))
    if lipschitz == 0:
        if linear.min() < 0:
            beta[np.argmin(linear)] = 1.
    else:
        for _ in range(iterations):
            beta = project_transfer_simplex(beta-(2*matrix@beta+2*linear)/lipschitz)
    upper = float(beta@matrix@beta+2*linear@beta)
    if upper >= 0:
        beta, upper = np.zeros_like(beta), 0.
    return {"beta": beta, "upper_advantage": upper,
            "optimization_gap_bound": lipschitz/(2*iterations)}


def cost_contract(spec, n):
    _positive_int(n, "replicates")
    d, k = spec.directions.shape
    return {"expected_probe_transitions": n*((spec.horizon-1)/2+spec.return_length),
            "worst_probe_transitions": n*(spec.horizon-1+spec.return_length),
            "reset_requests": 2*n,
            "expected_forward_TD_steps": n*(spec.horizon-1)/2,
            "initial_donor_scalars": d*k,
            "core_mutable_numeric_scalars": d*(k+1)+k*k+k,
            "time_order_tabular": "O(n*(d*k + H*k + L + k^2)); excludes external reset/communication latency",
            "time_order_dense_features": "O(n*(H*d*k + L*d + k^2)); not horizon-independent",
            "QP_time_order": "O(k^3 + I*(k^2 + k*log(k))) for I projected-gradient iterations",
            "data_contract": "fresh independent conditional simulator/reset replicates; not a single uncontrolled continuing trajectory"}


def enumerate_paths(p, rewards, start, transitions):
    """Qualification oracle only. Estimator above never receives p/rewards table."""
    for tail in itertools.product(range(len(p)), repeat=transitions):
        current, probability, records = start, 1., []
        for nxt in tail:
            probability *= p[current, nxt]
            records.append((current, float(rewards[current]), nxt))
            current = nxt
        if probability:
            yield probability, records


def exact_probe_mean(spec, p, rewards):
    k = spec.directions.shape[1]
    G, g, total = np.zeros((k, k)), np.zeros(k), 0.
    for time in range(spec.horizon):
        for pr_train, training in enumerate_paths(p, rewards, spec.start_state, time):
            for state in range(len(p)):
                for pr_return, returns in enumerate_paths(p, rewards, state, spec.return_length):
                    weight = pr_train*pr_return/(spec.horizon*len(p))
                    sample = encode_probe(spec, training, state, returns)
                    G += weight*sample["G"]
                    g += weight*sample["g"]
                    total += weight
    return G, g, total


def exact_cost_contrast(spec, p, rewards, beta, target):
    """Actual coupled local training cost, evaluated with a supplied oracle target."""
    total = 0.
    for probability, records in enumerate_paths(p, rewards, spec.start_state, spec.horizon-1):
        local = spec.value.copy()
        changed = local+spec.directions@np.asarray(beta)
        cost = float(np.mean((changed-target)**2-(local-target)**2))
        for s, reward, j in records:
            local[s] += spec.step*(reward+spec.discount*local[j]-local[s])
            changed[s] += spec.step*(reward+spec.discount*changed[j]-changed[s])
            cost += float(np.mean((changed-target)**2-(local-target)**2))
        total += probability*cost
    return total


def projection_leakage_witness():
    from rl_collaboration_interface_audit import markov_td_risk_gramian
    p = np.array([[.9, .1], [.4, .6]])
    eta, gamma = .5, .9
    # One-dimensional donor space span(e_0). Start environmental state is 1.
    true = float(markov_td_risk_gramian(p, gamma, eta, 2)[1, 0, 0])
    return {"true_projected_two_step_risk": true,
            "naive_closed_projected_recursion": 2.,
            "missed_leakage": true-2.,
            "exact_missed_leakage": p[1, 0]*(eta*gamma)**2,
            "scope": "counterexample to projection-only closure, not to all matrix-free estimators"}


def qualification_report():
    from rl_collaboration_interface_audit import return_moments, value_oracle
    p, rewards = np.array([[.8, .2], [.3, .7]]), np.array([0., .5])
    spec = ProbeSpec(np.array([.2, -.1]), np.array([[.4, -.3], [-.2, .5]]),
                     0, .6, .4, 3, 4, .5)
    G, g, probability = exact_probe_mean(spec, p, rewards)
    target_L, _ = return_moments(p, rewards, spec.discount, spec.return_length)
    target = value_oracle(p, rewards, spec.discount)
    beta = np.array([.3, .4])
    estimated = float(beta@G@beta+2*g@beta)
    true_L = exact_cost_contrast(spec, p, rewards, beta, target_L)
    true = exact_cost_contrast(spec, p, rewards, beta, target)
    h64 = ProbeSpec(np.array([1.]), np.array([[-.5]]), 0, .9, .5, 64, 32, 1.)
    return {"kind": "exact_observable_estimator_qualification_not_efficacy",
            "probability_mass": probability, "mean_G": G.tolist(), "mean_g": g.tolist(),
            "estimated_finite_return_contrast": estimated,
            "exact_finite_return_contrast": true_L, "exact_infinite_value_contrast": true,
            "identity_residual": estimated-true_L,
            "target_bias_absolute": abs(true-estimated),
            "target_bias_bound": 2*confidence_terms(spec, 1, .01)["return_bias_radius"]*sum(beta),
            "projection_closure": projection_leakage_witness(),
            "H64_n128_cost": cost_contract(h64, 128),
            "H64_n128_confidence": confidence_terms(h64, 128, .01),
            "decision": {"observable_reference_contract": "qualified_with_fresh_reset_access",
                         "projected_only_recursion": "rejected_without_invariance_or_leakage_control",
                         "cheap_long_horizon_final_candidate": "reject_this_direct_probe_implementation",
                         "general_impossibility_claim": False,
                         "new_efficacy_pilot_authorized": False, "formal_authorized": False,
                         "gpu_authorized": False}}


if __name__ == "__main__":
    print(json.dumps(qualification_report(), indent=2, sort_keys=True, allow_nan=False))
