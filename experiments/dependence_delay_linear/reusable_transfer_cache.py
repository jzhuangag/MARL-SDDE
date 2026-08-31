"""Fixed-law, all-prefix directional cache: qualification, not efficacy.

No original/frozen controller is modified. Online inputs are observations and
parameter vectors. Functions explicitly named exact/qualification may use P.
Fresh independent conditional reset replicas are an external data contract.
"""

from dataclasses import replace
import json
import math

import numpy as np

from observable_transfer_reference import (
    ProbeSpec, _positive_int, _state, encode_probe, enumerate_paths,
    exact_cost_contrast, qp_controller,
)


def cache_radii(anchor, states, n, delta, horizon):
    _positive_int(n, "replicates per state")
    _positive_int(horizon, "query horizon")
    if not 0 < delta < 1 or not states or horizon > anchor.horizon:
        raise ValueError("invalid simultaneous coverage contract")
    r = anchor.directions.shape[1]
    entries = len(states)*anchor.horizon*(r*(r+1)//2+r)
    scale = math.sqrt(2*math.log(2*entries/delta)/n)
    H, E, B = anchor.horizon, anchor.direction_bound, anchor.value_bound
    return {"gram_radius": H*E*E*scale,
            "linear_radius": 2*H*E*B*scale,
            "return_bias_radius": horizon*E*B*anchor.discount**anchor.return_length,
            "entries": entries, "delta": delta}


def query_upper(anchor, G, g, terms, horizon, value, directions, coordinates, mapping):
    """Certified reuse at v=v0+E c+e, D=E A+F, including e/F residuals.

    Coordinates need not be orthogonal projections. Their actual residuals
    are calculated; no subspace invariance or low-rank TD dynamics is assumed.
    Inputs may depend on the completed cache because coverage is uniform.
    """
    _positive_int(horizon, "query horizon")
    if horizon > anchor.horizon:
        raise ValueError("query horizon exceeds cache")
    v, D, c, A = [np.asarray(x, dtype=float) for x in
                   (value, directions, coordinates, mapping)]
    E, B = anchor.directions, anchor.value_bound
    r = E.shape[1]
    if (v.shape != anchor.value.shape or D.ndim != 2 or D.shape[0] != len(v)
            or D.shape[1] < 1 or c.shape != (r,) or A.shape != (r, D.shape[1])
            or not all(np.isfinite(x).all() for x in (v, D, c, A))
            or np.max(np.abs(v)) > B or np.max(np.abs(v[:, None]+D)) > B):
        raise ValueError("invalid bounded query or coordinates")
    G, g = np.asarray(G, dtype=float), np.asarray(g, dtype=float)
    if (G.shape != (r, r) or g.shape != (r,) or not np.isfinite(G).all()
            or not np.isfinite(g).all() or not np.allclose(G, G.T)
            or np.linalg.eigvalsh(G).min() < -1e-10):
        raise ValueError("invalid cached moments")
    rg, rl, rb = [terms[key] for key in
                   ("gram_radius", "linear_radius", "return_bias_radius")]
    if not np.isfinite([rg, rl, rb]).all() or min(rg, rl, rb) < 0:
        raise ValueError("invalid radii")
    projected = E@A
    residual_v = float(np.max(np.abs(v-anchor.value-E@c)))
    residual_D = np.max(np.abs(D-projected), axis=0)
    coordinate_l1 = np.sum(np.abs(A), axis=0)
    projected_bound = np.max(np.abs(projected), axis=0)
    max_f, max_b = float(residual_D.max()), float(projected_bound.max())
    residual_quadratic = horizon*(2*max_b*max_f+max_f*max_f)
    matrix = (A.T@G@A+rg*np.outer(coordinate_l1, coordinate_l1)
              +residual_quadratic*np.ones((D.shape[1], D.shape[1])))
    linear = (A.T@(g+G@c)+(rg*np.sum(np.abs(c))+rl+rb)*coordinate_l1
              +horizon*(residual_v*projected_bound+2*B*residual_D))
    return {"matrix": (matrix+matrix.T)/2, "linear": linear,
            "value_residual": residual_v, "direction_residuals": residual_D,
            "coordinate_l1": coordinate_l1, "projected_direction_bounds": projected_bound,
            "residual_quadratic": residual_quadratic}


def cached_controller(problem, iterations=100):
    zero = {"gram_radius": 0., "linear_radius": 0., "return_bias_radius": 0.}
    return qp_controller(problem["matrix"], problem["linear"], zero, iterations)


def add_short_unroll_tail(problem, directions, value_bound, observed_horizon, full_horizon):
    """Matched short-unroll reference: do not silently equate short/full risk.

    Uses only nonexpansivity and |v-V|<=2B, so the omitted tail can be large.
    A tighter tail needs an independently justified contraction statement.
    """
    _positive_int(observed_horizon, "observed horizon")
    _positive_int(full_horizon, "full horizon")
    D = np.asarray(directions, dtype=float)
    if (full_horizon < observed_horizon or D.ndim != 2 or not D.size
            or not np.isfinite(D).all() or not np.isfinite(value_bound) or value_bound <= 0):
        raise ValueError("invalid omitted-tail contract")
    delta = np.max(np.abs(D), axis=0)
    tail = full_horizon-observed_horizon
    return {"matrix": problem["matrix"]+tail*np.outer(delta, delta),
            "linear": problem["linear"]+2*tail*value_bound*delta}


class PrefixCache:
    """One fixed anchor/law, n fresh replicas per registered starting state.

    K~Uniform(0..H-1); store H-weighted coefficients in bin K. Prefix estimates
    divide by ALL n replicas, never by the number landing in that prefix.
    Collection ends before training; no adaptive refresh/optional stopping.
    """
    def __init__(self, anchor, states, n, delta, law_tag):
        _positive_int(n, "replicates per state")
        states = tuple(states)
        if (not states or len(set(states)) != len(states) or not isinstance(law_tag, str)
                or not law_tag or not 0 < delta < 1):
            raise ValueError("invalid cache contract")
        for state in states:
            _state(state, len(anchor.value))
        self.anchor, self.states, self.n = anchor, states, n
        self.delta, self.law_tag = delta, law_tag
        r, H = anchor.directions.shape[1], anchor.horizon
        self.bins_G = {s: np.zeros((H, r, r)) for s in states}
        self.bins_g = {s: np.zeros((H, r)) for s in states}
        self.counts = {s: 0 for s in states}
        self.training = self.returns = self.resets = 0
        self.ready = False

    def add(self, state, index, training, evaluation_state, returns, law_tag):
        if (self.ready or law_tag != self.law_tag or state not in self.counts
                or isinstance(index, bool) or not isinstance(index, int)
                or index != self.counts[state] or index >= self.n):
            raise ValueError("wrong law, state, duplicate ID, or completed batch")
        sample = encode_probe(replace(self.anchor, start_state=state),
                              training, evaluation_state, returns)
        K = sample["training_transitions"]
        self.bins_G[state][K] += sample["G"]
        self.bins_g[state][K] += sample["g"]
        self.counts[state] += 1
        self.training += K
        self.returns += sample["return_transitions"]
        self.resets += 2

    def finalize(self):
        if self.ready or any(count != self.n for count in self.counts.values()):
            raise ValueError("requires exactly n replicas at every registered state")
        for state in self.states:
            self.bins_G[state][:] = np.cumsum(self.bins_G[state], axis=0)/self.n
            self.bins_g[state][:] = np.cumsum(self.bins_g[state], axis=0)/self.n
            self.bins_G[state].flags.writeable = False
            self.bins_g[state].flags.writeable = False
        self.ready = True

    def query(self, state, horizon, value, directions, coordinates, mapping,
              *, law_tag, discount, step):
        if (not self.ready or law_tag != self.law_tag or state not in self.states
                or discount != self.anchor.discount or step != self.anchor.step):
            raise ValueError("cache not ready or unregistered law/state/update rule")
        _positive_int(horizon, "query horizon")
        if horizon > self.anchor.horizon:
            raise ValueError("query horizon exceeds cache")
        terms = cache_radii(self.anchor, self.states, self.n, self.delta, horizon)
        return query_upper(self.anchor, self.bins_G[state][horizon-1],
                           self.bins_g[state][horizon-1], terms, horizon,
                           value, directions, coordinates, mapping)


def reuse_cost(anchor, states, n, queries):
    """Counts, not wall time. Queries list all remaining horizons in one stream."""
    _positive_int(n, "replicates")
    queries = tuple(queries)
    if not queries:
        raise ValueError("at least one query required")
    for h in queries:
        _positive_int(h, "query horizon")
        if h > anchor.horizon:
            raise ValueError("query horizon exceeds cache")
    d, r = anchor.directions.shape
    S, H, L = len(states), anchor.horizon, anchor.return_length
    if not S:
        raise ValueError("no registered states")
    cached = S*n*((H-1)/2+L)
    direct = n*sum((h-1)/2+L for h in queries)
    return {"cached_expected_transitions": cached,
            "cached_maximum_transitions": S*n*(H-1+L),
            "direct_expected_transitions_same_n": direct,
            "direct_to_cache_transition_ratio": direct/cached,
            "cached_resets": 2*S*n, "direct_resets": 2*len(queries)*n,
            "cached_expected_forward_steps": S*n*(H-1)/2,
            "prefix_coefficient_scalars": S*H*(r*r+r),
            "anchor_scalars": d*(r+1),
            "expected_probe_transitions_per_query": cached/len(queries),
            "cache_build_time": "O(S*n*(d*r+H*r+L+r^2)+S*H*r^2)",
            "query_time_excluding_QP": "O(d*r*(k+1)+r^2*k+r*k^2)",
            "QP_time": "O(k^3+I*(k^2+k*log(k)))",
            "donor_parameter_scalars_per_query": "d*k; alignment/coordinate-fit costs additional",
            "comparison_scope": "equal fixed n only, not equal confidence widths or equal accuracy",
            "same_total_budget_no_harm": False}


def exact_prefix_means(anchor, p, rewards, state):
    """Oracle enumeration of the implemented random-K bin estimator."""
    r, H = anchor.directions.shape[1], anchor.horizon
    bins_G, bins_g = np.zeros((H, r, r)), np.zeros((H, r))
    mass = 0.
    spec = replace(anchor, start_state=state)
    for K in range(H):
        for pt, training in enumerate_paths(p, rewards, state, K):
            for u in range(len(p)):
                for py, returns in enumerate_paths(p, rewards, u, anchor.return_length):
                    weight = pt*py/(H*len(p))
                    sample = encode_probe(spec, training, u, returns)
                    bins_G[K] += weight*sample["G"]
                    bins_g[K] += weight*sample["g"]
                    mass += weight
    return np.cumsum(bins_G, axis=0), np.cumsum(bins_g, axis=0), mass


def qualification_report():
    from rl_collaboration_interface_audit import return_moments, value_oracle
    p, rewards = np.array([[.8, .2], [.3, .7]]), np.array([0., .5])
    anchor = ProbeSpec(np.array([.2, -.1]), np.array([[.4, -.3], [-.2, .5]]),
                       0, .6, .4, 3, 3, .5)
    c, A, beta = np.array([.1, -.2]), np.array([[.7, -.1], [.1, .4]]), np.array([.3, .4])
    v, D = anchor.value+anchor.directions@c, anchor.directions@A
    target_L, _ = return_moments(p, rewards, .6, 3)
    identity_errors = []
    for s in (0, 1):
        gs, ls, _ = exact_prefix_means(anchor, p, rewards, s)
        for h in (1, 2, 3):
            predicted = beta@(A.T@gs[h-1]@A)@beta+2*beta@A.T@(ls[h-1]+gs[h-1]@c)
            true = exact_cost_contrast(ProbeSpec(v, D, s, .6, .4, h, 3, .5), p, rewards, beta, target_L)
            identity_errors.append(abs(float(predicted)-true))
    # Off-anchor query: correctness uses explicit residual, not projection closure.
    v_off, D_off = v+np.array([.03, -.02]), D+np.array([[.02, -.01], [-.04, .03]])
    Gs, gs, _ = exact_prefix_means(anchor, p, rewards, 0)
    terms = {"gram_radius": 0., "linear_radius": 0., "return_bias_radius": 0.}
    problem = query_upper(anchor, Gs[-1], gs[-1], terms, 3, v_off, D_off, c, A)
    upper = float(beta@problem["matrix"]@beta+2*problem["linear"]@beta)
    true = exact_cost_contrast(ProbeSpec(v_off, D_off, 0, .6, .4, 3, 3, .5), p, rewards, beta, target_L)
    # Analytic non-vacuity only: exact means plus hypothetical fixed-n radii.
    # NO 4096-replica collection or scientific seed has occurred.
    positive = ProbeSpec(np.array([1., -1.]), np.array([[-1.], [1.]]), 0, .2, .5, 2, 6, 1.)
    pos_G, pos_g, _ = exact_prefix_means(positive, p, np.zeros(2), 0)
    radii = cache_radii(positive, (0, 1), 4096, .01, 2)
    prob = query_upper(positive, pos_G[-1], pos_g[-1], radii, 2,
                       positive.value, positive.directions, [0.], [[1.]])
    result = cached_controller(prob)
    target = value_oracle(p, np.zeros(2), .2)
    adv = exact_cost_contrast(positive, p, np.zeros(2), result["beta"], target)
    cost_anchor = replace(anchor, horizon=64, return_length=32)
    long_cost = reuse_cost(cost_anchor, (0, 1), 128, range(64, 0, -1))
    short_cost = reuse_cost(replace(cost_anchor, horizon=8), (0, 1), 128,
                           [min(h, 8) for h in range(64, 0, -1)])
    return {"kind": "deterministic_cache_qualification_not_efficacy",
            "all_state_prefix_affine_identity_max_error": max(identity_errors),
            "off_anchor_true_finite_return_advantage": true,
            "off_anchor_certified_upper_with_exact_moments": upper,
            "off_anchor_margin": upper-true,
            "analytic_nonvacuity": {"uses_exact_means_not_collected_data": True,
                                    "hypothetical_n_per_state": 4096,
                                    "beta": result["beta"].tolist(),
                                    "upper_advantage": result["upper_advantage"],
                                    "true_advantage": adv},
            "H64_S2_n128_arithmetic_cost": long_cost,
            "cached_short_unroll_H8_arithmetic_cost": short_cost,
            "long_to_cached_short_transition_ratio": long_cost["cached_expected_transitions"]/short_cost["cached_expected_transitions"],
            "decision": {"fixed_law_uniform_parameter_horizon_reuse": "qualified_with_residual_penalty_and_fresh_reset_access",
                         "global_low_cost_claim": False,
                         "reuse_alone_final_mechanism": "reject_as_sufficient_paper_contribution",
                         "new_efficacy_pilot_authorized": False,
                         "formal_authorized": False, "gpu_authorized": False}}


if __name__ == "__main__":
    print(json.dumps(qualification_report(), indent=2, sort_keys=True, allow_nan=False))
