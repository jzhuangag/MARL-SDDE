"""Observable TD product-norm certificate and truncated-risk reference.

Fixed-law independent reset blocks, not arbitrary replay. Exact-profile and
qualification functions are oracle diagnostics; online code never receives P.
No frozen runner, gate, seed or scientific result is modified.
"""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from observable_transfer_reference import _positive_int, _state
from reusable_transfer_cache import cached_controller


@dataclass(frozen=True)
class ContractionSpec:
    dimension: int
    discount: float
    step: float
    block: int
    replicas: int
    delta: float
    law_tag: str

    def __post_init__(self):
        for name in ("dimension", "block", "replicas"):
            _positive_int(getattr(self, name), name)
        if (not 0 <= self.discount < 1 or not 0 < self.step <= 1
                or not 0 < self.delta < 1 or not isinstance(self.law_tag, str)
                or not self.law_tag):
            raise ValueError("invalid fixed contraction contract")

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()


def observe_norm_block(spec, start, transitions):
    """Compute ||B_(m-1)...B_0||_infinity with one propagated d-vector.

    Each observation is (state,reward,next_state). Rewards do not affect this
    sensitivity but their transitions are charged. Independence is a collection
    requirement; it cannot be inferred from distinct IDs or numerical records.
    """
    _state(start, spec.dimension)
    z, current, count = np.ones(spec.dimension), start, 0
    for s, reward, j in transitions:
        _state(s, spec.dimension)
        _state(j, spec.dimension)
        if s != current or not np.isfinite(reward) or count >= spec.block:
            raise ValueError("invalid observed block")
        z[s] = (1-spec.step)*z[s]+spec.step*spec.discount*z[j]
        current, count = j, count+1
    if count != spec.block:
        raise ValueError("block length mismatch")
    norm = float(np.max(z))
    return {"norm": norm, "norm_square": norm*norm, "transitions": count,
            "reset_requests": 1, "start_state": start, "contract": spec.digest}


def bounds_from_means(spec, means, square_means):
    """Pure algebra. Empirical certification additionally requires valid batches."""
    x, y = np.asarray(means, float), np.asarray(square_means, float)
    if (x.shape != (spec.dimension,) or y.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(x < 0) or np.any(x > 1) or np.any(y < 0)
            or np.any(y > x+1e-12) or np.any(y+1e-12 < x*x)):
        raise ValueError("invalid per-state first/second norm moments")
    radius = math.sqrt(math.log(2*spec.dimension/spec.delta)/(2*spec.replicas))
    kappa1 = min(1., float(x.max())+radius)
    kappa2 = min(kappa1, float(y.max())+radius)
    return {"kappa1": kappa1, "kappa2": kappa2, "radius": radius,
            "strict_contraction": kappa1 < 1,
            "structural_unvisited_state_obstruction": spec.block < spec.dimension}


class ContractionBatch:
    """Exactly n fresh blocks from EVERY state; no peeking/stopping/refresh."""
    def __init__(self, spec):
        self.spec = spec
        self.counts = [0]*spec.dimension
        self.sums = np.zeros(spec.dimension)
        self.squares = np.zeros(spec.dimension)
        self.transitions = self.resets = 0
        self.ready = False

    def add(self, start, index, transitions, law_tag):
        _state(start, self.spec.dimension)
        if (self.ready or law_tag != self.spec.law_tag or isinstance(index, bool)
                or not isinstance(index, int) or index != self.counts[start]
                or index >= self.spec.replicas):
            raise ValueError("wrong law, duplicate ID, or completed batch")
        sample = observe_norm_block(self.spec, start, transitions)
        self.sums[start] += sample["norm"]
        self.squares[start] += sample["norm_square"]
        self.counts[start] += 1
        self.transitions += sample["transitions"]
        self.resets += 1

    def finalize(self):
        if self.ready or any(n != self.spec.replicas for n in self.counts):
            raise ValueError("requires complete fixed batches at every state")
        result = bounds_from_means(self.spec, self.sums/self.spec.replicas,
                                   self.squares/self.spec.replicas)
        self._bounds = result
        self.sums.flags.writeable = self.squares.flags.writeable = False
        self.ready = True
        return dict(result)

    @property
    def bounds(self):
        if not self.ready:
            raise ValueError("contraction batch is not finalized")
        return dict(self._bounds)


def integrated_tail(horizon, block, kappa):
    """Sum_{t=m}^{h-1} kappa**floor(t/m), computed in O(1)."""
    _positive_int(horizon, "horizon")
    _positive_int(block, "block")
    if not np.isfinite(kappa) or not 0 <= kappa <= 1:
        raise ValueError("invalid contraction upper bound")
    if horizon <= block or kappa == 0:
        return 0.
    if kappa == 1:
        return float(horizon-block)
    blocks, remainder = divmod(horizon, block)
    log_k = math.log(kappa)
    full = block*kappa*(-math.expm1((blocks-1)*log_k))/(1-kappa)
    return full+remainder*math.exp(blocks*log_k)


def terminal_resolvent(horizon, block, kappa):
    """Sum after an already propagated m-step terminal sensitivity."""
    _positive_int(horizon, "horizon")
    _positive_int(block, "block")
    if not np.isfinite(kappa) or not 0 <= kappa <= 1:
        raise ValueError("invalid contraction upper bound")
    if horizon <= block:
        return 0.
    remaining = horizon-block
    return min(remaining, block)+integrated_tail(remaining, block, kappa)


def terminal_bounds_from_means(spec, basis, first, second, global_bounds):
    """Per-state directional bounds; delta/2 is allocated to these moments."""
    E = np.asarray(basis, dtype=float)
    x, y = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (E.ndim != 2 or E.shape[0] != spec.dimension or E.shape[1] < 1
            or not np.isfinite(E).all() or x.shape != (spec.dimension, E.shape[1])
            or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("invalid directional moments")
    Delta = np.max(np.abs(E), axis=0)
    if (np.any(x < 0) or np.any(x > Delta+1e-12) or np.any(y < 0)
            or np.any(y > Delta*x+1e-12) or np.any(y+1e-12 < x*x)):
        raise ValueError("directional moments violate boundedness")
    radius = math.sqrt(math.log(4*spec.dimension*E.shape[1]/spec.delta)/(2*spec.replicas))
    upper1 = np.minimum(x+Delta*radius, Delta*global_bounds["kappa1"])
    upper2 = np.minimum(y+Delta*Delta*radius, Delta*Delta*global_bounds["kappa2"])
    upper2 = np.minimum(upper2, Delta*upper1)
    return {"first": upper1, "second": upper2, "radius": radius}


class DirectionalContractionBatch:
    """Same paid blocks yield the global norm AND fixed-basis terminal norms.

    No new environment transitions are charged for deterministic propagation
    on existing records. Work is O(n*d*(d*r+m*r)) for all d starting states;
    rewards are not independent training samples merely because reused here.
    """
    def __init__(self, spec, basis):
        from dataclasses import replace
        E = np.array(basis, dtype=float, copy=True)
        if E.ndim != 2 or E.shape[0] != spec.dimension or E.shape[1] < 1 or not np.isfinite(E).all():
            raise ValueError("invalid fixed basis")
        E.flags.writeable = False
        self.spec, self.basis = spec, E
        self.global_batch = ContractionBatch(replace(spec, delta=spec.delta/2))
        self.first = np.zeros((spec.dimension, E.shape[1]))
        self.second = np.zeros_like(self.first)
        self.ready = False

    def add(self, start, index, transitions, law_tag):
        records = list(transitions)
        # Validation and charging occur exactly once. Re-propagation is CPU work.
        self.global_batch.add(start, index, records, law_tag)
        U = self.basis.copy()
        for s, _, j in records:
            U[s] = (1-self.spec.step)*U[s]+self.spec.step*self.spec.discount*U[j]
        norms = np.max(np.abs(U), axis=0)
        self.first[start] += norms
        self.second[start] += norms*norms

    def finalize(self):
        global_bounds = self.global_batch.finalize()
        self._terminal = terminal_bounds_from_means(
            self.spec, self.basis, self.first/self.spec.replicas,
            self.second/self.spec.replicas, global_bounds)
        self.first.flags.writeable = self.second.flags.writeable = False
        self.ready = True

    @property
    def bounds(self):
        return self.global_batch.bounds

    @property
    def terminal(self):
        if not self.ready:
            raise ValueError("directional batch is not finalized")
        return {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._terminal.items()}


def augment_directional_tail(head, horizon, block, value_bound, mapping,
                             residuals, terminal_first, terminal_second, bounds):
    """Minkowski controls mixed directions; residual F uses global norm bounds."""
    A, nu, p, q = [np.asarray(x, dtype=float) for x in
                   (mapping, residuals, terminal_first, terminal_second)]
    if (A.ndim != 2 or nu.shape != (A.shape[1],) or p.shape != (A.shape[0],)
            or q.shape != p.shape or not all(np.isfinite(x).all() for x in (A, nu, p, q))
            or np.any(nu < 0) or np.any(p < 0) or np.any(q < 0)
            or value_bound <= 0 or not np.isfinite(value_bound)
            or not 0 <= bounds["kappa2"] <= bounds["kappa1"] <= 1):
        raise ValueError("invalid directional tail inputs")
    psi1 = terminal_resolvent(horizon, block, bounds["kappa1"])
    psi2 = terminal_resolvent(horizon, block, bounds["kappa2"])
    linear_residue = np.abs(A).T@p+bounds["kappa1"]*nu
    rms_residue = np.abs(A).T@np.sqrt(q)+math.sqrt(bounds["kappa2"])*nu
    return {"matrix": head["matrix"]+psi2*np.outer(rms_residue, rms_residue),
            "linear": head["linear"]+2*value_bound*psi1*linear_residue,
            "psi1": psi1, "psi2": psi2}


def directional_transfer(cache, contraction, state, horizon, value, directions,
                         coordinates, mapping, *, law_tag, iterations=100):
    spec, anchor = contraction.spec, cache.anchor
    if (not cache.ready or not contraction.ready or law_tag != spec.law_tag
            or law_tag != cache.law_tag or anchor.horizon != spec.block
            or not np.array_equal(anchor.directions, contraction.basis)
            or anchor.step != spec.step or anchor.discount != spec.discount
            or set(cache.states) != set(range(spec.dimension)) or cache.delta+spec.delta >= 1):
        raise ValueError("incompatible directional cache/certificate")
    head = cache.query(state, min(horizon, spec.block), value, directions, coordinates,
                        mapping, law_tag=law_tag, discount=spec.discount, step=spec.step)
    terminal = contraction.terminal
    upper = augment_directional_tail(head, horizon, spec.block, anchor.value_bound,
                                      mapping, head["direction_residuals"],
                                      terminal["first"][state], terminal["second"][state],
                                      contraction.bounds)
    return {**cached_controller(upper, iterations), "psi1": upper["psi1"],
            "joint_failure_probability": cache.delta+spec.delta}


def augment_with_tail(problem, directions, value_bound, horizon, block, bounds):
    D = np.asarray(directions, dtype=float)
    if (D.ndim != 2 or not D.size or not np.isfinite(D).all()
            or not np.isfinite(value_bound) or value_bound <= 0
            or bounds["kappa2"] > bounds["kappa1"]):
        raise ValueError("invalid tail inputs")
    delta = np.max(np.abs(D), axis=0)
    phi1 = integrated_tail(horizon, block, bounds["kappa1"])
    phi2 = integrated_tail(horizon, block, bounds["kappa2"])
    return {"matrix": problem["matrix"]+phi2*np.outer(delta, delta),
            "linear": problem["linear"]+2*value_bound*phi1*delta,
            "phi1": phi1, "phi2": phi2,
            "structural_fallback_on_coverage": phi1 >= min(block, horizon)}


def certified_transfer(cache, contraction, state, horizon, value, directions,
                       coordinates, mapping, *, law_tag, iterations=100):
    """A short-unroll QP with a certified tail, not a separate novel optimizer."""
    spec, anchor = contraction.spec, cache.anchor
    if (not cache.ready or not contraction.ready or law_tag != spec.law_tag
            or law_tag != cache.law_tag or anchor.horizon != spec.block
            or len(anchor.value) != spec.dimension or anchor.step != spec.step
            or anchor.discount != spec.discount or set(cache.states) != set(range(spec.dimension))
            or cache.delta+spec.delta >= 1):
        raise ValueError("mismatched/incomplete cache, law, or probability contract")
    head = cache.query(state, min(horizon, spec.block), value, directions,
                        coordinates, mapping, law_tag=law_tag,
                        discount=spec.discount, step=spec.step)
    upper = augment_with_tail(head, directions, anchor.value_bound, horizon,
                              spec.block, contraction.bounds)
    # On the simultaneous event this branch cannot certify nonzero benefit.
    # Zero is always feasible, even on the event's complement.
    if upper["structural_fallback_on_coverage"]:
        result = {"beta": np.zeros(np.asarray(directions).shape[1]),
                  "upper_advantage": 0., "optimization_gap_bound": None}
    else:
        result = cached_controller(upper, iterations)
    return {**result, "phi1": upper["phi1"], "phi2": upper["phi2"],
            "structural_fallback": upper["structural_fallback_on_coverage"],
            "joint_failure_probability": cache.delta+spec.delta}


def collection_cost(dimension, block, full_horizon, return_length, n_gram, n_contract):
    for name, value in locals().copy().items():
        _positive_int(value, name)
    if block > full_horizon:
        raise ValueError("block exceeds full horizon")
    head = dimension*n_gram*((block-1)/2+return_length)
    contraction = dimension*n_contract*block
    full = dimension*n_gram*((full_horizon-1)/2+return_length)
    return {"head_expected_transitions": head, "contraction_transitions": contraction,
            "combined_expected_transitions": head+contraction,
            "full_cache_expected_transitions_same_n": full,
            "combined_to_full_ratio": (head+contraction)/full,
            "reset_requests": dimension*(2*n_gram+n_contract),
            "strictly_cheaper_same_n": head+contraction < full,
            "equal_n_criterion_3m_less_than_H": (3*block < full_horizon) if n_gram == n_contract else None,
            "cost_scope": "separate paid batches; equal n is not equal confidence or accuracy"}


def exact_norm_profile(p, discount, step, max_block):
    """Finite oracle tree, vectorized. No random seed or observed sample batch."""
    p = np.asarray(p, dtype=float)
    d = len(p)
    if p.shape != (d, d) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("invalid oracle transition matrix")
    profile = np.empty((d, max_block, 2))
    for start in range(d):
        states, weight, rows = np.array([start]), np.ones(1), np.ones((1, d))
        for t in range(max_block):
            old = np.repeat(states, d)
            nxt = np.tile(np.arange(d), len(states))
            weight = np.repeat(weight, d)*p[old, nxt]
            rows = np.repeat(rows, d, axis=0)
            idx = np.arange(len(old))
            rows[idx, old] = (1-step)*rows[idx, old]+step*discount*rows[idx, nxt]
            states = nxt
            norm = rows.max(axis=1)
            profile[start, t] = weight@norm, weight@(norm*norm)
    return profile


def exact_terminal_profile(p, discount, step, max_block, basis):
    """Oracle-only terminal directional moments; no observed certificate."""
    p, E = np.asarray(p, dtype=float), np.asarray(basis, dtype=float)
    d, r = E.shape
    profile = np.empty((d, max_block, r, 2))
    for start in range(d):
        states, weight, rows = np.array([start]), np.ones(1), E[None, :, :].copy()
        for t in range(max_block):
            old = np.repeat(states, d)
            nxt = np.tile(np.arange(d), len(states))
            weight = np.repeat(weight, d)*p[old, nxt]
            rows = np.repeat(rows, d, axis=0)
            idx = np.arange(len(old))
            rows[idx, old] = (1-step)*rows[idx, old]+step*discount*rows[idx, nxt]
            states = nxt
            norm = np.max(np.abs(rows), axis=1)
            profile[start, t, :, 0] = weight@norm
            profile[start, t, :, 1] = weight@(norm*norm)
    return profile


def qualification_report():
    from observable_transfer_reference import ProbeSpec
    from reusable_transfer_cache import cache_radii, query_upper
    from rl_collaboration_interface_audit import markov_td_risk_gramian
    # Declared public, outcome-free diagnostic grid, not a tuned benchmark.
    p = np.array([[.8, .2], [.3, .7]])
    blocks, horizons, discounts = (2, 4, 8, 12, 16), (64, 512), (.2, .6, .9)
    rows = []
    for gamma in discounts:
        profile = exact_norm_profile(p, gamma, .5, max(blocks))
        B = 1/(1-gamma)
        v = B*np.array([1., -1.])
        terminal_profile = exact_terminal_profile(p, gamma, .5, max(blocks), -v[:, None])
        for m in blocks:
            spec = ContractionSpec(2, gamma, .5, m, 128, .005, "public-fixed-law")
            bounds = bounds_from_means(spec, profile[:, m-1, 0], profile[:, m-1, 1])
            from dataclasses import replace
            directional_global = bounds_from_means(replace(spec, delta=spec.delta/2),
                                                    profile[:, m-1, 0], profile[:, m-1, 1])
            directional_terminal = terminal_bounds_from_means(
                spec, -v[:, None], terminal_profile[:, m-1, :, 0],
                terminal_profile[:, m-1, :, 1], directional_global)
            # An explicitly privileged falsification diagnostic: homogeneous
            # zero rewards and a perfect zero donor. Never an efficacy task.
            anchor = ProbeSpec(v, -v[:, None], 0, gamma, .5, m, 32, 1.)
            gram = float(v@markov_td_risk_gramian(p, gamma, .5, m)[0]@v/2)
            head = query_upper(anchor, [[gram]], [-gram],
                               cache_radii(anchor, (0, 1), 128, .005, m),
                               m, v, -v[:, None], [0.], [[1.]])
            for h in horizons:
                phi = integrated_tail(h, m, bounds["kappa1"])
                cost = collection_cost(2, m, h, 32, 128, 128)
                actual_rule = cached_controller(augment_with_tail(head, -v[:, None], B, h, m, bounds))
                ideal_bounds = {"kappa1": float(profile[:, m-1, 0].max()),
                                "kappa2": float(profile[:, m-1, 1].max())}
                ideal_head = {"matrix": np.array([[gram]]), "linear": np.array([-gram])}
                ideal = cached_controller(augment_with_tail(ideal_head, -v[:, None], B, h, m, ideal_bounds))
                directional_ideal = cached_controller(augment_directional_tail(
                    ideal_head, h, m, B, [[1.]], [0.], terminal_profile[0, m-1, :, 0],
                    terminal_profile[0, m-1, :, 1], ideal_bounds))
                directional_finite = cached_controller(augment_directional_tail(
                    head, h, m, B, [[1.]], [0.], directional_terminal["first"][0],
                    directional_terminal["second"][0], directional_global))
                rows.append({"discount": gamma, "block": m, "horizon": h,
                             "population_max_norm_mean": float(profile[:, m-1, 0].max()),
                             **bounds, "phi1": phi,
                             "structural_fallback_at_population_plus_radius": phi >= m,
                             "cost": cost,
                             "perfect_donor_population_head_QP_beta": float(actual_rule["beta"][0]),
                             "perfect_donor_zero_uncertainty_QP_beta": float(ideal["beta"][0]),
                             "perfect_donor_directional_zero_uncertainty_beta": float(directional_ideal["beta"][0]),
                             "perfect_donor_directional_population_beta": float(directional_finite["beta"][0]),
                             "necessary_nontriviality_and_cost_screen": phi < m and cost["strictly_cheaper_same_n"]})
    return {"kind": "outcome_free_exact_norm_and_arithmetic_qualification",
            "uses_population_moments_not_a_collected_certificate": True,
            "grid": {"P": p.tolist(), "step": .5, "discounts": discounts,
                     "blocks": blocks, "horizons": horizons, "n_gram": 128,
                     "n_contraction": 128, "return_length": 32, "contraction_delta": .005},
            "rows": rows,
            "necessary_screen_pass_count": sum(r["necessary_nontriviality_and_cost_screen"] for r in rows),
            "perfect_donor_population_head_positive_actions": sum(r["perfect_donor_population_head_QP_beta"] > 0 for r in rows),
            "perfect_donor_zero_uncertainty_positive_actions": sum(r["perfect_donor_zero_uncertainty_QP_beta"] > 0 for r in rows),
            "directional_zero_uncertainty_positive_actions": sum(r["perfect_donor_directional_zero_uncertainty_beta"] > 0 for r in rows),
            "directional_population_positive_actions": sum(r["perfect_donor_directional_population_beta"] > 0 for r in rows),
            "total_cells": len(rows),
            "decision": {"norm_and_tail_theory": "qualified_under_fixed_law_reset_access",
                         "only_kappa_less_than_one_suffices": False,
                         "distinct_from_matched_certified_short_unrolling": False,
                         "empirical_activation_or_efficacy_shown": False,
                         "same_total_budget_learning_benefit_proved": False,
                         "pilot_authorized": False, "formal_authorized": False}}


if __name__ == "__main__":
    print(json.dumps(qualification_report(), indent=2, sort_keys=True, allow_nan=False))
