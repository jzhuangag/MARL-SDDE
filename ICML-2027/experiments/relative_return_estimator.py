"""First-disagreement conditional Monte Carlo for actual episodic team return.

This is a simulator-based estimator, not an observational off-policy shortcut.
Every common prefix, replayed prefix and suffix is charged. Neither benchmark
validation returns nor critic losses enter the estimator.
"""
import itertools
import math

import numpy as np


def probability(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("invalid probability vector")
    return p / p.sum()


def draw(p, rng):
    return min(int(np.searchsorted(np.cumsum(probability(p)), rng.random())), len(p) - 1)


def overlap(p, q):
    p, q = probability(p), probability(q)
    common = np.minimum(p, q)
    c = min(1.0, float(common.sum()))
    return p, q, common, c


def maximal_pair(p, q, rng):
    p, q, common, c = overlap(p, q)
    if c >= 1.0 or rng.random() < c:
        a = draw(common, rng)
        return a, a
    return draw(np.maximum(p - common, 0), rng), draw(np.maximum(q - common, 0), rng)


def common_joint(ps, qs, rng):
    parts = [overlap(p, q) for p, q in zip(ps, qs)]
    c = float(np.prod([x[3] for x in parts]))
    # If the common event has zero mass, the arbitrary continuation is never
    # used in any later positive-weight first-disagreement term.
    actions = [draw(x[2] if x[3] > 0 else x[0], rng) for x in parts]
    return actions, c


def disagreeing_joint(ps, qs, rng):
    """Product maximal coupling conditioned on at least one disagreement."""
    parts = [overlap(p, q) for p, q in zip(ps, qs)]
    survival, masses = 1.0, []
    for x in parts:
        masses.append(survival * (1 - x[3]))
        survival *= x[3]
    if sum(masses) <= 0:
        raise ValueError("cannot condition on a zero-probability disagreement")
    first = draw(masses, rng)
    aa, bb = [], []
    for i, (p, q, common, c) in enumerate(parts):
        if i < first:
            a = b = draw(common, rng)
        elif i == first:
            a, b = draw(np.maximum(p - common, 0), rng), draw(np.maximum(q - common, 0), rng)
        else:
            a, b = maximal_pair(p, q, rng)
        aa.append(a)
        bb.append(b)
    return aa, bb


def first_disagreement(common_path, branch, horizon, rng):
    """Callbacks return (actions, overlap) and a conditional suffix difference.

    The complete common path is generated before sampling the branch index.
    The branch callback must replay its prefix from the identical initial seed.
    """
    survival, weights, actions = 1.0, [], []
    for t in range(horizon):
        a, c = common_path(t)
        weights.append(survival * (1 - c))
        survival *= c
        actions.append(a)
    total = float(sum(weights))
    if total <= 0:
        return 0.0, {"weight": 0.0, "branch_time": -1}
    chosen = draw(weights, rng)
    difference = branch(chosen, actions[:chosen])
    return total * difference, {"weight": total, "branch_time": chosen}


def empirical_bernstein(samples, support_width, delta):
    """Fixed-n iid bound; known support width, never an observed sample range.

    Maurer--Pontil (2009), Theorem 4 with the two-tail union bound. A separate
    union allocation is required across candidates and repeated decision times.
    """
    x = np.asarray(samples, dtype=float)
    if len(x) < 2 or not 0 < delta < 1 or support_width < 0:
        raise ValueError("invalid confidence interface")
    log = math.log(4 / delta)
    radius = math.sqrt(2 * x.var(ddof=1) * log / len(x))
    radius += 7 * support_width * log / (3 * (len(x) - 1))
    return float(x.mean() - radius), float(radius)


def finite_game_audit():
    """Exact first-disagreement identity on a two-agent two-step Markov game.

    All prefix/suffix expectations are enumerated, not estimated or fitted.
    """
    h = 2
    def policies(s, t):
        return ([np.array([.2 + .1 * s, .8 - .1 * s]), np.array([.65, .35])],
                [np.array([.55, .45]), np.array([.4 + .1 * t, .6 - .1 * t])])
    def step(s, a):
        ns = (s + a[0] + 2 * a[1]) % 3
        return ns, float(2 * a[0] - a[1] + .3 * s + a[0] * a[1])
    def value(s, t, which):
        if t == h:
            return 0.0
        pp = policies(s, t)[which]
        out = 0.0
        for a in itertools.product(range(2), repeat=2):
            ns, r = step(s, a)
            out += pp[0][a[0]] * pp[1][a[1]] * (r + value(ns, t + 1, which))
        return out
    def joint_coupling(s, t):
        ps, qs = policies(s, t)
        couplings = []
        for p, q in zip(ps, qs):
            p, q, m, c = overlap(p, q)
            cc = np.diag(m)
            if c < 1:
                cc += np.outer(p - m, q - m) / (1 - c)
            couplings.append(cc)
        for aa in itertools.product(range(2), repeat=2):
            for bb in itertools.product(range(2), repeat=2):
                mass = couplings[0][aa[0], bb[0]] * couplings[1][aa[1], bb[1]]
                if mass > 0:
                    yield aa, bb, mass
    # Integrate each possible common-prefix path. At every prefix, the
    # unnormalized disagreement mass is multiplied by its exact suffix value.
    def terms(s, t, prefix_mass):
        if t == h:
            return 0.0
        result = 0.0
        for aa, bb, mass in joint_coupling(s, t):
            sa, ra = step(s, aa)
            if aa == bb:
                result += terms(sa, t + 1, prefix_mass * mass)
            else:
                sb, rb = step(s, bb)
                result += prefix_mass * mass * (ra - rb + value(sa, t + 1, 0) - value(sb, t + 1, 1))
        return result
    exact, reconstructed = value(0, 0, 0) - value(0, 0, 1), terms(0, 0, 1.0)
    if abs(exact - reconstructed) > 1e-12:
        raise AssertionError("first-disagreement identity failed")
    return {"exact_return_difference": exact, "first_disagreement_expectation": reconstructed,
            "absolute_error": abs(exact - reconstructed)}
