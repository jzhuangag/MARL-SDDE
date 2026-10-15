"""Current-context receipt selection with explicitly paid predictable probing.

Successor prototype, NOT part of the frozen causal_recycle experiment.
The action may use the current bank; the probe count may not. Selection bias is
handled by an expected selection-regret bound, not an unbiased selected score.
No empirical variance is advertised as a confidence upper bound.
"""
from dataclasses import dataclass
import math
import numpy as np


def project_simplex(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty vector required")
    ordered = np.sort(values)[::-1]
    partial = np.cumsum(ordered) - 1.
    support = np.flatnonzero(ordered - partial / np.arange(1, len(values) + 1) > 0)
    projected = np.maximum(values - partial[support[-1]] / (support[-1] + 1), 0.)
    return projected / projected.sum()


@dataclass(frozen=True)
class ProbeOption:
    episodes: int
    young_scale: float

    def __post_init__(self):
        if isinstance(self.episodes, bool) or int(self.episodes) != self.episodes or self.episodes < 2:
            raise ValueError("sample-variance feedback needs at least two independent episodes")
        if not math.isfinite(self.young_scale) or self.young_scale <= 0:
            raise ValueError("strictly positive finite Young scale required")


class CurrentPacketController:
    """Two virtual resource queues; finite (probe count, Young scale) catalogue.

    Candidate zero is the unchanged policy, with zero contrast and incremental
    cost. All full-policy hashes must be unique: effective duplicates are removed
    before scoring. Costs use any fixed, preregistered nonnegative unit system.
    Column 0 is communication, column 1 is computation. Source work, proposal
    work and eventual incoming packets remain in sunk_cost, never refunded.
    """

    def __init__(self, options, *, gain_scale, value_weight, meta_step, resource_targets):
        self.options = tuple(options)
        if not self.options or any(not isinstance(p, ProbeOption) for p in self.options):
            raise ValueError("explicit nonempty ProbeOption catalogue required")
        for value in (gain_scale, value_weight, meta_step):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite design constants required")
        self.targets = self._costs(resource_targets, (2,)).copy()
        self.scale, self.V, self.eta = gain_scale, value_weight, meta_step
        self.queues = np.zeros(2)
        self.z = np.full(len(self.options), 1. / len(self.options))
        self.round = 0
        self._open = None

    @staticmethod
    def _costs(values, shape):
        array = np.asarray(values, dtype=float)
        if array.shape != shape or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("finite nonnegative costs with exact registered shape required")
        return array

    def begin(self, *, candidate_hashes, candidate_costs, probe_costs, sunk_cost, rng):
        """Freeze policies/costs and choose m BEFORE opening independent episodes.

        probe_costs is the predictable extra computation charge for each option.
        Caller must freeze the behavior policy, MC potential, candidates and all
        eligibility constraints too. This software contract cannot prove that a
        caller's return observations are unbiased or sampled independently.
        """
        if self._open is not None:
            raise RuntimeError("previous receipt transaction is not closed")
        hashes = tuple(candidate_hashes)
        if not hashes or any(not isinstance(x, str) or not x for x in hashes) or len(set(hashes)) != len(hashes):
            raise ValueError("unique immutable full-policy identifiers required")
        costs = self._costs(candidate_costs, (len(hashes), 2)).copy()
        if np.any(costs[0] != 0):
            raise ValueError("candidate zero must be the unchanged, zero-increment reference")
        probes = self._costs(probe_costs, (len(self.options),)).copy()
        sunk = self._costs(sunk_cost, (2,)).copy()
        lambdas = np.array([p.young_scale for p in self.options])
        known_loss = self.queues[1] * probes + self.V * lambdas
        distribution = project_simplex(self.z - self.eta * known_loss)
        selected = int(rng.choice(len(self.options), p=distribution))
        self._open = dict(hashes=hashes, costs=costs, probe_costs=probes, sunk=sunk,
                          known_loss=known_loss, distribution=distribution, probe=selected,
                          queues=self.queues.copy(), selection=None)
        return {"round": self.round, "probe_index": selected,
                "episodes": int(self.options[selected].episodes),
                "probe_probability": distribution.tolist()}

    def select(self, contrast_samples, *, candidate_hashes):
        """Use this bank's scores to select this bank's receipt action.

        Exact, unclipped actual-return contrasts are required, not PPO losses or
        evaluation returns. Same-bank selection makes its selected score biased.
        """
        d = self._open
        if d is None or d["selection"] is not None:
            raise RuntimeError("begin once, select once, then close")
        if tuple(candidate_hashes) != d["hashes"]:
            raise ValueError("a candidate changed after the probe decision")
        m = self.options[d["probe"]].episodes
        samples = np.asarray(contrast_samples, dtype=float)
        if samples.shape != (m, len(d["hashes"])) or not np.isfinite(samples).all():
            raise ValueError("finite, independent episode matrix with frozen shape required")
        if np.any(samples[:, 0] != 0):
            raise ValueError("unchanged-policy contrast must be exactly zero")
        normalized = samples / self.scale
        means = normalized.mean(axis=0)
        variance_trace = float(normalized.var(axis=0, ddof=1).sum())
        utility = self.V * means - d["costs"] @ d["queues"]
        if not np.isfinite(utility).all() or not math.isfinite(variance_trace):
            raise FloatingPointError("nonfinite selection or variance feedback")
        index = int(np.argmax(utility))  # deterministic, idle-first tie breaking
        d["selection"] = dict(index=index, means=means, variance_trace=variance_trace)
        return {"round": self.round, "candidate_index": index,
                "candidate_hash": d["hashes"][index],
                "expected_total_charge": (d["sunk"] + d["costs"][index] +
                                          np.array([0., d["probe_costs"][d["probe"]]])).tolist(),
                "selected_score_is_unbiased": False,
                "sample_variance_is_upper_certificate": False}

    def close(self, *, committed_hash, actual_total_charge):
        """Validate the commit and complete full-cost queues and probe feedback.

        Mismatch fails closed; no state update or replacement attempt is made.
        A failed native integration must preserve its transaction and stop.
        """
        d = self._open
        if d is None or d["selection"] is None:
            raise RuntimeError("selection and commit must precede close")
        chosen = d["selection"]
        if committed_hash != d["hashes"][chosen["index"]]:
            raise ValueError("committed policy differs from frozen selection")
        expected = d["sunk"] + d["costs"][chosen["index"]] + np.array([0., d["probe_costs"][d["probe"]]])
        charged = self._costs(actual_total_charge, (2,))
        if not np.allclose(charged, expected, rtol=0., atol=1e-10):
            raise ValueError("paid probe/source/proposal/receipt charges do not reconcile")
        factors = np.array([1. / (p.young_scale * p.episodes) for p in self.options])
        noise_loss = self.V * chosen["variance_trace"] * factors
        loss = d["known_loss"] + noise_loss
        updated_z = project_simplex(self.z - self.eta * loss)
        updated_queues = np.maximum(0., d["queues"] + charged - self.targets)
        record = {"round": self.round, "probe_index": d["probe"],
                  "episodes": int(self.options[d["probe"]].episodes),
                  "candidate_index": chosen["index"], "committed_hash": committed_hash,
                  "normalized_score_vector": chosen["means"].tolist(),
                  "sample_variance_trace": chosen["variance_trace"],
                  "probe_probability": d["distribution"].tolist(),
                  "probe_loss_estimates": loss.tolist(),
                  "sunk_cost": d["sunk"].tolist(), "total_charge": charged.tolist(),
                  "queues_before": d["queues"].tolist(), "queues_after": updated_queues.tolist(),
                  "selected_score_is_unbiased": False,
                  "sample_variance_is_upper_certificate": False,
                  "hard_budget_guarantee": False}
        self.z, self.queues = updated_z, updated_queues
        self.round += 1
        self._open = None
        return record
