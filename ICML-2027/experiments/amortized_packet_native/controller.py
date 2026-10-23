"""Predictable geometry/probe control; full actual-return receipt argmax."""
import numpy as np
from current_packet_value_control import CurrentPacketController, project_simplex
from amortized_packet_value import value_geometry, expected_risk_feedback


class AmortizedController(CurrentPacketController):
    def __init__(self, options, families, **kwargs):
        super().__init__(options, **kwargs)
        self.families = tuple(families)
        assert len(self.families) == len(self.options)
        assert set(self.families) <= {"full_vector", "response_basis"}

    def begin(self, *, coefficients, basis_indices, **kwargs):
        a = np.array(coefficients, dtype=float, copy=True)
        a.setflags(write=False)
        indices = tuple(basis_indices)
        result = super().begin(**kwargs)
        self._open.update(coefficients=a, basis_indices=indices)
        result["geometry"] = self.families[result["probe_index"]]
        return result

    def select(self, samples, **kwargs):
        result = super().select(samples, **kwargs)
        self._open["geometry"] = value_geometry(
            np.asarray(samples)/self.scale, self._open["coefficients"], self._open["basis_indices"])
        return result

    def close(self, *, committed_hash, actual_total_charge):
        d = self._open
        assert d is not None and d["selection"] is not None
        chosen = d["selection"]
        assert committed_hash == d["hashes"][chosen["index"]]
        expected = d["sunk"] + d["costs"][chosen["index"]] + d["probe_costs"][d["probe"]]
        charged = self._costs(actual_total_charge, (2,))
        assert np.allclose(charged, expected, rtol=0, atol=1e-10)
        risk = np.array([expected_risk_feedback(d["geometry"], g, p.episodes, p.young_scale)
                         for g,p in zip(self.families,self.options)])
        loss = d["probe_costs"] @ d["queues"] + self.V*risk
        z = project_simplex(self.z-self.eta*loss)
        q = np.maximum(0., d["queues"]+charged-self.targets)
        record = dict(round=self.round, probe_index=d["probe"],
            episodes=self.options[d["probe"]].episodes, geometry=self.families[d["probe"]],
            candidate_index=chosen["index"], committed_hash=committed_hash,
            normalized_score_vector=chosen["means"].tolist(),
            sample_variance_trace=chosen["variance_trace"], geometry_statistics=d["geometry"],
            expected_risk_feedback=risk.tolist(), probe_loss_estimates=loss.tolist(),
            probe_probability=d["distribution"].tolist(), sunk_cost=d["sunk"].tolist(),
            total_charge=charged.tolist(), queues_before=d["queues"].tolist(), queues_after=q.tolist(),
            representation_before_bank=True, selected_score_is_unbiased=False,
            sample_variance_is_upper_certificate=False, hard_budget_guarantee=False)
        self.z,self.queues=z,q; self.round+=1; self._open=None
        return record
