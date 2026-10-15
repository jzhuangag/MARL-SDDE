"""Pre-data geometry transaction and complete-charge unit checks; no env."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/"amortized_packet_native"))
import unittest
import numpy as np
from controller import AmortizedController
from current_packet_value_control import ProbeOption


class ControllerTests(unittest.TestCase):
    def test_geometry_is_prebank_and_all_option_feedback_retains_remainder(self):
        c=AmortizedController([ProbeOption(2,.1)]*2,["full_vector","response_basis"],
            gain_scale=2.,value_weight=1.,meta_step=.1,resource_targets=(1.,1.))
        opened=c.begin(coefficients=[[0.],[1.],[2.]],basis_indices=[1],
            candidate_hashes=["idle","unit","double"],candidate_costs=[[0,0],[.1,0],[.1,0]],
            probe_costs=[[.2,.3],[.2,.3]],sunk_cost=[.5,.5],rng=np.random.default_rng(4))
        selected=c.select(np.array([[0.,2.,8.],[0.,-2.,0.]]),candidate_hashes=["idle","unit","double"])
        closed=c.close(committed_hash=selected["candidate_hash"],actual_total_charge=selected["expected_total_charge"])
        self.assertEqual(opened["geometry"],closed["geometry"])
        self.assertGreater(closed["geometry_statistics"]["response_basis"]["remainder_statistic"],0)
        self.assertEqual(len(closed["expected_risk_feedback"]),2)
        self.assertFalse(closed["selected_score_is_unbiased"])

    def test_bad_charge_fails_without_queue_update(self):
        c=AmortizedController([ProbeOption(2,.1)],["full_vector"],gain_scale=1.,
            value_weight=1.,meta_step=.1,resource_targets=(1.,1.))
        c.begin(coefficients=np.empty((1,0)),basis_indices=[],candidate_hashes=["idle"],
            candidate_costs=[[0,0]],probe_costs=[[0,0]],sunk_cost=[1,1],rng=np.random.default_rng(1))
        c.select(np.zeros((2,1)),candidate_hashes=["idle"])
        with self.assertRaises(AssertionError):
            c.close(committed_hash="idle",actual_total_charge=[0,0])
        np.testing.assert_array_equal(c.queues,[0,0])


if __name__ == "__main__":
    unittest.main()
