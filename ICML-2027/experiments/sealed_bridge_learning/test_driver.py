"""NEW audit/guard regression, synthetic data only; not another native study."""
import copy
import json
import unittest
from unittest.mock import patch
import torch
from sealed_bridge_learning import benchmark as b
from sealed_bridge_learning.analyze import audit, mapping
from sealed_bridge_learning.guard import PLAN, profile_plan
from sealed_bridge_learning.test_backend import TensorSampler, initialize_tensors


class AuditAndGuard(unittest.TestCase):
    def test_no_lock_fails_before_initialization(self):
        with patch.object(b.qualified, 'initialize', side_effect=AssertionError('MUST NOT INITIALIZE')):
            with self.assertRaises(RuntimeError):
                b.run_cell(0, 1, 'full', {}, lambda: None)

    def test_complete_prospective_mapping(self):
        plan = json.loads(PLAN.read_text())
        self.assertEqual(len(mapping(plan)), 117)
        self.assertEqual(len(mapping(plan, True)), 6)
        self.assertTrue(set(mapping(plan, True)) <= set(mapping(plan)))
        self.assertNotIn(plan['profile']['seed'], plan['seeds'])
        self.assertEqual(profile_plan(plan)['tasks'][2]['horizon'], 1000)

    def test_new_cost_and_controller_audit_and_tamper_rejection(self):
        torch.set_num_threads(1)
        plan = json.loads(PLAN.read_text())
        plan['tasks'] = [dict(name='spread', family='mpe', horizon=3, training_transitions=240)]
        plan.update(hidden_sizes=[8, 8], ppo_epochs=2, critic_epochs=2,
                    evaluation_intervals=2, evaluation_episodes=2, calibration_episodes=2,
                    source_episodes=2, bridge_episodes=[2, 4], fixed_bridge_episodes=4,
                    fresh_batch_episodes=4, vector_width=4)
        with patch.object(b, 'Sampler', TensorSampler), patch.object(b, 'require_cell', return_value=None), \
             patch.object(b.qualified, 'initialize', initialize_tensors), \
             patch.object(b.qualified, 'make_env', side_effect=AssertionError('NO NATIVE CALL')):
            for method in ('full', 'fresh_happo'):
                row = b.run_cell(0, 73113, method, plan, lambda: None)
                audit(row, plan)
                for field in ('native_training', 'logical_bytes', 'work_rows'):
                    broken = copy.deepcopy(row); broken['costs'][field] += 1
                    with self.assertRaises(AssertionError):
                        audit(broken, plan)
                if method == 'full':
                    broken = copy.deepcopy(row)
                    t = next(t for t in broken['transactions'] if t['kind'] == 'bridge_receipt')
                    t['feedback']['queues_after'][0] += 1
                    with self.assertRaises(AssertionError):
                        audit(broken, plan)


if __name__ == '__main__':
    unittest.main()
