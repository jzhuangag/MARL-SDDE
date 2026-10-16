"""Complete learning code-path regression on TENSORS, never native MARL data.

No environment constructor/reset/step is invoked. Synthetic evaluation is a
constant zero fixture, never ranked, saved as a learning result or used to tune
any algorithm. Real HARL actors, critics, SGD, Adam and loss backwards execute.
"""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from sealed_bridge_learning import benchmark as b
from sealed_bridge_learning.native import actor_bytes, qualified


def initialize_tensors(task, seed, p):
    import gym
    import yaml
    from harl.algorithms.actors.happo import HAPPO
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.valuenorm import ValueNorm
    cfg = yaml.safe_load((qualified.base.UPSTREAM/'harl/configs/algos_cfgs/happo.yaml').read_text())
    args = {**cfg['model'], **cfg['algo'], 'hidden_sizes': p['hidden_sizes'],
            'activation_func': p['activation']}
    actors = []
    n = 3 if task['name'] == 'spread' else 2
    for i in range(n):
        torch.manual_seed(seed+i)
        obs = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        ac = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float32) if task['family'] == 'mamujoco' else gym.spaces.Discrete(5)
        actors.append(HAPPO(args, obs, ac))
    torch.manual_seed(seed+100)
    critic = VCritic(args, gym.spaces.Box(-np.inf, np.inf, shape=(n*4,), dtype=np.float32))
    return actors, critic, ValueNorm(1)


class TensorSampler:
    def __init__(self, task, horizon, seed, task_index, charges, vector_width):
        self.h, self.seed, self.task, self.charges = horizon, seed, task_index, charges

    def close(self):
        pass

    def collect(self, actors, ordinal, role, count, *, source=None, source_fraction=.5,
                purpose='training', deterministic=False):
        keys = [qualified.base.seed_for(self.seed, self.task, ordinal, role, e) for e in range(count)]
        P = actor_bytes(actors)+(actor_bytes(source) if source is not None else 0)
        self.charges.add('tensor_fixture:policies', policy_broadcast_bytes=P,
                         messages=len(actors)*(2 if source is not None else 1))
        self.charges.add('tensor_fixture:rows', **{'native_'+purpose: count*self.h},
                         sampling_actor_rows=len(actors)*count*self.h)
        if purpose != 'training':
            self.charges.add('tensor_fixture:evaluation_upload', program_upload_bytes=8*count, messages=1)
            return dict(returns=np.zeros(count), identities=keys, native_transitions=count*self.h,
                        controller_eligible=False)
        rows = []
        for key in keys:
            rng = np.random.default_rng(key)
            mode = int(source is not None and rng.random() < source_fraction)
            team = source if mode else actors
            obs, actions, logs = [], [], []
            for actor in team:
                o = rng.normal(size=(self.h, 4)).astype(np.float32)
                with torch.no_grad():
                    dist = qualified.distribution(actor, o)
                    if actor.actor.act.action_type == 'Discrete':
                        prob = dist.probs.numpy()
                        ac = np.asarray([rng.choice(5, p=x) for x in prob])
                        lp = dist.log_prob(torch.as_tensor(ac)).numpy()[:, None]
                    else:
                        ac = dist.mean.numpy()+dist.stddev.numpy()*rng.normal(size=dist.mean.shape)
                        lp = dist.log_prob(torch.as_tensor(ac)).numpy()
                obs.append(o); actions.append(ac); logs.append(lp)
            rows.append(dict(obs=obs, actions=actions, logs=logs, states=np.concatenate(obs, axis=-1),
                             rewards=rng.normal(size=self.h), mode=mode))
        data = dict(obs=[np.stack([x['obs'][i] for x in rows]) for i in range(len(actors))],
                    actions=[np.stack([x['actions'][i] for x in rows]) for i in range(len(actors))],
                    logs=[np.stack([x['logs'][i] for x in rows]) for i in range(len(actors))],
                    states=np.stack([x['states'] for x in rows]), rewards=np.stack([x['rewards'] for x in rows]),
                    identities=keys, modes=[x['mode'] for x in rows], SYNTHETIC_TEST_ONLY=True)
        self.charges.add('tensor_fixture:program', program_upload_bytes=qualified.payload_bytes(data), messages=len(actors)+1)
        return data


class CompleteBackend(unittest.TestCase):
    def test_every_learning_route_on_real_networks_without_native_data(self):
        torch.set_num_threads(1)
        plan = dict(tasks=[dict(name='spread', family='mpe', horizon=3, training_transitions=120),
                           dict(name='reference', family='mpe', horizon=3, training_transitions=120),
                           dict(name='halfcheetah_2x3', family='mamujoco', horizon=3, training_transitions=120)],
                    hidden_sizes=[8, 8], activation='tanh', gamma=.99, ppo_epochs=2, critic_epochs=2,
                    ppo_learning_rate=.0005, pg_base_rate=.0003, pg_rate_cap=.001,
                    evaluation_intervals=4, evaluation_episodes=2, deterministic_evaluation=False,
                    vector_width=4, calibration_episodes=2, source_episodes=2,
                    bridge_episodes=[2, 4], betas=[0., .5, 1.], fixed_bridge_episodes=4, fixed_beta=.5,
                    resource_targets=[.85, .9], variance_tradeoff=1., source_fraction=.5,
                    delay_min=1, delay_max=2, fresh_batch_episodes=4, joint_kl_cap=.1,
                    vtrace_rho_cap=1., vtrace_c_cap=1., page_refresh_probability=.5,
                    page_refresh_episodes=4, page_bridge_episodes=2,
                    roles=dict(controller=71, evaluation=72, calibration=73, source=74,
                               bridge=75, fresh=76, terminal=77))
        initial = {}
        with patch.object(b, 'Sampler', TensorSampler), patch.object(b.qualified, 'initialize', initialize_tensors), \
                patch.object(b.qualified, 'make_env', side_effect=AssertionError('NATIVE DATA FORBIDDEN IN TENSOR TEST')):
            for task_index in range(3):
                for method in b.METHODS:
                    with self.subTest(task=task_index, method=method):
                        row = b.run_cell(task_index, 73111, method, plan, lambda: None)
                        self.assertEqual(row['training_transitions'], 120)
                        self.assertEqual(row['costs']['native_training'], 120)
                        self.assertTrue(row['all_sources_drained'])
                        self.assertFalse(row['evaluation_used_by_controller'])
                        self.assertEqual(len(row['curves']), 5)
                        self.assertEqual(row['curves'][-1]['training_transitions'], 120)
                        self.assertEqual(row['curves'][-1]['policy_version'], len([x for x in row['transactions']
                            if x['kind'] not in ('charged_calibration',) and not (
                                x['kind'] == 'delayed_reference_receipt' and not x['details'].get('accepted', True))]))
                        initial.setdefault(task_index, row['initial_hash'])
                        self.assertEqual(initial[task_index], row['initial_hash'])


if __name__ == '__main__':
    unittest.main()
