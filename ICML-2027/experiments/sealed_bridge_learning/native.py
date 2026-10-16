"""Batched complete-policy MARL sampling, evaluation and full-cost primitives.

No module-level rollout or environment creation. Native calls require the later
science driver to verify its separately pushed source/config/role lock first.
This is a NEW backend, never an amendment to the passed integration sources.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealed_bridge import native as qualified
from sealed_bridge_learning.prefix import paired_episode_gradients


class Charges:
    """Declared logical transport and neural row proxies, not wire bytes/FLOPs."""
    FIELDS = ('native_training', 'native_benchmark', 'native_frozen_policy',
              'sampling_actor_rows', 'actor_forward_rows', 'actor_backward_rows',
              'critic_forward_rows', 'critic_backward_rows', 'actor_adam_calls',
              'actor_sgd_calls', 'critic_adam_calls', 'copy_tensor_bytes',
              'program_upload_bytes', 'policy_broadcast_bytes',
              'gradient_storage_bytes', 'prefix_scalar_entries',
              'variance_scalar_entries', 'messages')

    def __init__(self):
        self.values = dict.fromkeys(self.FIELDS, 0)
        self.events = {}

    def add(self, event, **values):
        if (set(values)-set(self.FIELDS) or
                any(not isinstance(v, (int, np.integer)) or v < 0 for v in values.values())):
            raise ValueError('Unregistered or noninteger cost entry')
        values = {k: int(v) for k, v in values.items()}
        for k, v in values.items():
            self.values[k] += v
        aggregate = self.events.setdefault(event, dict(calls=0))
        aggregate['calls'] += 1
        for k, v in values.items():
            aggregate[k] = aggregate.get(k, 0)+v

    def snapshot(self):
        values = dict(self.values)
        values['logical_bytes'] = values['program_upload_bytes']+values['policy_broadcast_bytes']
        values['work_rows'] = sum(values[k] for k in ('sampling_actor_rows', 'actor_forward_rows',
                                                     'actor_backward_rows', 'critic_forward_rows',
                                                     'critic_backward_rows'))
        return values


def module_bytes(module):
    return qualified.tensor_bytes(module)


def actor_bytes(actors):
    return sum(module_bytes(a.actor) for a in actors)


def actor_parameter_count(actors):
    return sum(p.numel() for a in actors for p in a.actor.parameters())


def flat_data(data):
    episodes, horizon = data['rewards'].shape
    count = episodes*horizon
    return dict(obs=[x.reshape(count, -1) for x in data['obs']],
                actions=[x.reshape(count, -1) for x in data['actions']],
                # HARL's Gaussian PPO expects a log likelihood PER coordinate,
                # unlike the scalar joint likelihood used by bridge gradients.
                logs=[x.reshape(count, -1) for x in data['logs']],
                states=data['states'].reshape(count, -1), rewards=data['rewards'])


def snapshot(actors, critic, normalizer, charges, event):
    clones = []
    for actor in actors:
        clone = copy.copy(actor)
        clone.actor = copy.deepcopy(actor.actor)
        clone.actor_optimizer = None
        clones.append(clone)
    cc = copy.copy(critic)
    cc.critic = copy.deepcopy(critic.critic)
    cc.critic_optimizer = None
    result = clones, cc, copy.deepcopy(normalizer)
    charges.add(event, copy_tensor_bytes=actor_bytes(actors)+module_bytes(critic.critic)+module_bytes(normalizer))
    return result


class Sampler:
    """One CPU worker, vectorized actor inference across independent episodes.

A bridge coin selects the complete team for each entire episode. The explicit
logical transport model puts gradients/critics on the server and rollout data
on workers: programs upload once, policy snapshots broadcast once per cohort.
There is no fictitious transmitted gradient for a server-computed backward.
"""
    def __init__(self, task, horizon, seed, task_index, charges, vector_width=16):
        self.task, self.horizon, self.seed = task, int(horizon), int(seed)
        self.task_index, self.charges = int(task_index), charges
        self.vector_width = int(vector_width)
        if self.horizon < 1 or self.vector_width < 1:
            raise ValueError('Invalid complete-episode shape')
        self.pools = {}
        self.identities = set()

    def close(self):
        for envs in self.pools.values():
            for env in envs:
                env.close()
        self.pools.clear()

    def collect(self, actors, ordinal, role, count, *, source=None, source_fraction=.5,
                purpose='training', deterministic=False):
        if purpose not in ('training', 'benchmark', 'frozen_policy') or count < 1:
            raise ValueError('Invalid independent data role')
        if source is not None and (purpose != 'training' or not 0 < source_fraction < 1):
            raise ValueError('Only training can use a complete-policy bridge')
        agents = len(actors)
        # Server -> worker policy payload, including BOTH policies for a bridge.
        policy_payload = actor_bytes(actors)+(actor_bytes(source) if source is not None else 0)
        self.charges.add(f'{purpose}:policy_broadcast', policy_broadcast_bytes=policy_payload,
                         messages=agents*(2 if source is not None else 1))
        rows = []
        for start in range(0, count, self.vector_width):
            amount = min(self.vector_width, count-start)
            if amount not in self.pools:
                self.pools[amount] = [qualified.make_env(self.task, self.horizon) for _ in range(amount)]
            envs = self.pools[amount]
            keys = [qualified.base.seed_for(self.seed, self.task_index, ordinal, role, e)
                    for e in range(start, start+amount)]
            if len(set(keys)) != amount or self.identities.intersection(keys):
                raise AssertionError('Episode role collision or data reuse')
            self.identities.update(keys)
            observation = [env.reset(seed=key)[0] for env, key in zip(envs, keys)]
            names = list(envs[0].possible_agents)
            if len(names) != agents:
                raise AssertionError('Agent catalogue differs from policy catalogue')
            rngs = [np.random.default_rng(qualified.base.seed_for(self.seed, self.task_index,
                                                                ordinal, role+100000, e))
                    for e in range(start, start+amount)]
            # A separate substream for mixture coins, not selection/action draws.
            modes = [int(source is not None and np.random.default_rng(
                qualified.base.seed_for(self.seed, self.task_index, ordinal, role+200000, e)
            ).random() < source_fraction) for e in range(start, start+amount)]
            local = [dict(obs=[[] for _ in actors], actions=[[] for _ in actors],
                          logs=[[] for _ in actors], states=[], rewards=[], identity=keys[e], mode=modes[e])
                     for e in range(amount)]
            for _ in range(self.horizon):
                chosen = [dict() for _ in range(amount)]
                for e, env in enumerate(envs):
                    if purpose == 'training':
                        local[e]['states'].append(np.array(env.state(), dtype=np.float32, copy=True))
                for mode in (0, 1):
                    indices = [e for e, x in enumerate(modes) if x == mode]
                    if not indices:
                        continue
                    team = actors if mode == 0 else source
                    if team is None:
                        raise AssertionError('Invalid policy-mixture mode')
                    for i, actor in enumerate(team):
                        obs = np.stack([observation[e][names[i]] for e in indices]).astype(np.float32)
                        with torch.no_grad():
                            dist = qualified.distribution(actor, obs)
                            discrete = actor.actor.act.action_type == 'Discrete'
                            for j, e in enumerate(indices):
                                if discrete:
                                    prob = dist.probs[j].cpu().numpy()
                                    latent = int(prob.argmax()) if deterministic else min(
                                        int(np.searchsorted(prob.cumsum(), rngs[e].random())), len(prob)-1)
                                    action, logp = latent, np.asarray([np.log(prob[latent])])
                                else:
                                    mean, std = dist.mean[j].cpu().numpy(), dist.stddev[j].cpu().numpy()
                                    latent = mean if deterministic else mean+std*rngs[e].normal(size=mean.shape)
                                    action = np.tanh(latent)
                                    logp = torch.distributions.Normal(dist.mean[j], dist.stddev[j]).log_prob(
                                        torch.as_tensor(latent, dtype=torch.float64)).cpu().numpy()
                                chosen[e][names[i]] = action
                                if purpose == 'training':
                                    local[e]['obs'][i].append(obs[j].copy())
                                    local[e]['actions'][i].append(np.array(latent, copy=True))
                                    local[e]['logs'][i].append(logp)
                        self.charges.add(f'{purpose}:actor_sampling', sampling_actor_rows=len(indices))
                following = []
                for e, env in enumerate(envs):
                    next_obs, rewards, term, trunc, _ = env.step(chosen[e])
                    self.charges.add(f'{purpose}:native_step', **{'native_'+purpose: 1})
                    local[e]['rewards'].append(float(np.mean(list(rewards.values()))))
                    if len(local[e]['rewards']) < self.horizon and (any(term.values()) or any(trunc.values())):
                        raise AssertionError('Unexpected early termination: do not change registered horizon')
                    following.append(next_obs)
                observation = following
            rows.extend(local)
        reward = np.stack([r['rewards'] for r in rows]).astype(np.float64)
        if not np.isfinite(reward).all():
            raise FloatingPointError('Nonfinite native reward')
        if purpose != 'training':
            self.charges.add(f'{purpose}:return_upload', program_upload_bytes=8*count, messages=1)
            return dict(returns=reward.sum(-1), identities=[r['identity'] for r in rows],
                        controller_eligible=False, native_transitions=count*self.horizon)
        data = dict(obs=[np.stack([r['obs'][i] for r in rows]).astype(np.float32) for i in range(agents)],
                    actions=[np.stack([r['actions'][i] for r in rows]) for i in range(agents)],
                    logs=[np.stack([r['logs'][i] for r in rows]).astype(np.float64) for i in range(agents)],
                    states=np.stack([r['states'] for r in rows]).astype(np.float32), rewards=reward,
                    identities=[r['identity'] for r in rows], modes=[r['mode'] for r in rows])
        self.charges.add('training:program_upload', program_upload_bytes=qualified.payload_bytes(data), messages=agents+1)
        return data


def values(actors, critic, normalizer, data, charges, event):
    episodes, horizon = data['rewards'].shape
    count = episodes*horizon
    states = data['states'].reshape(count, -1)
    hidden = actors[0].actor.hidden_sizes[-1]
    with torch.no_grad():
        raw, _ = critic.get_values(states, np.zeros((count, 1, hidden), dtype=np.float32),
                                   np.ones((count, 1), dtype=np.float32))
        result = normalizer.denormalize(raw).reshape(episodes, horizon)
    charges.add(event, critic_forward_rows=count)
    return result


def source_gradients(actors, critic, normalizer, data, gamma, charges):
    gradients, logs = qualified.episode_gradients(actors, critic, normalizer, data, gamma)
    episodes, horizon = data['rewards'].shape
    charges.add('source:on_policy_mc_pg', actor_forward_rows=len(actors)*episodes*horizon,
                actor_backward_rows=len(actors)*episodes*horizon, critic_forward_rows=episodes*horizon,
                gradient_storage_bytes=gradients.nbytes)
    return gradients, logs


def bridge_gradients(actors, source_actors, critic, normalizer, source_critic, source_normalizer,
                     data, gamma, charges, source_fraction=.5):
    target_values = values(actors, critic, normalizer, data, charges, 'bridge:current_baseline')
    old_values = values(source_actors, source_critic, source_normalizer, data, charges, 'bridge:source_baseline')
    paired = paired_episode_gradients(actors, source_actors, target_values, old_values,
                                      data, gamma, qualified.distribution, source_fraction)
    charges.add('bridge:prefix_pg', actor_forward_rows=paired['actor_forward_rows'],
                actor_backward_rows=paired['actor_backward_rows'],
                gradient_storage_bytes=paired['target'].nbytes+paired['control'].nbytes,
                prefix_scalar_entries=paired['prefix_scalar_entries'])
    return paired


def fit_critic(actors, critic, normalizer, data, p, charges, *, target=None):
    flat = flat_data(data)
    count = len(flat['states'])
    if target is None:
        qualified.base.add_advantages(flat, critic, normalizer, p)
    else:
        hidden = actors[0].actor.hidden_sizes[-1]
        with torch.no_grad():
            raw, _ = critic.get_values(flat['states'], np.zeros((count, 1, hidden), np.float32),
                                       np.ones((count, 1), np.float32))
        flat['raw_values'] = raw.cpu().numpy()
        flat['returns'] = np.asarray(target, dtype=np.float32).reshape(count, 1)
    charges.add('critic:prepare_targets', critic_forward_rows=count)
    qualified.base.train_critic(critic, normalizer, flat, p)
    epochs = p['critic_optimizer_steps_per_block']
    charges.add('critic:fit', critic_forward_rows=epochs*count, critic_backward_rows=epochs*count,
                critic_adam_calls=epochs)


def semantic_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
