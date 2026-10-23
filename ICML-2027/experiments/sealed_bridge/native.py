"""Complete feedforward native-MPE bridge loop, not a parameter-ray selector.

The source cohorts are delayed DATA programs. Gradients are recomputed at the
immutable source policy on receipt; server updates never import worker moments.
All source episodes, source-policy/value copies and dual bridge gradients count.
This backend emulates transport clocks locally, not measured parallel speedup.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import happo_batch_cpu as base
FARAMA = base.ROOT / 'ICML-2027/runtime/sealed_bridge_farama_20261016'
# This separate runtime never replaces the old pinned directories/environments.
sys.path.insert(0, str(FARAMA))
from sealed_bridge.core import (Option, RiskQueueController, SealedAnchor,
                               bridge_weights, risk_feedback, transported_direction)


def tensor_bytes(module):
    return sum(p.numel()*p.element_size() for p in module.state_dict().values())


def policy_hash(actors):
    h = hashlib.sha256()
    for actor in actors:
        for key, p in actor.actor.state_dict().items():
            h.update(key.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def array_hash(data):
    h = hashlib.sha256()
    for key in ('obs', 'actions', 'logs', 'states', 'rewards'):
        for x in data[key] if isinstance(data[key], list) else [data[key]]:
            x = np.asarray(x)
            h.update(str((key, x.shape, x.dtype.str)).encode()); h.update(x.tobytes())
    return h.hexdigest()


def probabilities(actor, obs):
    distribution = actor.actor.act.action_out(actor.actor.base(torch.as_tensor(obs, dtype=torch.float32)))
    p = distribution.probs.to(torch.float64)
    p = p/p.sum(-1, keepdim=True)
    if not bool(torch.isfinite(p).all()) or not bool((p > 0).all()):
        raise FloatingPointError('Lost categorical support')
    return p


def distribution(actor, obs):
    raw = actor.actor.act.action_out(actor.actor.base(torch.as_tensor(obs, dtype=torch.float32)))
    if actor.actor.act.action_type == 'Discrete':
        prob=raw.probs.to(torch.float64); prob=prob/prob.sum(-1,keepdim=True)
        if not bool(torch.isfinite(prob).all()) or not bool((prob>0).all()):
            raise FloatingPointError('Lost categorical support')
        return torch.distributions.Categorical(probs=prob)
    return torch.distributions.Normal(raw.mean.to(torch.float64), raw.stddev.to(torch.float64))


POOLS = {}


def make_env(task, horizon):
    if task.get('family','mpe') == 'mpe':
        return base.make_env(task, horizon)
    from gymnasium_robotics import mamujoco_v1
    return mamujoco_v1.parallel_env(task['scenario'], task['partition'], agent_obsk=1,
                                   max_episode_steps=horizon)


def initialize(task, seed, p):
    if task.get('family','mpe') == 'mpe':
        return base.initialize(task, seed, p)
    import gym
    import yaml
    from harl.algorithms.actors.happo import HAPPO
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.valuenorm import ValueNorm
    env = make_env(task,p['episode_length']); obs,_=env.reset(seed=seed)
    cfg=yaml.safe_load((base.UPSTREAM/'harl/configs/algos_cfgs/happo.yaml').read_text())
    args={**cfg['model'],**cfg['algo'],'hidden_sizes':p['hidden_sizes'],'activation_func':p['activation']}
    actors=[]
    for i,name in enumerate(env.possible_agents):
        torch.manual_seed(seed+i)
        os=gym.spaces.Box(-np.inf,np.inf,shape=obs[name].shape,dtype=np.float32)
        ac=env.action_space(name)
        actors.append(HAPPO(args,os,gym.spaces.Box(ac.low,ac.high,dtype=np.float32)))
    torch.manual_seed(seed+100)
    critic=VCritic(args,gym.spaces.Box(-np.inf,np.inf,shape=env.state().shape,dtype=np.float32))
    env.close()
    return actors,critic,ValueNorm(1)


def collect(actors, task, seed, ordinal, role, count, horizon, *, source=None, source_fraction=.5):
    n = len(actors); rows = []
    for e in range(count):
        key = base.seed_for(seed, 0, ordinal, role, e)
        # Policy mixture coins are independent of environment/action randomness.
        rng = np.random.default_rng(base.seed_for(seed, 1, ordinal, role, e))
        mode = int(source is not None and rng.random() < source_fraction)
        policies = source if mode else actors
        pool_key=(task['name'],horizon)
        if pool_key not in POOLS:POOLS[pool_key]=make_env(task,horizon)
        env=POOLS[pool_key]
        obs, _ = env.reset(seed=key)
        names = list(env.possible_agents)
        oo, aa, ll = [[] for _ in actors], [[] for _ in actors], [[] for _ in actors]
        ss, rr = [], []
        for t in range(horizon):
            ss.append(np.array(env.state(), np.float32, copy=True)); actions = {}
            for i, actor in enumerate(policies):
                o = np.asarray(obs[names[i]], np.float32)
                with torch.no_grad():
                    d = distribution(actor,o[None])
                    if actor.actor.act.action_type == 'Discrete':
                        prob=d.probs[0].numpy()
                        latent=min(int(np.searchsorted(np.cumsum(prob),rng.random())),len(prob)-1)
                        action=latent; logp=float(np.log(prob[latent]))
                    else:
                        latent=d.mean[0].numpy()+d.stddev[0].numpy()*rng.normal(size=d.mean.shape[-1])
                        action=np.tanh(latent)
                        logp=float(d.log_prob(torch.as_tensor(latent)[None]).sum())
                actions[names[i]] = action
                oo[i].append(o.copy()); aa[i].append(latent); ll[i].append(logp)
            obs, rewards, term, trunc, _ = env.step(actions)
            rr.append(float(np.mean(list(rewards.values()))))
            if t < horizon-1 and (any(term.values()) or any(trunc.values())):
                raise AssertionError('Unexpected early native episode termination')
        rows.append(dict(obs=[np.asarray(x) for x in oo], actions=[np.asarray(x) for x in aa],
                         logs=[np.asarray(x) for x in ll], states=np.asarray(ss), rewards=np.asarray(rr),
                         identity=key, mode=mode))
    return dict(obs=[np.stack([r['obs'][i] for r in rows]) for i in range(n)],
                actions=[np.stack([r['actions'][i] for r in rows]) for i in range(n)],
                logs=[np.stack([r['logs'][i] for r in rows]) for i in range(n)],
                states=np.stack([r['states'] for r in rows]), rewards=np.stack([r['rewards'] for r in rows]),
                identities=[r['identity'] for r in rows], modes=[r['mode'] for r in rows])


def flat_data(data):
    m,h=data['rewards'].shape
    return dict(obs=[x.reshape(-1, x.shape[-1]) for x in data['obs']],
                actions=[x.reshape(m*h, -1) for x in data['actions']],
                logs=[x.reshape(-1, 1) for x in data['logs']],
                states=data['states'].reshape(-1, data['states'].shape[-1]),
                rewards=data['rewards'])


def episode_gradients(actors, critic, normalizer, data, gamma):
    """Actual Monte Carlo PG, no GAE normalization, PPO clipping or extra epochs."""
    m, h = data['rewards'].shape
    fd = flat_data(data)
    params = [p for a in actors for p in a.actor.parameters()]
    with torch.no_grad():
        states = fd['states']; z = np.zeros((m*h, 1, actors[0].actor.hidden_sizes[-1]), np.float32)
        values, _ = critic.get_values(states, z, np.ones((m*h, 1), np.float32))
        values = normalizer.denormalize(values).reshape(m, h)
    returns = np.zeros_like(data['rewards']); future = np.zeros(m)
    for t in range(h-1, -1, -1):
        future = data['rewards'][:, t] + gamma*future; returns[:, t] = future
    advantages = torch.as_tensor((returns-values)*gamma**np.arange(h)[None], dtype=torch.float64)
    gradients, logp = [], []
    for e in range(m):
        # Each backward contains exactly ONE episode. Taking m VJPs through a
        # dense m-episode graph would cost m^2 rows, not the advertised m rows.
        logs = []
        for i, actor in enumerate(actors):
            dist=distribution(actor,data['obs'][i][e])
            dtype=torch.int64 if actor.actor.act.action_type=='Discrete' else torch.float64
            actions=torch.as_tensor(data['actions'][i][e],dtype=dtype)
            log=dist.log_prob(actions)
            logs.append(log if log.ndim==1 else log.sum(-1))
        joint = sum(logs)
        objective = (joint*advantages[e]).sum()
        g = torch.autograd.grad(objective, params)
        gradients.append(np.concatenate([x.detach().numpy().reshape(-1).astype(float) for x in g]))
        logp.append(float(joint.detach().sum()))
    result = np.stack(gradients)
    logp = np.asarray(logp)
    if not np.isfinite(result).all() or not np.isfinite(logp).all():
        raise FloatingPointError('Nonfinite actual PG or trajectory log probability')
    return result, logp


def sgd_commit(actors, direction, alpha):
    if alpha <= 0 or not np.isfinite(direction).all():
        raise ValueError('Invalid SGD transaction')
    start = 0
    with torch.no_grad():
        for actor in actors:
            for p in actor.actor.parameters():
                end = start+p.numel()
                p.add_(torch.as_tensor(direction[start:end].reshape(p.shape), dtype=p.dtype), alpha=alpha)
                if not bool(torch.isfinite(p).all()):
                    raise FloatingPointError('Nonfinite committed parameter')
                start = end
    assert start == len(direction)


def payload_bytes(data):
    return sum(x.nbytes for k in ('obs','actions','logs','states','rewards')
               for x in (data[k] if isinstance(data[k], list) else [data[k]]))


def run_cell(task, seed, method, plan, budget):
    p = base.config(); p.update(episode_length=plan['horizon'],
                               critic_optimizer_steps_per_block=plan['critic_epochs'])
    actors, critic, normalizer = initialize(task, seed, p)
    n = len(actors); initial = policy_hash(actors)
    options = [Option(m, b) for m in plan['bridge_episodes'] for b in plan['betas']]
    controller = RiskQueueController(options, step=plan['meta_step'],
                                    variance_weight=plan['variance_weight'], targets=plan['resource_targets'])
    rng = np.random.default_rng(base.seed_for(seed, 9, 0, plan['controller_role'], 0))
    costs = dict(source_transitions=0, bridge_transitions=0, actor_sampling_rows=0,
                 gradient_forward_rows=0, gradient_backward_rows=0, critic_value_rows=0,
                 critic_backward_rows=0, actor_sgd_commits=0, actor_adam_calls=0,
                 critic_adam_calls=0, source_payload_bytes=0, bridge_payload_bytes=0,
                 policy_broadcast_bytes=0, baseline_broadcast_bytes=0, model_copy_bytes=0,
                 gradient_array_bytes=0, moment_scalar_entries=0, messages=0)
    jobs, receipts, train_ids = [], [], []
    version = 0; used = set(); launch_events = []; tick = 0
    h, count = plan['horizon'], plan['source_episodes']
    policy_size = sum(tensor_bytes(a.actor) for a in actors)
    critic_size = tensor_bytes(critic.critic) + tensor_bytes(normalizer)
    # Byte/work scales are shape-based constants, not outcome-derived normalizers.
    env = make_env(task, h); ob, _ = env.reset(seed=seed)
    obs_dim = sum(len(x) for x in ob.values()); state_dim = len(env.state()); env.close()
    action_dim=sum(1 if a.actor.act.action_type=='Discrete' else a.actor.act.action_out.fc_mean.out_features for a in actors)
    per_episode_bytes = h*(4*obs_dim + 8*action_dim + 8*n + state_dim*4 + 8)
    parameter_count = sum(p.numel() for a in actors for p in a.actor.parameters())
    scales = np.asarray([h*(count+max(plan['bridge_episodes'])),
                         per_episode_bytes*(count+max(plan['bridge_episodes']))+policy_size+critic_size])
    while len(used) < plan['source_cohorts']:
        budget()
        # Launch clock and service randomness do not depend on receipt decisions.
        if tick < plan['source_cohorts']:
            source = copy.deepcopy(actors); sc = copy.deepcopy(critic); sn = copy.deepcopy(normalizer)
            data = collect(source, task, seed, tick, plan['source_role'], count, h)
            train_ids.extend(data['identities'])
            for k in ('obs','actions','logs','states','rewards'):
                for x in data[k] if isinstance(data[k], list) else [data[k]]:
                    x.setflags(write=False)
            word = hashlib.sha256(f'{seed}:source:{tick}'.encode()).digest()
            delay = plan['delay_min'] + int.from_bytes(word[:8], 'big') % (plan['delay_max']-plan['delay_min']+1)
            job = dict(id=tick, due=tick+delay, version=version, policy=source, critic=sc, normalizer=sn,
                       data=data, data_hash=array_hash(data), source_hash=policy_hash(source))
            jobs.append(job)
            launch_events.append(dict(id=tick, launch_tick=tick, due=job['due'], source_version=version,
                                      source_hash=job['source_hash'], data_hash=job['data_hash'],
                                      native_transitions=count*h, source_payload_bytes=payload_bytes(data)))
            costs['source_transitions'] += count*h; costs['actor_sampling_rows'] += n*count*h
            costs['policy_broadcast_bytes'] += policy_size; costs['baseline_broadcast_bytes'] += critic_size
            costs['model_copy_bytes'] += policy_size+critic_size
            costs['source_payload_bytes'] += payload_bytes(data); costs['messages'] += 2*n+1
        ready = sorted((j for j in jobs if j['id'] not in used and j['due'] <= tick), key=lambda j:(j['due'],j['id']))
        for job in ready:
            budget(); before = policy_hash(actors)
            assert array_hash(job['data']) == job['data_hash'] and policy_hash(job['policy']) == job['source_hash']
            # Prospective costs include the paid source cohort, not only its receipt.
            prospective = np.asarray([[(count+o.episodes)*h,
                (count+o.episodes)*per_episode_bytes+policy_size+critic_size] for o in options])/scales
            if method == 'bridge_control':
                decision = controller.begin(prospective, rng, source_id=str(job['id']), current_hash=before)
            else:
                index = options.index(Option(plan['fixed_bridge_episodes'], plan['fixed_beta']))
                decision = dict(index=index, episodes=options[index].episodes, beta=options[index].beta,
                                source_id=str(job['id']), current_hash=before, before_anchor_and_bridge=True)
            # The choice was made without source returns/gradient and fresh bridge.
            sg, _ = episode_gradients(job['policy'], job['critic'], job['normalizer'], job['data'], plan['gamma'])
            anchor = SealedAnchor(str(job['id']), sg)
            ag = anchor.open_once(decision)
            m = decision['episodes']; beta = decision['beta']
            data = collect(actors, task, seed, job['id'], plan['bridge_role'], m, h,
                           source=job['policy'], source_fraction=plan['source_fraction'])
            train_ids.extend(data['identities'])
            tg, tl = episode_gradients(actors, critic, normalizer, data, plan['gamma'])
            cg, cl = episode_gradients(job['policy'], job['critic'], job['normalizer'], data, plan['gamma'])
            wt, wc = bridge_weights(tl, cl, plan['source_fraction'])
            f, c = wt[:, None]*tg, wc[:, None]*cg
            direction = transported_direction(f, c, ag, beta)
            risks = risk_feedback(f, c, ag, options)
            assert policy_hash(actors) == before
            sgd_commit(actors, direction, plan['actor_step'])
            version += 1
            details = controller.close(risks, prospective[decision['index']]) if method == 'bridge_control' else {}
            # Baseline training follows every actor-gradient use, never precedes it.
            fd = flat_data(data)
            base.add_advantages(fd, critic, normalizer, p)
            base.train_critic(critic, normalizer, fd, p)
            costs['bridge_transitions'] += m*h; costs['actor_sampling_rows'] += n*m*h
            costs['gradient_forward_rows'] += n*(count+2*m)*h
            costs['gradient_backward_rows'] += n*(count+2*m)*h
            costs['critic_value_rows'] += (count+3*m)*h
            costs['critic_backward_rows'] += plan['critic_epochs']*m*h
            costs['actor_sgd_commits'] += n
            costs['critic_adam_calls'] += plan['critic_epochs']
            costs['bridge_payload_bytes'] += payload_bytes(data)
            costs['gradient_array_bytes'] += sg.nbytes+tg.nbytes+cg.nbytes
            costs['moment_scalar_entries'] += len(options)*m*parameter_count
            costs['messages'] += n+1
            used.add(job['id'])
            receipts.append(dict(source_id=job['id'], tick=tick, source_version=job['version'],
                target_version=version-1, actual_version_age=version-1-job['version'],
                target_before=before, target_after=policy_hash(actors),
                source_hash=job['source_hash'], data_hash=job['data_hash'], anchor_hash=anchor.sha256,
                direction_hash=hashlib.sha256(direction.tobytes()).hexdigest(),
                source_ids=job['data']['identities'], bridge_ids=data['identities'],
                complete_episode_modes=data['modes'], decision=decision, control=details,
                target_weight_max=float(wt.max()), source_weight_max=float(wc.max()),
                source_cohort_consumed_once=anchor.consumed, source_cohort_reused=False,
                current_return_ranked=False, evaluation_used=False,
                actor_optimizer='server_sgd', ordinary_adam_theorem_claimed=False,
                exact_native_transitions=(count+m)*h,
                source_payload_bytes=payload_bytes(job['data']), bridge_payload_bytes=payload_bytes(data),
                costs=dict(costs)))
        tick += 1
        if tick > plan['source_cohorts']+plan['delay_max']+1:
            raise AssertionError('Undrained source service queue')
    assert len(set(train_ids)) == len(train_ids) and len(used) == len(jobs) == plan['source_cohorts']
    assert costs['source_transitions'] == plan['source_cohorts']*count*h
    assert costs['bridge_transitions'] == sum(r['decision']['episodes']*h for r in receipts)
    return dict(task=task['name'], seed=seed, method=method, initial_hash=initial,
                final_hash=policy_hash(actors), launches=launch_events, receipts=receipts, costs=costs,
                episode_ids=train_ids, all_source_cohorts_consumed=True, evaluations_collected=0,
                qualification_returns_recorded=False, performance_evidence=False,
                logical_transport_only=True, charge_scales=scales.tolist(),
                policy_parameter_bytes=policy_size, baseline_parameter_bytes=critic_size,
                per_episode_payload_bytes=per_episode_bytes)
