"""Exact prefix-likelihood bridge PG; not a claimed new DR algorithm.

One complete-policy mixture coin per EPISODE is mandatory. Policy and frozen
state-baseline snapshots precede all data. This module only evaluates supplied
histories and never collects native data or chooses an action from its scores.
"""
import numpy as np
import torch


def prefix_weights(target_joint_logs, source_joint_logs, source_fraction=.5):
    if not 0 < source_fraction < 1:
        raise ValueError('Two positive mixture masses required')
    target = torch.cumsum(target_joint_logs.detach().to(torch.float64), dim=-1)
    source = torch.cumsum(source_joint_logs.detach().to(torch.float64), dim=-1)
    denominator = torch.logaddexp(target+np.log1p(-source_fraction), source+np.log(source_fraction))
    wt, ws = torch.exp(target-denominator), torch.exp(source-denominator)
    if (not bool(torch.isfinite(wt).all()) or not bool(torch.isfinite(ws).all()) or
            bool((wt > 1/(1-source_fraction)+1e-10).any()) or
            bool((ws > 1/source_fraction+1e-10).any())):
        raise FloatingPointError('Invalid defensive prefix weights')
    return wt, ws


def prefix_objective(joint_logs, rewards, baseline, weights, gamma):
    """Autodiff scalar whose gradient is the unbiased finite-episode PG.

Reward term: sum_t gamma^t w_t R_t sum_{k<=t} score_k.
Baseline term: -sum_t gamma^t w_t b(s_t) score_t.
Weights, observed rewards and pre-data baseline are detached coefficients.
Reward-to-go times only w_k is NOT this estimator and is generally biased.
"""
    if not 0 < gamma <= 1:
        raise ValueError('Invalid discount')
    rewards = torch.as_tensor(rewards, dtype=torch.float64, device=joint_logs.device).detach()
    baseline = torch.as_tensor(baseline, dtype=torch.float64, device=joint_logs.device).detach()
    weights = torch.as_tensor(weights, dtype=torch.float64, device=joint_logs.device).detach()
    if not (joint_logs.shape == rewards.shape == baseline.shape == weights.shape):
        raise ValueError('One finite history per gradient required')
    if joint_logs.ndim != 1 or any(not bool(torch.isfinite(x).all()) for x in (joint_logs, rewards, baseline, weights)):
        raise FloatingPointError('Invalid history')
    discount = gamma**torch.arange(len(joint_logs), dtype=torch.float64, device=joint_logs.device)
    return (discount*weights*(rewards*torch.cumsum(joint_logs, dim=-1)-baseline*joint_logs)).sum()


def paired_episode_gradients(target_actors, source_actors, target_values, source_values,
                             data, gamma, distribution, source_fraction=.5):
    """Complete actual neural PG primitive; source and target share geometry.

Both joint likelihoods include all agents. Continuous actions are the original
tanh-Gaussian latent draws, whose Jacobians cancel in the likelihood ratio.
Only two per-episode backward graphs are constructed; no H-fold VJP expansion.
"""
    count, horizon = data['rewards'].shape
    if len(target_actors) != len(source_actors):
        raise ValueError('Joint policies must have matching actor catalogues')
    targets, controls, overlaps = [], [], []
    for episode in range(count):
        logs = []
        for actors in (target_actors, source_actors):
            per_actor = []
            for i, actor in enumerate(actors):
                dist = distribution(actor, data['obs'][i][episode])
                dtype = torch.int64 if actor.actor.act.action_type == 'Discrete' else torch.float64
                actions = torch.as_tensor(data['actions'][i][episode], dtype=dtype)
                local = dist.log_prob(actions)
                per_actor.append(local if local.ndim == 1 else local.sum(-1))
            logs.append(sum(per_actor))
        wt, ws = prefix_weights(*logs, source_fraction=source_fraction)
        for actors, joint, baseline, weight, destination in (
                (target_actors, logs[0], target_values[episode], wt, targets),
                (source_actors, logs[1], source_values[episode], ws, controls)):
            parameters = [p for actor in actors for p in actor.actor.parameters()]
            objective = prefix_objective(joint, data['rewards'][episode], baseline, weight, gamma)
            gradients = torch.autograd.grad(objective, parameters)
            destination.append(np.concatenate([g.detach().cpu().numpy().reshape(-1).astype(float) for g in gradients]))
        overlaps.append((wt*ws).detach().cpu().numpy())
    result = dict(target=np.stack(targets), control=np.stack(controls),
                  prefix_overlap=np.stack(overlaps), backward_passes=2*count,
                  actor_forward_rows=2*count*horizon*len(target_actors),
                  actor_backward_rows=2*count*horizon*len(target_actors),
                  prefix_scalar_entries=2*count*horizon,
                  native_transitions_collected=0, performance_evidence=False)
    if not np.isfinite(result['target']).all() or not np.isfinite(result['control']).all():
        raise FloatingPointError('Nonfinite paired prefix gradients')
    return result
