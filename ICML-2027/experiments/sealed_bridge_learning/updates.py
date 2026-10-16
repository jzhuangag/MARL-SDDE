"""Actual server-owned fresh/delayed PPO and joint V-trace learning adapters.

Adapters are labelled adaptations, not official reproductions. No current-bank
return maximum, imported source Adam moments or repeated steps on a stored
gradient are used. Native calls belong only to a later separately locked driver.
"""
import numpy as np
import torch
from sealed_bridge_learning.native import flat_data, qualified, values


def ppo_update(actors, critic, normalizer, data, p, charges, *, sequential=True, kl_cap=None):
    """Every epoch recomputes PPO losses and gradients at server parameters.

Joint source/current log diagnostics precede any commit. KL filtering cannot
refund the already collected/transported source program. No evaluation is read.
"""
    flat = flat_data(data)
    count, agents = len(flat['states']), len(actors)
    qualified.base.add_advantages(flat, critic, normalizer, p)
    charges.add('ppo:prepare_advantages', critic_forward_rows=count)
    before = [qualified.base.actor_logs(a, flat, i, p) for i, a in enumerate(actors)]
    charges.add('ppo:precommit_logs', actor_forward_rows=agents*count)
    joint_log_ratio = sum((before[i]-flat['logs'][i]).sum(-1, keepdims=True) for i in range(agents))
    # The source-sampled k3 KL estimate is nonnegative without using returns.
    ratio = np.exp(joint_log_ratio)
    diagnostic = float(np.mean(ratio-1-joint_log_ratio))
    if not np.isfinite(diagnostic):
        raise FloatingPointError('Nonfinite joint KL diagnostic')
    if kl_cap is not None and diagnostic > kl_cap:
        return dict(accepted=False, joint_kl_diagnostic=diagnostic, source_optimizer_imported=False,
                    already_paid_source_refunded=False)
    factor = np.ones_like(flat['advantages'])
    for i, actor in enumerate(actors):
        qualified.base.train_actor(actor, flat, i, factor, p)
        epochs = p['actor_optimizer_steps_per_packet']
        charges.add('ppo:actual_epochs', actor_forward_rows=epochs*count, actor_backward_rows=epochs*count,
                    actor_adam_calls=epochs)
        if sequential:
            after = qualified.base.actor_logs(actor, flat, i, p)
            charges.add('happo:sequential_factor', actor_forward_rows=count)
            factor *= np.exp((after-before[i]).sum(-1, keepdims=True))
            if not np.isfinite(factor).all():
                raise FloatingPointError('Nonfinite sequential advantage factor')
    return dict(accepted=True, joint_kl_diagnostic=diagnostic, source_optimizer_imported=False,
                actor_update='server_recomputed_happo' if sequential else 'server_recomputed_mappo',
                policy_evaluation_used=False)


def joint_vtrace_update(actors, critic, normalizer, data, p, charges, rho_cap=1., c_cap=1.):
    """Joint-MA-Trace-style native adaptation, one genuine current PG/actor.

All teammate likelihoods enter the joint ratio. This is not the HAPPO PPO loss
renamed V-trace. Target returns and actor advantages follow the clipped joint
V-trace recursion; episode terminal value is0 for the finite-episode objective.
"""
    episodes, horizon = data['rewards'].shape
    flat = flat_data(data)
    count, agents = episodes*horizon, len(actors)
    baseline = np.asarray(values(actors, critic, normalizer, data, charges, 'vtrace:current_values'))
    joint = np.zeros((episodes, horizon), dtype=np.float64)
    for i, actor in enumerate(actors):
        logs = qualified.base.actor_logs(actor, flat, i, p).reshape(episodes, horizon, -1).sum(-1)
        joint += logs-data['logs'][i].reshape(episodes, horizon, -1).sum(-1)
    charges.add('vtrace:joint_ratios', actor_forward_rows=agents*count)
    ratio = np.exp(joint)
    if not np.isfinite(ratio).all():
        raise FloatingPointError('Nonfinite untruncated joint importance ratio')
    rho, c = np.minimum(ratio, rho_cap), np.minimum(ratio, c_cap)
    gamma = p['discount']
    following = np.concatenate([baseline[:, 1:], np.zeros((episodes, 1))], axis=1)
    td = rho*(data['rewards']+gamma*following-baseline)
    correction = np.zeros(episodes)
    targets = np.empty_like(baseline)
    for t in range(horizon-1, -1, -1):
        correction = td[:, t]+gamma*c[:, t]*correction
        targets[:, t] = baseline[:, t]+correction
    next_target = np.concatenate([targets[:, 1:], np.zeros((episodes, 1))], axis=1)
    advantage = rho*(data['rewards']+gamma*next_target-baseline)
    coefficient = torch.as_tensor(advantage.reshape(-1), dtype=torch.float64)
    for i, actor in enumerate(actors):
        dist = qualified.distribution(actor, flat['obs'][i])
        dtype = torch.int64 if actor.actor.act.action_type == 'Discrete' else torch.float64
        ac = torch.as_tensor(flat['actions'][i], dtype=dtype)
        if dtype == torch.int64:
            ac = ac.reshape(-1)
        logs = dist.log_prob(ac)
        if logs.ndim > 1:
            logs = logs.sum(-1)
        loss = -(logs*coefficient).mean()
        actor.actor_optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.actor.parameters(), actor.max_grad_norm)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite genuine V-trace update')
        actor.actor_optimizer.step()
        charges.add('vtrace:server_actor_pg', actor_forward_rows=count, actor_backward_rows=count,
                    actor_adam_calls=1)
    return dict(critic_targets=targets, joint_rho_clipped_at=rho_cap, joint_c_clipped_at=c_cap,
                actor_update='native_joint_vtrace_adaptation', official_reproduction=False,
                source_optimizer_imported=False, evaluation_used=False)


class PredictableActorScale:
    """Past-only global RMS, preserving conditional gradient direction.

Initialize from a separately charged calibration cohort with NO actor commit.
Each subsequent step uses only prior gradient energy. No claim that this
normalizer measures a neural smoothness upper bound. The finite-time smoothness
inequality retains its actual alpha and requires the declared L condition.
"""
    def __init__(self, initial_gradients, parameter_count, base_rate, rate_cap, epsilon=1e-12):
        if base_rate <= 0 or rate_cap <= 0 or parameter_count <= 0 or epsilon <= 0:
            raise ValueError('Invalid predictable actor scale')
        gradients = np.asarray(initial_gradients, dtype=float)
        if gradients.ndim != 2 or gradients.shape[1] != parameter_count or not np.isfinite(gradients).all():
            raise ValueError('Invalid charged calibration geometry')
        self.energy = float(np.mean(np.square(gradients)))
        self.updates = 1
        self.base_rate, self.rate_cap, self.epsilon = float(base_rate), float(rate_cap), float(epsilon)
        self.parameter_count = int(parameter_count)
        self.pending = None

    def begin(self):
        if self.pending is not None:
            raise RuntimeError('Actor scale transaction not closed')
        alpha = min(self.rate_cap, self.base_rate/np.sqrt(self.epsilon+self.energy))
        self.pending = alpha
        return dict(alpha=float(alpha), past_gradient_energy=self.energy,
                    before_source_and_bridge=True, neural_smoothness_certificate=False)

    def close(self, gradient):
        if self.pending is None:
            raise RuntimeError('Missing predictable actor step')
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != (self.parameter_count,) or not np.isfinite(gradient).all():
            raise ValueError('Invalid committed gradient')
        self.updates += 1
        self.energy += (float(np.mean(np.square(gradient)))-self.energy)/self.updates
        alpha = self.pending
        self.pending = None
        return dict(alpha=float(alpha), future_gradient_energy=self.energy)
