"""Real MPE trajectories and observable MAPPO response diagnostics.

q counts rollout streams, not agents in a game. All complete trajectory
generation is charged. No validation label is a team-return certificate.
"""

import copy
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ICML-2027/runtime/pettingzoo_1_24_3"))
sys.path.insert(0, str(ROOT / "tmp/HARL"))


def make_env(task, cycles=25):
    module = importlib.import_module("pettingzoo.mpe." + task["scenario"])
    kwargs = {"max_cycles": cycles, "continuous_actions": task["continuous_actions"]}
    if task["scenario"] == "simple_spread_v3":
        kwargs["N"] = 3
    return module.parallel_env(**kwargs)


def discounted_returns(rewards, discount):
    result = np.zeros_like(rewards, dtype=np.float64)
    total = 0.0
    for index in range(len(rewards)-1, -1, -1):
        total = rewards[index] + discount * total
        result[index] = total
    return result.astype(np.float32)


def episode_seed(seed, stage, role, episode, worker, shared_probability):
    """Shared/independent mixture of entire trajectories preserves marginals."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, stage, role, episode, 1]))
    common = rng.random() < shared_probability
    sequence = np.random.SeedSequence([seed, stage, role, episode, 2, 0 if common else worker+1])
    return int(sequence.generate_state(1)[0] % (2**31-1))


def initialize(task, seed):
    import gym
    import yaml
    from harl.algorithms.actors.mappo import MAPPO
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.valuenorm import ValueNorm

    env = make_env(task)
    obs, _ = env.reset(seed=seed)
    agent = env.agents[0]
    obs_dim, state_dim = len(obs[agent]), len(env.state())
    action = env.action_space(agent)
    space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
    central = gym.spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
    action_space = (gym.spaces.Box(action.low, action.high, dtype=np.float32)
                    if task["continuous_actions"] else gym.spaces.Discrete(action.n))
    config = yaml.safe_load((ROOT / "tmp/HARL/harl/configs/algos_cfgs/mappo.yaml").read_text())
    args = {**config["model"], **config["algo"], "activation_func": "tanh"}
    torch.manual_seed(seed)
    actor = MAPPO(args, space, action_space, torch.device("cpu"))
    critic = VCritic(args, central, torch.device("cpu"))
    normalizer = ValueNorm(1, device=torch.device("cpu"))
    env.close()
    return actor, critic, normalizer


def collect(task, actor, critic, normalizer, *, seed, stage, role,
            workers, episodes, coupling, cycles=25):
    """On-policy, fixed-horizon complete episodes; latent actions are retained.

    Continuous actions are sampled from the HARL Gaussian and clipped only by
    the execution map. PPO log probabilities use the sampled latent action.
    """
    buffers, identities = [], []
    agents_count = None
    for worker in range(workers):
        worker_batches = []
        for episode in range(episodes):
            key = episode_seed(seed, stage, role, episode, worker, coupling)
            env = make_env(task, cycles)
            obs, _ = env.reset(seed=key)
            names = list(env.agents)
            agents_count = len(names)
            generator = torch.Generator(device="cpu").manual_seed(key)
            states_a = np.zeros((agents_count, 1, 128), np.float32)
            states_c = np.zeros((1, 1, 128), np.float32)
            masks_a = np.ones((agents_count, 1), np.float32)
            masks_c = np.ones((1, 1), np.float32)
            rows = {name: [] for name in ("obs", "state", "actions", "logs", "values", "rewards")}
            for tick in range(cycles):
                observations = np.stack([obs[name] for name in names]).astype(np.float32)
                central = np.asarray(env.state(), np.float32)[None]
                with torch.no_grad():
                    features = actor.actor.base(torch.as_tensor(observations))
                    distribution = actor.actor.act.action_out(features)
                    if task["continuous_actions"]:
                        action = distribution.mean + distribution.stddev * torch.randn(distribution.mean.shape, generator=generator)
                    else:
                        action = torch.multinomial(distribution.probs, 1, generator=generator)
                    log_prob = distribution.log_probs(action)
                    values = normalizer.denormalize(critic.get_values(central, states_c, masks_c)[0]).reshape(-1)[0]
                latent = action.numpy()
                if task["continuous_actions"]:
                    executed = {name: np.clip(latent[i], env.action_space(name).low, env.action_space(name).high)
                                for i, name in enumerate(names)}
                else:
                    executed = {name: int(latent[i, 0]) for i, name in enumerate(names)}
                following, reward, terminated, truncated, _ = env.step(executed)
                rows["obs"].append(observations)
                rows["state"].append(central[0])
                rows["actions"].append(latent)
                rows["logs"].append(log_prob.numpy())
                rows["values"].append(values)
                rows["rewards"].append(sum(reward.values()))
                obs = following
                if tick < cycles-1 and (any(terminated.values()) or any(truncated.values())):
                    raise AssertionError("unexpected early episode end")
            env.close()
            data = {k: np.asarray(v, np.float32) for k, v in rows.items()}
            mc = discounted_returns(data["rewards"], .99)
            following_values = np.r_[data["values"][1:], 0.].astype(np.float32)
            deltas = data["rewards"] + .99 * following_values - data["values"]
            gae = discounted_returns(deltas, .99 * .95)
            data.update({"mc": mc[:, None], "returns": (gae+data["values"])[:, None],
                         "advantages": np.repeat(gae[:, None, None], agents_count, axis=1)})
            worker_batches.append(data)
            identities.append({"seed": key, "stage": stage, "role": role,
                               "episode": episode, "worker": worker})
        buffers.append({k: np.concatenate([b[k] for b in worker_batches]) for k in worker_batches[0]})
    ledger = {"joint_environment_transitions": workers*episodes*cycles,
              "individual_agent_transitions": workers*episodes*cycles*agents_count,
              "agents_per_game": agents_count, "streams": workers,
              "trajectory_identities": identities}
    return buffers, ledger


def join(buffers, q):
    return {k: np.concatenate([b[k] for b in buffers[:q]]) for k in buffers[0]}


def actor_batch(data):
    obs = data["obs"].reshape(-1, data["obs"].shape[-1])
    count = len(obs)
    advantages = data["advantages"].reshape(-1, 1)
    advantages = (advantages-advantages.mean())/(advantages.std()+1e-5)
    return (obs, np.zeros((count, 1, 128), np.float32),
            data["actions"].reshape(count, -1), np.ones((count, 1), np.float32),
            np.ones((count, 1), np.float32), data["logs"].reshape(count, -1),
            advantages.astype(np.float32), None)


def critic_batch(data, critic):
    n = len(data["state"])
    states = np.zeros((n, 1, 128), np.float32)
    masks = np.ones((n, 1), np.float32)
    with torch.no_grad():
        old = critic.get_values(data["state"], states, masks)[0].numpy()
    return data["state"], states, old, data["returns"], masks


def training_update(actor, critic, normalizer, data, steps):
    a_batch, c_batch = actor_batch(data), critic_batch(data, critic)
    for _ in range(steps):
        actor.update(a_batch)
        critic.update(c_batch, normalizer)


def critic_coordinates(critic, normalizer):
    mean, variance = normalizer.running_mean_var()
    return [p.detach().clone() for p in critic.critic.parameters()] + [mean.detach().clone(), variance.sqrt().detach().clone()]


def actor_coordinates(actor):
    return [p.detach().clone() for p in actor.actor.parameters()]


def objective(actor, critic, data, kind):
    """A differentiable empirical potential, not the policy's true return."""
    if kind == "critic":
        names = [name for name, _ in critic.critic.named_parameters()]
        state = torch.as_tensor(data["state"])
        target = torch.as_tensor(data["mc"])
        zeros = torch.zeros((len(state), 1, 128))
        masks = torch.ones((len(state), 1))

        def value(coordinates):
            pred = torch.func.functional_call(critic.critic, dict(zip(names, coordinates[:-2])),
                                              (state, zeros, masks))[0]
            denormalized = pred*coordinates[-1]+coordinates[-2]
            return (denormalized-target).square().mean()
        return value
    names = [name for name, _ in actor.actor.named_parameters()]
    sample = actor_batch(data)
    obs, states, actions, masks, active, old_logs, advantages, _ = sample
    obs, actions = torch.as_tensor(obs), torch.as_tensor(actions)
    old_logs, advantages = torch.as_tensor(old_logs), torch.as_tensor(advantages)

    def value(coordinates):
        params = dict(zip(names, coordinates))
        base = {key.removeprefix("base."): v for key, v in params.items() if key.startswith("base.")}
        act = {key.removeprefix("act.action_out."): v for key, v in params.items() if key.startswith("act.action_out.")}
        features = torch.func.functional_call(actor.actor.base, base, (obs,))
        distribution = torch.func.functional_call(actor.actor.act.action_out, act, (features,))
        logs = distribution.log_probs(actions)
        ratio = torch.exp((logs-old_logs).sum(-1, keepdim=True))
        return -(ratio*advantages).mean()-.01*distribution.entropy().mean()
    return value


class QuadraticResponse:
    """Derivative-based prediction at a frozen sensor anchor; no fitted curve."""

    def __init__(self, function, coordinates):
        self.function = function
        self.anchor = [p.detach().clone().requires_grad_(True) for p in coordinates]
        self.base = function(self.anchor)
        self.gradient = torch.autograd.grad(self.base, self.anchor, create_graph=True)
        self.hvp_calls = 0

    def forecast(self, coordinates, *, freeze_last=0):
        delta = [p.detach()-p0.detach() for p, p0 in zip(coordinates, self.anchor)]
        if freeze_last:
            delta[-freeze_last:] = [torch.zeros_like(d) for d in delta[-freeze_last:]]
        first = sum((g*d).sum() for g, d in zip(self.gradient, delta))
        hvp = torch.autograd.grad(first, self.anchor, retain_graph=True, allow_unused=True)
        self.hvp_calls += 1
        second = .5*sum((h*d).sum() for h, d in zip(hvp, delta) if h is not None)
        return float((first+second).detach()), float(first.detach())


def model_hash(actor, critic, normalizer):
    values = actor_coordinates(actor)+critic_coordinates(critic, normalizer)
    return hashlib.sha256(b"".join(p.numpy().tobytes() for p in values)).hexdigest()


def clone_models(actor, critic, normalizer):
    return copy.deepcopy(actor), copy.deepcopy(critic), copy.deepcopy(normalizer)
