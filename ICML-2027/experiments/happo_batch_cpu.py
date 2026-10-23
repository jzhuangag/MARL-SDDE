"""Pinned HAPPO end-to-end CPU allocation screen with complete cost ledgers.

The outcome-aware branch measures a finite candidate-menu opportunity and
records all extra work. It is never labelled a deployable algorithm or exact
oracle upper bound. Scientific execution requires a previously saved lock.
"""

import argparse
import atexit
import copy
import csv
import hashlib
import importlib
import itertools
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = HERE / "happo_batch_cpu_preregistration.json"
RUNTIME = ROOT / "ICML-2027/runtime/pettingzoo_1_24_3"
UPSTREAM = ROOT / "tmp/HARL"
ENV_POOLS = {}
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(UPSTREAM))
# Pure-Python YAML is already installed in base; append rather than override
# the pinned ust2 NumPy/PyTorch packages. No environment installation occurs.
sys.path.append("D:/anaconda/Lib/site-packages")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def config(smoke=False):
    p = json.loads(PLAN.read_text(encoding="utf-8"))
    if smoke:
        p.update(blocks=12, episodes_per_training_block=4,
                 evaluation_every_blocks=3, evaluation_episodes=4,
                 oracle_search_episodes=4, actor_optimizer_steps_per_packet=1,
                 critic_optimizer_steps_per_block=1)
    return p


def seed_for(seed, task, block, role, episode):
    return int(np.random.SeedSequence([seed, task, block, role, episode])
               .generate_state(1)[0] % (2**31 - 1))


def make_env(task, length):
    module = importlib.import_module("pettingzoo.mpe." + task["scenario"])
    kw = dict(max_cycles=length, continuous_actions=False)
    if "N" in task:
        kw["N"] = task["N"]
    return module.parallel_env(**kw)


def env_pool(task, length, count):
    key = (task["name"], length, count)
    if key not in ENV_POOLS:
        ENV_POOLS[key] = [make_env(task, length) for _ in range(count)]
    return ENV_POOLS[key]


@atexit.register
def close_pools():
    for pool in ENV_POOLS.values():
        for env in pool:
            env.close()


def initialize(task, seed, p):
    import gym
    import yaml
    from harl.algorithms.actors.happo import HAPPO
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.valuenorm import ValueNorm

    env = make_env(task, p["episode_length"])
    obs, _ = env.reset(seed=seed)
    names = list(env.agents)
    c = yaml.safe_load((UPSTREAM / "harl/configs/algos_cfgs/happo.yaml").read_text())
    args = {**c["model"], **c["algo"], "hidden_sizes": p["hidden_sizes"],
            "activation_func": p["activation"]}
    actors = []
    for i, name in enumerate(names):
        torch.manual_seed(seed + i)
        obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(obs[name]),), dtype=np.float32)
        actors.append(HAPPO(args, obs_space, gym.spaces.Discrete(int(env.action_space(name).n))))
    torch.manual_seed(seed + 100)
    state_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(env.state()),), dtype=np.float32)
    critic = VCritic(args, state_space)
    normalizer = ValueNorm(1)
    env.close()
    return actors, critic, normalizer


def returns_to_go(rewards, discount):
    result = np.zeros_like(rewards, dtype=np.float32)
    total = np.zeros(len(rewards), dtype=np.float32)
    for t in range(rewards.shape[1] - 1, -1, -1):
        total = rewards[:, t] + discount * total
        result[:, t] = total
    return result


def rollouts(actors, task, task_index, seed, block, role, episodes, p, training=False):
    """Native fixed-length joint episodes, CRN coupled only across comparisons."""
    h, n = p["episode_length"], len(actors)
    all_obs = [[] for _ in actors]
    all_actions = [[] for _ in actors]
    all_logs = [[] for _ in actors]
    all_states, all_rewards, identities = [], [], []
    for start in range(0, episodes, 16):
        e_count = min(16, episodes - start)
        keys = [seed_for(seed, task_index, block, role, e) for e in range(start, start + e_count)]
        # MPE initializes pygame even for headless episodes. Reusing a reset
        # pool removes that setup cost; every episode still gets its exact seed.
        envs = env_pool(task, h, e_count)
        obs = [env.reset(seed=key)[0] for env, key in zip(envs, keys)]
        names = list(envs[0].agents)
        uniforms = np.stack([np.random.default_rng(key + 1).random((h, n)) for key in keys])
        ob = [[] for _ in actors]
        ac = [[] for _ in actors]
        lp = [[] for _ in actors]
        states, rewards = [], []
        for t in range(h):
            if training:
                states.append(np.stack([env.state() for env in envs]))
            choices = []
            for i, actor in enumerate(actors):
                o = np.stack([item[names[i]] for item in obs]).astype(np.float32)
                with torch.no_grad():
                    probs = actor.actor.act.action_out(actor.actor.base(torch.as_tensor(o))).probs.numpy()
                cumulative = np.cumsum(probs, axis=1)
                a = np.minimum((uniforms[:, t, i, None] > cumulative).sum(axis=1), probs.shape[1] - 1)
                choices.append(a)
                if training:
                    ob[i].append(o)
                    ac[i].append(a[:, None])
                    lp[i].append(np.log(probs[np.arange(e_count), a])[:, None])
            following, r = [], []
            for e, env in enumerate(envs):
                next_o, rew, term, trunc, _ = env.step({name: int(choices[i][e]) for i, name in enumerate(names)})
                if t < h - 1 and (any(term.values()) or any(trunc.values())):
                    raise AssertionError("unexpected early termination")
                following.append(next_o)
                r.append(float(np.mean(list(rew.values()))))
            rewards.append(r)
            obs = following
        all_rewards.append(np.asarray(rewards, np.float32).T)
        if training:
            all_states.append(np.asarray(states, np.float32).transpose(1, 0, 2))
            for i in range(n):
                all_obs[i].append(np.asarray(ob[i], np.float32).transpose(1, 0, 2))
                all_actions[i].append(np.asarray(ac[i], np.float32).transpose(1, 0, 2))
                all_logs[i].append(np.asarray(lp[i], np.float32).transpose(1, 0, 2))
        identities.extend(keys)
    rewards = np.concatenate(all_rewards)
    if not np.isfinite(rewards).all() or len(set(identities)) != episodes:
        raise AssertionError("nonfinite rewards or repeated episode identity")
    if not training:
        return rewards.sum(axis=1).astype(np.float64)
    return {"obs": [np.concatenate(x).reshape(-1, x[0].shape[-1]) for x in all_obs],
            "actions": [np.concatenate(x).reshape(-1, 1) for x in all_actions],
            "logs": [np.concatenate(x).reshape(-1, 1) for x in all_logs],
            "states": np.concatenate(all_states).reshape(-1, all_states[0].shape[-1]),
            "rewards": rewards, "identities": identities}


def add_advantages(data, critic, normalizer, p):
    count = len(data["states"])
    hidden = p["hidden_sizes"][-1]
    zeros, ones = np.zeros((count, 1, hidden), np.float32), np.ones((count, 1), np.float32)
    with torch.no_grad():
        raw = critic.get_values(data["states"], zeros, ones)[0].numpy()
        values = normalizer.denormalize(raw).reshape(data["rewards"].shape)
    following = np.concatenate([values[:, 1:], np.zeros((len(values), 1), np.float32)], axis=1)
    delta = data["rewards"] + p["discount"] * following - values
    adv = returns_to_go(delta, p["discount"] * p["gae_lambda"])
    data["raw_values"] = raw
    data["returns"] = (adv + values).reshape(-1, 1).astype(np.float32)
    data["advantages"] = ((adv - adv.mean()) / (adv.std() + 1e-5)).reshape(-1, 1).astype(np.float32)


def actor_logs(actor, data, agent, p):
    count = len(data["obs"][agent])
    zeros = np.zeros((count, 1, p["hidden_sizes"][-1]), np.float32)
    ones = np.ones((count, 1), np.float32)
    with torch.no_grad():
        return actor.evaluate_actions(data["obs"][agent], zeros, data["actions"][agent],
                                      ones, None, ones)[0].numpy()


def train_actor(actor, data, agent, factor, p):
    count = len(data["obs"][agent])
    zeros = np.zeros((count, 1, p["hidden_sizes"][-1]), np.float32)
    ones = np.ones((count, 1), np.float32)
    sample = (data["obs"][agent], zeros, data["actions"][agent], ones, ones,
              data["logs"][agent], data["advantages"], None, factor)
    for _ in range(p["actor_optimizer_steps_per_packet"]):
        loss, entropy, grad, _ = actor.update(sample)
        if not all(np.isfinite(float(x)) for x in (loss.detach(), entropy.detach(), grad)):
            raise AssertionError("nonfinite actor optimization")


def sequential_update(actors, data, indices, p):
    factor = np.ones_like(data["advantages"])
    for i in indices:
        old = actor_logs(actors[i], data, i, p)
        train_actor(actors[i], data, i, factor, p)
        factor *= np.exp(actor_logs(actors[i], data, i, p) - old)


def train_critic(critic, normalizer, data, p):
    count = len(data["states"])
    zeros = np.zeros((count, 1, p["hidden_sizes"][-1]), np.float32)
    sample = (data["states"], zeros, data["raw_values"], data["returns"], np.ones((count, 1), np.float32))
    for _ in range(p["critic_optimizer_steps_per_block"]):
        loss, grad = critic.update(sample, normalizer)
        if not np.isfinite(float(loss.detach())) or not np.isfinite(float(grad)):
            raise AssertionError("nonfinite critic optimization")


def packet(history, agent, version, p):
    actors, data = history[version]
    before = actors[agent]
    after = copy.deepcopy(before)
    train_actor(after, data, agent, np.ones_like(data["advantages"]), p)
    delta = {name: value.detach().clone() - dict(before.actor.named_parameters())[name].detach()
             for name, value in after.actor.named_parameters()}
    return delta, copy.deepcopy(after.actor_optimizer.state_dict())


def apply_packets(actors, packets, multipliers, accept_optimizer=True):
    with torch.no_grad():
        for i, alpha in enumerate(multipliers):
            if alpha:
                delta, opt = packets[i]
                for name, value in actors[i].actor.named_parameters():
                    value.add_(delta[name], alpha=float(alpha))
                if accept_optimizer:
                    actors[i].actor_optimizer.load_state_dict(copy.deepcopy(opt))


def actions(n, credit, multipliers):
    return [a for a in itertools.product(multipliers, repeat=n)
            if sum(x > 0 for x in a) <= credit]


def cyclic_action(n, k, block):
    a = [0.0] * n
    for j in range(k):
        a[(block * k + j) % n] = 1.0
    return tuple(a)


def surrogate_action(actors, data, deltas, credit, p):
    """Causal proxy using previously returned increments, before new launches.

    Product ratios include cross-agent terms. This empirical proxy is a strong
    adaptive control, not a return certificate. Its forward work is charged.
    """
    n = len(actors)
    log_ratios = []
    forwards = 0
    for i in range(n):
        per_step = {0.0: np.zeros_like(data["logs"][i])}
        for alpha in p["packet_step_multipliers"][1:]:
            candidate = copy.deepcopy(actors[i])
            with torch.no_grad():
                for name, value in candidate.actor.named_parameters():
                    value.add_(deltas[i][name], alpha=float(alpha))
            per_step[alpha] = actor_logs(candidate, data, i, p) - data["logs"][i]
            forwards += len(data["obs"][i])
        log_ratios.append(per_step)
    menu = actions(n, credit, p["packet_step_multipliers"])
    scores = []
    for a in menu:
        log_ratio = sum(log_ratios[i][alpha] for i, alpha in enumerate(a))
        score = float(((np.exp(log_ratio) - 1) * data["advantages"]).mean())
        scores.append(score)
    best = min(range(len(menu)), key=lambda j: (-scores[j], sum(x > 0 for x in menu[j]), menu[j]))
    return menu[best], scores[best], forwards


def parameters_hash(actors, critic, normalizer):
    tensors = [p.detach().cpu().numpy().tobytes() for actor in actors for p in actor.actor.parameters()]
    tensors += [p.detach().cpu().numpy().tobytes() for p in critic.critic.parameters()]
    tensors += [p.detach().cpu().numpy().tobytes() for p in normalizer.parameters()]
    return hashlib.sha256(b"".join(tensors)).hexdigest()


def run_payload(task_index, seed, method, out, smoke=False):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    p = config(smoke)
    task = p["tasks"][task_index]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    actors, critic, normalizer = initialize(task, seed, p)
    n, k, h = len(actors), task["launch_credit_per_block"], p["episode_length"]
    history, previous, curves, decisions = {}, {}, [], []
    credit = 0
    totals = dict(training_joint_transitions=0, training_actor_transitions=0,
                  packet_messages=0, actor_optimizer_rows=0, critic_optimizer_rows=0,
                  oracle_counterfactual_optimizer_rows=0, oracle_counterfactual_packets=0,
                  oracle_search_joint_transitions=0, oracle_search_actor_transitions=0,
                  benchmark_evaluation_joint_transitions=0, benchmark_evaluation_actor_transitions=0,
                  oracle_diagnostic_joint_transitions=0, oracle_diagnostic_actor_transitions=0,
                  proxy_forward_rows=0, packet_bytes=0, unused_credit_overflow=0)
    expected_initial = parameters_hash(actors, critic, normalizer)

    def evaluate(block):
        values = rollouts(actors, task, task_index, seed, block, 30, p["evaluation_episodes"], p)
        totals["benchmark_evaluation_joint_transitions"] += len(values) * h
        totals["benchmark_evaluation_actor_transitions"] += len(values) * h * n
        curves.append(dict(task=task["name"], seed=seed, method=method, block=block,
                           resource_fraction=block / p["blocks"], mean_return=float(values.mean()),
                           std_return=float(values.std(ddof=1)), evaluation_episodes=len(values),
                           training_joint_transitions=totals["training_joint_transitions"],
                           training_actor_transitions=totals["training_actor_transitions"],
                           packet_messages=totals["packet_messages"],
                           actor_optimizer_rows=totals["actor_optimizer_rows"]))
        return values

    evaluate(0)
    for block in range(p["blocks"]):
        data = rollouts(actors, task, task_index, seed, block, 10, p["episodes_per_training_block"], p, training=True)
        add_advantages(data, critic, normalizer, p)
        history[block] = (copy.deepcopy(actors), data)
        for old in list(history):
            if old < block - 8:
                del history[old]
        totals["training_joint_transitions"] += len(data["states"])
        totals["training_actor_transitions"] += len(data["states"]) * n
        # Full-cadence HAPPO launches the entire team. A cap of N would drop
        # credits whenever K does not divide N; N+K-1 is the smallest lossless
        # cap. Subset rules need at most N credits for their current action.
        bank_cap = n + k - 1 if method == "happo_full_cadence" else n
        totals["unused_credit_overflow"] += max(0, credit + k - bank_cap)
        credit = min(bank_cap, credit + k)
        ages = np.roll(np.asarray(p["ages"][task["name"]]), block % n).tolist()
        versions = [max(0, block - int(age)) for age in ages]
        proxy_score, oracle_margin = 0.0, 0.0
        chosen = cyclic_action(n, k, block)
        packets = {}
        if method == "happo_full_cadence":
            chosen = tuple([1.0] * n) if credit >= n else tuple([0.0] * n)
        elif method == "age_first":
            selected = sorted(range(n), key=lambda i: (block - versions[i], (i - block) % n))[:k]
            chosen = tuple(float(i in selected) for i in range(n))
        elif method == "history_surrogate" and len(previous) == n:
            chosen, proxy_score, forwards = surrogate_action(actors, data, previous, credit, p)
            totals["proxy_forward_rows"] += forwards
        if method == "outcome_aware_oracle":
            for i in range(n):
                packets[i] = packet(history, i, versions[i], p)
            menu = actions(n, credit, p["packet_step_multipliers"])
            candidates = []
            for a in menu:
                trial = copy.deepcopy(actors)
                apply_packets(trial, packets, a, accept_optimizer=False)
                values = rollouts(trial, task, task_index, seed, block, 20,
                                  p["oracle_search_episodes"], p)
                candidates.append(float(values.mean()))
            best = min(range(len(menu)), key=lambda j: (-candidates[j], sum(x > 0 for x in menu[j]), menu[j]))
            chosen = menu[best]
            oracle_margin = candidates[best] - candidates[menu.index(tuple([0.0] * n))]
            search_work = len(menu) * p["oracle_search_episodes"] * h
            totals["oracle_search_joint_transitions"] += search_work
            totals["oracle_search_actor_transitions"] += search_work * n
            extra = n - sum(x > 0 for x in chosen)
            totals["oracle_counterfactual_packets"] += extra
            totals["oracle_counterfactual_optimizer_rows"] += extra * len(data["states"]) * p["actor_optimizer_steps_per_packet"]
        launched = [i for i, a in enumerate(chosen) if a]
        if len(launched) > credit:
            raise AssertionError("launch credit exceeded")
        if method in ("happo_cyclic", "happo_full_cadence"):
            sequential_update(actors, data, launched, p)
            versions = [block] * n
        else:
            if method == "parallel_cyclic":
                versions = [block] * n
            for i in launched:
                if i not in packets:
                    packets[i] = packet(history, i, versions[i], p)
                previous[i] = {name: d.clone() for name, d in packets[i][0].items()}
            apply_packets(actors, packets, chosen)
        credit -= len(launched)
        totals["packet_messages"] += len(launched)
        totals["actor_optimizer_rows"] += len(launched) * len(data["states"]) * p["actor_optimizer_steps_per_packet"]
        for i in launched:
            # Local Adam packets carry the increment and both optimizer moments.
            totals["packet_bytes"] += sum(v.numel() * v.element_size() for v in actors[i].actor.parameters())
            totals["packet_bytes"] += sum(v.numel() * v.element_size()
                                         for state in actors[i].actor_optimizer.state.values()
                                         for v in state.values() if torch.is_tensor(v))
        train_critic(critic, normalizer, data, p)
        totals["critic_optimizer_rows"] += len(data["states"]) * p["critic_optimizer_steps_per_block"]
        decisions.append(dict(task=task["name"], seed=seed, method=method, block=block,
                              action=json.dumps(chosen, separators=(",", ":")),
                              source_versions=json.dumps(versions, separators=(",", ":")),
                              launched_packets=len(launched), remaining_credit=credit,
                              differs_from_cyclic=chosen != cyclic_action(n, k, block),
                              proxy_score=proxy_score, oracle_search_margin=oracle_margin,
                              heldout_increment_measured=False, heldout_oracle_increment=0.0,
                              train_episode_seed_hash=hashlib.sha256(np.asarray(data["identities"], dtype=np.int64).tobytes()).hexdigest()))
        if (block + 1) % p["evaluation_every_blocks"] == 0:
            current_values = evaluate(block + 1)
            if method == "outcome_aware_oracle":
                old_values = rollouts(history[block][0], task, task_index, seed,
                                      block + 1, 30, p["evaluation_episodes"], p)
                decisions[-1]["heldout_increment_measured"] = True
                decisions[-1]["heldout_oracle_increment"] = float((current_values - old_values).mean())
                totals["oracle_diagnostic_joint_transitions"] += len(old_values) * h
                totals["oracle_diagnostic_actor_transitions"] += len(old_values) * h * n
        dump(out / "progress.json", dict(block=block + 1, blocks=p["blocks"], task=task["name"], seed=seed, method=method))
    csv_write(out / "curves.csv", curves)
    csv_write(out / "decisions.csv", decisions)
    expected_joint = p["blocks"] * p["episodes_per_training_block"] * h
    max_messages = p["blocks"] * k
    if totals["training_joint_transitions"] != expected_joint or totals["packet_messages"] > max_messages:
        raise AssertionError("exact data/message accounting failed")
    scientific = dict(task=task["name"], seed=seed, method=method, agents=n,
                      initial_parameters_sha256=expected_initial,
                      final_parameters_sha256=parameters_hash(actors, critic, normalizer),
                      totals=totals, max_training_packet_messages=max_messages,
                      max_actor_optimizer_rows=max_messages * p["episodes_per_training_block"] * h * p["actor_optimizer_steps_per_packet"],
                      benchmark_evaluation_used_by_selection=False,
                      oracle_search_used_by_selection=method == "outcome_aware_oracle",
                      source_sha256=sha(__file__), plan_sha256=sha(PLAN), smoke=smoke)
    dump(out / "scientific.json", scientific)
    dump(out / "runtime.json", dict(seconds=time.perf_counter() - started, python=sys.version,
                                    torch=torch.__version__, platform=platform.platform()))
    (out / "SHA256SUMS").write_text("".join(f"{sha(out / name)}  {name}\n" for name in
                                            ("curves.csv", "decisions.csv", "scientific.json", "runtime.json")), encoding="utf-8")
    print(f"COMPLETE {task['name']} {seed} {method}", flush=True)
    return scientific


def lock_state():
    p = config()
    head = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    if head != p["harl_commit"] or dirty:
        raise RuntimeError("clean pinned HARL required")
    runtime_files = sorted(RUNTIME.rglob("*.py"))
    return dict(experiment_id=p["experiment_id"], harl_commit=head,
                files={str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                       for path in (Path(__file__), PLAN, HERE / "analyze_happo_batch_cpu.py")},
                mpe_runtime_tree_sha256=hashlib.sha256("".join(
                    str(path.relative_to(RUNTIME)).replace("\\", "/") + ":" + sha(path) + "\n"
                    for path in runtime_files).encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--registered-replay", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        if args.freeze.exists():
            raise FileExistsError("lock exists")
        dump(args.freeze, lock_state())
        return
    if not args.smoke and (not args.lock or json.loads(args.lock.read_text()) != lock_state()):
        raise RuntimeError("scientific run requires exact frozen lock")
    if not args.out or args.out.exists():
        raise FileExistsError("fresh output root required")
    args.out.mkdir(parents=True)
    p = config(args.smoke)
    seeds = [p["qualification_seed"]] if args.smoke else p["development_seeds"] + p["holdout_seeds"]
    methods = p["methods"]
    if args.registered_replay:
        seeds, methods = [p["replay"]["seed"]], p["replay"]["methods"]
    jobs = [(ti, s, m, args.out / f"{task['name']}_{s}_{m}")
            for ti, task in enumerate(p["tasks"]) for s in seeds for m in methods
            if args.smoke or args.registered_replay or m != "outcome_aware_oracle"
            or s not in p["development_seeds"] or p["oracle_development_runs"]]
    dump(args.out / "execution.json", dict(command=sys.argv, lock=lock_state(), payloads=len(jobs), smoke=args.smoke))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_payload, ti, s, m, path, args.smoke) for ti, s, m, path in jobs]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
