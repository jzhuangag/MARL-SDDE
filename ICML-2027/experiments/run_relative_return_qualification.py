"""Frozen, one-worker native-MPE qualification of relative-return estimators."""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np
import torch
import happo_batch_cpu as base
from relative_return_estimator import draw, maximal_pair, common_joint, disagreeing_joint, first_disagreement, finite_game_audit

HERE = Path(__file__).resolve().parent
PLAN = HERE / "relative_return_qualification.json"


def dump(path, obj):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def probs(actors, obs, names, ledger):
    result = []
    with torch.no_grad():
        for actor, name in zip(actors, names):
            x = torch.as_tensor(np.asarray(obs[name], dtype=np.float32)[None])
            result.append(actor.actor.act.action_out(actor.actor.base(x)).probs.numpy()[0].astype(float))
            ledger["policy_forward_rows"] += 1
    return result


def transition(env, names, actions, ledger):
    obs, rewards, _, _, _ = env.step(dict(zip(names, map(int, actions))))
    ledger["joint_transitions"] += 1
    return obs, float(np.mean(list(rewards.values())))


def estimate(candidate, reference, task, key, method, h):
    ledger = {"joint_transitions":0,"policy_forward_rows":0}
    envs = base.env_pool(task, h, 3)
    names = list(envs[0].possible_agents)
    rng = np.random.default_rng(key + 19)
    if method in ("paired_crn", "paired_maximal"):
        oa, ob = envs[0].reset(seed=key)[0], envs[1].reset(seed=key)[0]
        ra = rb = 0.0
        for _ in range(h):
            pp, qq = probs(candidate, oa, names, ledger), probs(reference, ob, names, ledger)
            aa, bb = [], []
            for p, q in zip(pp, qq):
                if method == "paired_maximal":
                    a, b = maximal_pair(p, q, rng)
                else:
                    u = rng.random()
                    a = min(int(np.searchsorted(np.cumsum(p / p.sum()), u)), len(p)-1)
                    b = min(int(np.searchsorted(np.cumsum(q / q.sum()), u)), len(q)-1)
                aa.append(a)
                bb.append(b)
            oa, r = transition(envs[0], names, aa, ledger)
            ra += r
            ob, r = transition(envs[1], names, bb, ledger)
            rb += r
        return ra-rb, ledger, {}
    common_obs = envs[0].reset(seed=key)[0]
    prefix_checks = []
    def common(t):
        nonlocal common_obs
        pp, qq = probs(candidate, common_obs, names, ledger), probs(reference, common_obs, names, ledger)
        aa, c = common_joint(pp, qq, rng)
        prefix_checks.append({name:common_obs[name].copy() for name in names})
        common_obs, _ = transition(envs[0], names, aa, ledger)
        return aa, c
    def branch(t, prefix):
        oa, ob = envs[1].reset(seed=key)[0], envs[2].reset(seed=key)[0]
        for k, aa in enumerate(prefix):
            if not all(np.array_equal(oa[name], prefix_checks[k][name]) and
                       np.array_equal(ob[name], prefix_checks[k][name]) for name in names):
                raise AssertionError("replayed prefix differs")
            oa, _ = transition(envs[1], names, aa, ledger)
            ob, _ = transition(envs[2], names, aa, ledger)
        if not all(np.array_equal(oa[name], prefix_checks[t][name]) and
                   np.array_equal(ob[name], prefix_checks[t][name]) for name in names):
            raise AssertionError("branch state differs")
        pp, qq = probs(candidate, oa, names, ledger), probs(reference, ob, names, ledger)
        aa, bb = disagreeing_joint(pp, qq, rng)
        ra = rb = 0.0
        for k in range(t, h):
            if k > t:
                pp, qq = probs(candidate, oa, names, ledger), probs(reference, ob, names, ledger)
                pairs = [maximal_pair(p,q,rng) for p,q in zip(pp,qq)]
                aa, bb = [x[0] for x in pairs], [x[1] for x in pairs]
            oa, r = transition(envs[1], names, aa, ledger)
            ra += r
            ob, r = transition(envs[2], names, bb, ledger)
            rb += r
        return ra-rb
    value, aux = first_disagreement(common, branch, h, rng)
    return value, ledger, aux


def actor_hash(actors):
    hh = hashlib.sha256()
    for actor in actors:
        for p in actor.actor.parameters():
            hh.update(p.detach().numpy().tobytes())
    return hh.hexdigest()


def run(args):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    plan = json.loads(PLAN.read_text())
    lock = json.loads(Path(args.lock).read_text())
    for name, expected in lock["files"].items():
        if base.sha(base.ROOT / name) != expected:
            raise AssertionError("source lock mismatch: " + name)
    if args.qualification:
        plan.update(seeds=[74109999],stages=[1],score_samples=2,validation_episodes=2)
        plan["tasks"] = [plan["tasks"][0]]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "execution.json", {"state":"running","plan":plan,"lock":lock,"evaluation_used_for_scoring":False})
    p = base.config()
    p.update(episode_length=plan["episode_length"], episodes_per_training_block=plan["training_episodes"])
    started = time.process_time()
    context_index = 0
    for ti, task in enumerate(plan["tasks"]):
        for seed in plan["seeds"]:
            actors, critic, normalizer = base.initialize(task, seed, p)
            for stage in range(1, max(plan["stages"])+1):
                data = base.rollouts(actors,task,ti,seed,stage,plan["training_role"],plan["training_episodes"],p,training=True)
                base.add_advantages(data,critic,normalizer,p)
                before = copy.deepcopy(actors) if stage in plan["stages"] else None
                base.sequential_update(actors,data,list(range(len(actors))),p)
                base.train_critic(critic,normalizer,data,p)
                if stage not in plan["stages"]:
                    continue
                wanted = args.context is None or context_index == args.context
                if wanted:
                    context = {"task":task,"seed":seed,"stage":stage,"index":context_index,
                               "reference_hash":actor_hash(actors),"initial_hash":actor_hash(before),
                               "training_joint_transitions":stage*plan["training_episodes"]*p["episode_length"],
                               "training_actor_optimizer_rows":stage*len(actors)*p["actor_optimizer_steps_per_packet"]*len(data["states"]),
                               "training_critic_optimizer_rows":stage*p["critic_optimizer_steps_per_block"]*len(data["states"]),
                               "evaluation_used_for_scoring":False,"candidates":[]}
                    for ci, scale in enumerate(plan["candidate_scales"]):
                        candidate = copy.deepcopy(actors)
                        with torch.no_grad():
                            for old, ref, new in zip(before,actors,candidate):
                                for a,b,c in zip(old.actor.parameters(),ref.actor.parameters(),new.actor.parameters()):
                                    c.copy_(a+scale*(b-a))
                        row = {"scale":scale,"candidate_hash":actor_hash(candidate),"samples":{},"ledgers":{},"aux":{}}
                        for mi, method in enumerate(plan["methods"]):
                            samples, aux = [], []
                            costs = {"joint_transitions":0,"policy_forward_rows":0}
                            for e in range(plan["score_samples"]):
                                key = base.seed_for(seed,ti,stage,plan["score_role"]+mi*10+ci,e)
                                value, ledger, extra = estimate(candidate,actors,task,key,method,p["episode_length"])
                                samples.append(value)
                                aux.append(extra)
                                for k in costs:
                                    costs[k] += ledger[k]
                                if time.process_time()-started > plan["active_cpu_limit_seconds"]:
                                    raise TimeoutError("registered active CPU limit")
                            row["samples"][method], row["ledgers"][method], row["aux"][method] = samples,costs,aux
                        # Independent validation never influences a score or candidate.
                        aa = base.rollouts(candidate,task,ti,seed,stage,plan["validation_role"],plan["validation_episodes"],p)
                        bb = base.rollouts(actors,task,ti,seed,stage,plan["validation_role"],plan["validation_episodes"],p)
                        row["validation"] = (aa-bb).tolist()
                        row["validation_joint_transitions"] = 2*plan["validation_episodes"]*p["episode_length"]
                        row["candidate_parameter_messages"] = len(actors)
                        context["candidates"].append(row)
                    # Identity is a structural null, using a disjoint seed.
                    context["identity_checks"] = {}
                    for method in plan["methods"]:
                        key = base.seed_for(seed,ti,stage,410,0)
                        value,ledger,_ = estimate(actors,actors,task,key,method,p["episode_length"])
                        if value != 0:
                            raise AssertionError("nonzero identity score")
                        context["identity_checks"][method] = {"score":value,"ledger":ledger}
                    dump(out / ("context_%02d.json" % context_index),context)
                    dump(out / "progress.json", {"completed_contexts":context_index+1,"total_contexts":8})
                    print(json.dumps({"completed_context":context_index,"cpu_seconds":round(time.process_time()-started,2)}),flush=True)
                context_index += 1
                if args.context is not None and context_index > args.context:
                    break
            if args.context is not None and context_index > args.context:
                break
        if args.context is not None and context_index > args.context:
            break
    dump(out / "finite_game.json",finite_game_audit())
    hashes = {x.name:base.sha(x) for x in sorted(out.glob("context_*.json"))}
    hashes["finite_game.json"] = base.sha(out/"finite_game.json")
    dump(out / "hashes.json",hashes)
    dump(out / "execution.json", {"state":"complete","lock":lock,"cpu_seconds":time.process_time()-started,
                                   "contexts":len(list(out.glob("context_*.json"))),"evaluation_used_for_scoring":False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--lock",required=True)
    parser.add_argument("--context",type=int)
    parser.add_argument("--qualification",action="store_true")
    run(parser.parse_args())
