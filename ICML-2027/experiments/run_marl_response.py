"""Freeze-then-run real-trajectory MAPPO response and stage-transfer checks."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from marl_response import (ROOT, QuadraticResponse, actor_coordinates, clone_models,
                           collect, critic_coordinates, initialize, join, model_hash,
                           objective, training_update)


HERE = Path(__file__).resolve().parent
PLAN = HERE / "marl_response_preregistration.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scalar(function, coordinates):
    with torch.no_grad():
        return float(function(coordinates))


def sign_accuracy(records, predicted, truth):
    eligible = [r for r in records if abs(r[truth]) > 1e-8]
    return float(np.mean([np.sign(r[predicted]) == np.sign(r[truth]) for r in eligible])) if eligible else 0.


def summarize(rows, profiles, plan):
    counts, regrets, ties = {}, [], 0
    for row in rows:
        key = tuple(row[k] for k in ("task", "seed", "coupling", "stage", "work_mode"))
        counts.setdefault(key, []).append(row)
    for group in counts.values():
        actual = [r["critic_truth_delta"] for r in group]
        selected = min(group, key=lambda r: (r["critic_prediction"], r["q"]))
        spread = max(actual)-min(actual)
        tied = spread <= plan["risk_tie_relative_tolerance"]*max(1.,abs(group[0]["critic_truth_base"]))
        ties += int(tied)
        regrets.append(0. if tied else (selected["critic_truth_delta"]-min(actual))/spread)
    relative = [abs(r["critic_prediction"]-r["critic_sensor_delta"])/max(abs(r["critic_sensor_delta"]), 1e-8) for r in rows]
    naive = [abs(r["critic_parameter_only_prediction"]-r["critic_sensor_delta"])/max(abs(r["critic_sensor_delta"]), 1e-8) for r in rows]
    transferred = [r for r in rows if r["stage"] == 1]
    shared_difference = []
    for group in counts.values():
        if group[0]["coupling"] == 1 and group[0]["work_mode"] == "fixed_steps":
            reference = next(r for r in group if r["q"] == 1)
            shared_difference.extend(abs(r["critic_truth_delta"]-reference["critic_truth_delta"])/max(1.,abs(reference["critic_truth_base"])) for r in group)
    metrics = {
        "critic_same_sensor_quadratic_relative_error_median": float(np.median(relative)),
        "critic_parameter_only_relative_error_median": float(np.median(naive)),
        "critic_independent_truth_sign_accuracy": sign_accuracy(rows, "critic_prediction", "critic_truth_delta"),
        "critic_previous_stage_truth_sign_accuracy": sign_accuracy(transferred, "critic_previous_prediction", "critic_truth_delta"),
        "actor_independent_truth_sign_accuracy": sign_accuracy(rows, "actor_prediction", "actor_truth_delta"),
        "actor_previous_stage_truth_sign_accuracy": sign_accuracy(transferred, "actor_previous_prediction", "actor_truth_delta"),
        "critic_ranking_normalized_regret_median": float(np.median(regrets)),
        "numerically_tied_ranking_contexts": ties,
        "shared_fixed_step_critic_risk_relative_difference_max": max(shared_difference, default=0.),
        "shared_fixed_step_parameter_difference_max": max((r["shared_parameter_difference"] for r in rows),default=0.)}
    cfg = plan["mandatory_gates"]
    numeric = all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
    base_checks = {
        "complete_finite_disjoint_accounting": numeric and len(rows)==384 and len(profiles)==24
            and all(p["role_seed_overlap"] == 0 for p in profiles)
            and sum(p["joint_environment_transitions"] for p in profiles)==31200,
        "identical_candidate_initialization": all(r["initialization_matches"] for r in rows),
        "full_catalogue": all({r["q"] for r in group}==set(plan["q_catalogue"]) for group in counts.values()),
        "shared_fixed_step_parameter_invariance": metrics["shared_fixed_step_parameter_difference_max"] <= cfg["shared_fixed_step_parameter_invariance_tolerance"],
        "critic_same_sensor_quadratic_relative_error_median": metrics["critic_same_sensor_quadratic_relative_error_median"] <= cfg["critic_same_sensor_quadratic_relative_error_median_max"],
        "critic_independent_truth_sign_accuracy": metrics["critic_independent_truth_sign_accuracy"] >= cfg["critic_independent_truth_sign_accuracy_min"],
        "critic_previous_stage_truth_sign_accuracy": metrics["critic_previous_stage_truth_sign_accuracy"] >= cfg["critic_previous_stage_truth_sign_accuracy_min"],
        "critic_ranking_normalized_regret_median": metrics["critic_ranking_normalized_regret_median"] <= cfg["critic_ranking_normalized_regret_median_max"]}
    return {"metrics":metrics,"gates_without_replay":base_checks,
            "all_gates_without_replay_pass":all(base_checks.values()),
            "task_metrics":{task:{"critic_sign_accuracy":sign_accuracy([r for r in rows if r["task"]==task],"critic_prediction","critic_truth_delta"),
                                 "actor_sign_accuracy":sign_accuracy([r for r in rows if r["task"]==task],"actor_prediction","actor_truth_delta")}
                            for task in sorted({r["task"] for r in rows})}}


def run(out, smoke):
    if out.exists():
        raise FileExistsError("refusing to overwrite response artifacts")
    upstream = ROOT / "tmp/HARL"
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    dirty = subprocess.check_output(["git","-C",str(upstream),"status","--porcelain"],text=True).strip()
    if head!=plan["harl_commit"] or dirty:
        raise RuntimeError("exact clean pinned HARL required")
    import pettingzoo
    if pettingzoo.__version__!=plan["pettingzoo_version"]:
        raise RuntimeError("wrong isolated PettingZoo version")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True)
    (out/"buffers").mkdir()
    rows, profiles = [], []
    tasks = plan["tasks"]
    seeds = [plan["qualification_only_smoke_seed"]] if smoke else plan["seeds"]
    couplings = [1.] if smoke else plan["couplings"]
    started = time.perf_counter()
    try:
        for task in tasks:
            for seed in seeds:
                for coupling in couplings:
                    profile = {"task":task["name"],"seed":seed,"coupling":coupling,
                               "joint_environment_transitions":0,"individual_agent_transitions":0,
                               "actor_feature_rows":0,"critic_forward_rows":0,"hvp_calls":0,
                               "optimizer_actor_rows":0,"optimizer_critic_rows":0,
                               "role_seed_overlap":0}
                    actor, critic, normalizer = initialize(task, seed)
                    actor.actor.base.register_forward_pre_hook(lambda module,args,p=profile: p.__setitem__("actor_feature_rows",p["actor_feature_rows"]+len(args[0])))
                    critic.critic.register_forward_pre_hook(lambda module,args,p=profile: p.__setitem__("critic_forward_rows",p["critic_forward_rows"]+len(args[0])))
                    old_models = None
                    used_roles = {}
                    for stage in plan["stages"][:1] if smoke else plan["stages"]:
                        data_roles = {}
                        for role, workers, episodes in (("candidate",8,2),("sensor",1,2),("truth",1,6)):
                            buffers, ledger = collect(task,actor,critic,normalizer,seed=seed,stage=stage,
                                                      role=plan["roles"][role],workers=workers,episodes=episodes,coupling=coupling)
                            profile["joint_environment_transitions"] += ledger["joint_environment_transitions"]
                            profile["individual_agent_transitions"] += ledger["individual_agent_transitions"]
                            used_roles.setdefault(role,set()).update(i["seed"] for i in ledger["trajectory_identities"])
                            data_roles[role] = buffers
                            # Retain raw role buffers locally; only compact evidence is versioned.
                            np.savez_compressed(out/"buffers"/f"{task['name']}_{seed}_{coupling}_{stage}_{role}.npz",
                                                **{f"w{w}_{k}":v for w,b in enumerate(buffers) for k,v in b.items()})
                        sensor, truth = join(data_roles["sensor"],1), join(data_roles["truth"],1)
                        functions = {kind:objective(actor,critic,sensor,kind) for kind in ("actor","critic")}
                        truth_functions = {kind:objective(actor,critic,truth,kind) for kind in functions}
                        current = {"actor":actor_coordinates(actor),"critic":critic_coordinates(critic,normalizer)}
                        predictors = {kind:QuadraticResponse(functions[kind],current[kind]) for kind in functions}
                        truth_base = {kind:scalar(truth_functions[kind],current[kind]) for kind in functions}
                        initial = model_hash(actor,critic,normalizer)
                        shared_reference = None
                        for mode in plan["work_modes"]:
                            for q in plan["q_catalogue"]:
                                child_a,child_c,child_n = clone_models(actor,critic,normalizer)
                                initialized = model_hash(child_a,child_c,child_n)==initial
                                batch = join(data_roles["candidate"],q)
                                steps = 5 if mode=="fixed_steps" else 8//q
                                training_update(child_a,child_c,child_n,batch,steps)
                                n = len(batch["state"])
                                profile["optimizer_actor_rows"] += steps*n*batch["obs"].shape[1]
                                profile["optimizer_critic_rows"] += steps*n
                                following = {"actor":actor_coordinates(child_a),"critic":critic_coordinates(child_c,child_n)}
                                row = {"task":task["name"],"seed":seed,"coupling":coupling,"stage":stage,
                                       "work_mode":mode,"q":q,"optimizer_steps":steps,
                                       "training_joint_transitions":n,"training_agent_transitions":n*batch["obs"].shape[1],
                                       "initialization_matches":initialized,"shared_parameter_difference":0.}
                                if coupling==1 and mode=="fixed_steps":
                                    flat = np.concatenate([p.numpy().reshape(-1) for p in following["actor"]+following["critic"][:-2]])
                                    if q==1:
                                        shared_reference = flat.copy()
                                    row["shared_parameter_difference"] = float(np.max(np.abs(flat-shared_reference)))
                                for kind in functions:
                                    prediction,linear = predictors[kind].forecast(following[kind])
                                    sensor_delta = scalar(functions[kind],following[kind])-float(predictors[kind].base.detach())
                                    truth_delta = scalar(truth_functions[kind],following[kind])-truth_base[kind]
                                    previous = 0.
                                    if old_models is not None:
                                        previous = old_models[kind].forecast(following[kind])[0]-old_models[kind].forecast(current[kind])[0]
                                    row.update({kind+"_prediction":prediction,kind+"_linear_prediction":linear,
                                                kind+"_sensor_delta":sensor_delta,kind+"_truth_delta":truth_delta,
                                                kind+"_truth_base":truth_base[kind],kind+"_previous_prediction":previous})
                                row["critic_parameter_only_prediction"] = predictors["critic"].forecast(following["critic"],freeze_last=2)[0]
                                with torch.no_grad():
                                    sample = __import__("marl_response").actor_batch(truth)
                                    old_dist = actor.evaluate_actions(sample[0],sample[1],sample[2],sample[3],None,sample[4])[2]
                                    new_dist = child_a.evaluate_actions(sample[0],sample[1],sample[2],sample[3],None,sample[4])[2]
                                    kl = torch.distributions.kl_divergence(old_dist,new_dist)
                                    row["actor_policy_kl"] = float(kl.sum(-1).mean() if kl.ndim>1 else kl.mean())
                                rows.append(row)
                        if old_models is not None:
                            profile["hvp_calls"] += sum(m.hvp_calls for m in old_models.values())
                        old_models = predictors
                        write_rows(out/"response.csv",rows)
                        if stage==0 and not smoke:
                            for block in range(2):
                                warm, ledger = collect(task,actor,critic,normalizer,seed=seed,stage=10+block,
                                                       role=40,workers=2,episodes=1,coupling=0.)
                                profile["joint_environment_transitions"] += ledger["joint_environment_transitions"]
                                profile["individual_agent_transitions"] += ledger["individual_agent_transitions"]
                                used_roles.setdefault("warmup",set()).update(i["seed"] for i in ledger["trajectory_identities"])
                                warm = join(warm,2)
                                training_update(actor,critic,normalizer,warm,5)
                                profile["optimizer_actor_rows"] += 5*len(warm["state"])*warm["obs"].shape[1]
                                profile["optimizer_critic_rows"] += 5*len(warm["state"])
                        print(f"completed {task['name']} seed={seed} coupling={coupling} stage={stage}",flush=True)
                    profile["hvp_calls"] += sum(m.hvp_calls for m in old_models.values())
                    keys = list(used_roles)
                    profile["role_seed_overlap"] = sum(len(used_roles[a]&used_roles[b]) for i,a in enumerate(keys) for b in keys[i+1:])
                    profiles.append(profile)
                    write_json(out/"accounting.json",profiles)
        report = {"experiment_id":plan["experiment_id"],"smoke_only":smoke,"rows":len(rows),
                  "source_hashes":{p.name:sha(p) for p in (PLAN,HERE/"marl_response.py",Path(__file__))},
                  "harl_commit":head,"pettingzoo":pettingzoo.__version__,"torch":torch.__version__,
                  "scope":plan["scope"],"gpu_performance_authorized":False}
        if not smoke:
            report.update(summarize(rows,profiles,plan))
        else:
            report["operational_payload_finite"] = all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,(float,int)))
        write_json(out/"summary.json",report)
        write_json(out/"runtime.json",{"elapsed_seconds":time.perf_counter()-started,"python":sys.version})
        (out/"SHA256SUMS").write_text("".join(f"{sha(out/name)}  {name}\n" for name in ("response.csv","accounting.json","summary.json")),encoding="utf-8")
        print(json.dumps(report,indent=2),flush=True)
    except Exception as error:
        write_json(out/"failure.json",{"exception":type(error).__name__,"message":str(error),"completed_rows":len(rows),"no_retry":True})
        raise


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    run(args.out,args.smoke)
