"""Additional source/runtime/role audit, without changing estimator gates."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seed_for(seed,task,block,role,episode):
    return int(np.random.SeedSequence([seed,task,block,role,episode]).generate_state(1)[0]%(2**31-1))


def verify(root,lock,output):
    repo=Path(__file__).resolve().parents[2]
    root=Path(root)
    lock=json.loads(Path(lock).read_text())
    ex=json.loads((root/"execution.json").read_text())
    if ex["state"]!="complete" or ex["contexts"]!=8:
        raise AssertionError("incomplete payloads")
    for name,digest in lock["files"].items():
        if sha(repo/name)!=digest:
            raise AssertionError("source mutated: "+name)
    upstream=repo/"tmp/HARL"
    commit=subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(upstream),"status","--porcelain"],text=True).strip()
    if commit!=lock["harl_commit"] or dirty:
        raise AssertionError("pinned HARL changed")
    runtime=repo/"ICML-2027/runtime/pettingzoo_1_24_3"
    rt=hashlib.sha256("".join(p.relative_to(runtime).as_posix()+":"+sha(p)+"\n"
                              for p in sorted(runtime.rglob("*.py"))).encode()).hexdigest()
    if rt!=lock["mpe_runtime_tree_sha256"]:
        raise AssertionError("preserved native runtime changed")
    plan=json.loads((repo/"ICML-2027/experiments/relative_return_a1/relative_return_qualification.json").read_text())
    train,score,valid=set(),set(),set()
    for ti,_ in enumerate(plan["tasks"]):
        for seed in plan["seeds"]:
            for stage in range(1,max(plan["stages"])+1):
                for e in range(plan["training_episodes"]):
                    train.add(seed_for(seed,ti,stage,plan["training_role"],e))
            for stage in plan["stages"]:
                for mi,_ in enumerate(plan["methods"]):
                    for ci,_ in enumerate(plan["candidate_scales"]):
                        for e in range(plan["score_samples"]):
                            score.add(seed_for(seed,ti,stage,plan["score_role"]+mi*10+ci,e))
                for e in range(plan["validation_episodes"]):
                    valid.add(seed_for(seed,ti,stage,plan["validation_role"],e))
    if (train&score) or (train&valid) or (score&valid) or len(score)!=8*3*3*64 or len(valid)!=8*256:
        raise AssertionError("role collision or seed duplication")
    for path in sorted(root.glob("context_*.json")):
        p=json.loads(path.read_text())
        n=3 if p["task"]["name"]=="spread" else 2
        if p["training_actor_optimizer_rows"]!=p["stage"]*n*5*16*25 or p["training_critic_optimizer_rows"]!=p["stage"]*5*16*25:
            raise AssertionError("training optimizer accounting")
        if any(x["score"]!=0 for x in p["identity_checks"].values()):
            raise AssertionError("identity control failed")
        if p["evaluation_used_for_scoring"] or any(row["candidate_parameter_messages"]!=n for row in p["candidates"]):
            raise AssertionError("role or logical payload accounting")
        for name,digest in json.loads((root/"hashes.json").read_text()).items():
            if sha(root/name)!=digest:
                raise AssertionError("payload changed")
    result={"pass":True,"harl_commit":commit,"runtime_tree_sha256":rt,"contexts":8,
            "unique_training_episode_seeds":len(train),"unique_score_episode_seeds":len(score),
            "unique_validation_episode_seeds":len(valid),"three_role_sets_disjoint":True,
            "parameter_messages":"Logical actor-parameter payloads only; single-process simulator, not measured network messages",
            "source_files_verified":len(lock["files"]),"identity_controls":24}
    with Path(output).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--lock",required=True);ap.add_argument("--output",required=True)
    args=ap.parse_args();print(json.dumps(verify(args.root,args.lock,args.output),indent=2))
