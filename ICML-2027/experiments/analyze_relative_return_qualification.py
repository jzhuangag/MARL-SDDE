"""Registered estimator gates; empirical scores are not confidence certificates."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,x):
    with Path(p).open("w",encoding="utf-8",newline="\n") as f:
        json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")


def analyze(root,out,replay):
    root,out = Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    plan_path=Path(__file__).with_name("relative_return_qualification.json")
    plan=json.loads(plan_path.read_text())
    execution=json.loads((root/"execution.json").read_text())
    if execution["state"] != "complete":
        raise AssertionError("incomplete experiment")
    for name,expected in json.loads((root/"hashes.json").read_text()).items():
        if sha(root/name)!=expected:
            raise AssertionError("artifact hash mismatch")
    payloads=[json.loads(p.read_text()) for p in sorted(root.glob("context_*.json"))]
    expected=[(t["name"],s,k) for t in plan["tasks"] for s in plan["seeds"] for k in plan["stages"]]
    if [(p["task"]["name"],p["seed"],p["stage"]) for p in payloads]!=expected:
        raise AssertionError("mapping mismatch")
    replay_ok=False
    if replay:
        rr=Path(replay)
        replay_ok=(root/"context_00.json").read_bytes()==(rr/"context_00.json").read_bytes()
    rows=[]
    compatibility=True
    selected_positive=0
    contexts_improved=0
    task_ratios={t["name"]:[] for t in plan["tasks"]}
    for p in payloads:
        ctx_ratios=[]
        scores=[]
        for row in p["candidates"]:
            truth=np.asarray(row["validation"])
            truth_mean=float(truth.mean())
            truth_se=float(truth.std(ddof=1)/math.sqrt(len(truth)))
            variance={m:float(np.var(row["samples"][m],ddof=1)) for m in plan["methods"]}
            eff={m:variance[m]*row["ledgers"][m]["joint_transitions"]/plan["score_samples"] for m in plan["methods"]}
            denominator=min(eff[m] for m in ("paired_crn","paired_maximal"))
            ratio=eff["first_disagreement"]/max(denominator,1e-30)
            ctx_ratios.append(ratio)
            task_ratios[p["task"]["name"]].append(max(ratio,1e-30))
            for method in plan["methods"]:
                x=np.asarray(row["samples"][method])
                if len(x)!=plan["score_samples"] or not np.isfinite(x).all() or not np.isfinite(truth).all():
                    raise AssertionError("sample count/finiteness")
                cost=row["ledgers"][method]
                expected_cost=2*plan["score_samples"]*plan["episode_length"]
                if method=="first_disagreement":
                    expected_cost=sum(plan["episode_length"]*(3 if a["weight"]>0 else 1) for a in row["aux"][method])
                    n=3 if p["task"]["name"]=="spread" else 2
                    expected_forwards=sum(2*n*plan["episode_length"]+
                                          (2*n*(plan["episode_length"]-a["branch_time"]) if a["weight"]>0 else 0)
                                          for a in row["aux"][method])
                else:
                    n=3 if p["task"]["name"]=="spread" else 2
                    expected_forwards=2*n*plan["episode_length"]*plan["score_samples"]
                if cost["joint_transitions"]!=expected_cost or cost["policy_forward_rows"]!=expected_forwards:
                    raise AssertionError("exact estimator accounting")
                if row["validation_joint_transitions"]!=2*plan["validation_episodes"]*plan["episode_length"]:
                    raise AssertionError("exact validation accounting")
                mean=float(x.mean())
                se=float(x.std(ddof=1)/math.sqrt(len(x)))
                compatible=abs(mean-truth_mean)<=4*math.sqrt(se*se+truth_se*truth_se)+1e-10
                compatibility &= compatible
                rows.append({"context":p["index"],"task":p["task"]["name"],"seed":p["seed"],"stage":p["stage"],
                             "scale":row["scale"],"method":method,"mean":mean,"se":se,
                             "validation_mean":truth_mean,"validation_se":truth_se,"variance":variance[method],
                             "joint_transitions":cost["joint_transitions"],"policy_forward_rows":cost["policy_forward_rows"],
                             "cost_variance_ratio_vs_best_paired":ratio,"compatible_4se":bool(compatible),
                             "probe_to_prefix_ratio":cost["joint_transitions"]*len(p["candidates"])/p["training_joint_transitions"]})
                if method=="first_disagreement":
                    scores.append((mean-1.645*se,truth_mean-1.645*truth_se,row["scale"]))
        chosen=max(scores,key=lambda x:x[0])
        selected_positive+=int(chosen[0]>0 and chosen[1]>0)
        contexts_improved+=int(math.exp(float(np.log(np.maximum(ctx_ratios,1e-30)).mean()))<1)
    ratios={t:math.exp(float(np.log(v).mean())) for t,v in task_ratios.items()}
    gates={"complete_finite_mapping_accounting":True,"registered_payload_replay":replay_ok,
           "finite_game_identity":json.loads((root/"finite_game.json").read_text())["absolute_error"]<1e-12,
           "task_cost_variance":all(x<=plan["gates"]["task_geomean_cost_variance_ratio_vs_best_paired_max"] for x in ratios.values()),
           "context_cost_variance":contexts_improved/len(payloads)>=plan["gates"]["context_fraction_cost_variance_improved_min"],
           "truth_compatibility":bool(compatibility),
           "positive_validated_selected_scores":selected_positive>=plan["gates"]["positive_validated_selected_contexts_min"],
           "transition_overhead":all(r["joint_transitions"]<=1.5*2*plan["episode_length"]*plan["score_samples"] for r in rows),
           "forward_overhead":all(r["policy_forward_rows"]<=2*2*(3 if r["task"]=="spread" else 2)*plan["episode_length"]*plan["score_samples"] for r in rows)}
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary={"experiment":plan["id"],"contexts":len(payloads),"gates":gates,"pass":all(gates.values()),
             "task_geomean_cost_variance_ratio":ratios,"contexts_improved":contexts_improved,
             "positive_independently_validated_selected_scores":selected_positive,
             "rigorous_return_certificate_established":False,
             "interpretation":"Estimator qualification only. Normal-SE diagnostics are not distribution-free confidence bounds. No algorithm/GPU performance claim follows.",
             "score_transitions":sum(r["joint_transitions"] for r in rows),
             "validation_transitions":len(payloads)*len(plan["candidate_scales"])*2*plan["validation_episodes"]*plan["episode_length"]}
    dump(out/"summary.json",summary)
    return summary


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--output",required=True); ap.add_argument("--replay")
    args=ap.parse_args(); print(json.dumps(analyze(args.root,args.output,args.replay),indent=2))
