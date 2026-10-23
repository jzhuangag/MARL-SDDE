"""Prospective fixed-primary MC potential gates and precision diagnostics."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import happo_batch_cpu as base
from run_relative_return_qualification import dump


def analyze(root,out,replay):
    root,out=Path(root),Path(out)
    if json.loads((root/"execution.json").read_text())["state"]!="complete":raise AssertionError("incomplete outcomes")
    out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(__file__).with_name("mc_relative_return_qualification.json").read_text())
    payloads=[json.loads(p.read_text()) for p in sorted(root.glob("context_*.json"))]
    expected=[(t["name"],s,k) for t in plan["tasks"] for s in plan["seeds"] for k in plan["stages"]]
    if [(p["task"],p["seed"],p["stage"]) for p in payloads]!=expected:raise AssertionError("mapping")
    for name,h in json.loads((root/"hashes.json").read_text()).items():
        if base.sha(root/name)!=h:raise AssertionError("artifact changed")
    rows=[];ratios={t["name"]:[] for t in plan["tasks"]};positive=0;compatible=True;direct_compatible=True
    for p in payloads:
        ti=0 if p["task"]=="spread" else 1;n=3 if ti==0 else 2
        train={base.seed_for(p["seed"],ti,k,110,e) for k in range(1,17) for e in range(16)}
        for idx,(count,role) in enumerate(((64,610),(256,710))):
            ids=[base.seed_for(p["seed"],ti,p["stage"],role,e) for e in range(count)]
            if ids!=p["bank_identities"][idx] or len(set(ids))!=count or set(ids)&train:raise AssertionError("bank roles")
        if set(p["bank_identities"][0])&set(p["bank_identities"][1]):raise AssertionError("validation leakage")
        if (p["score_bank_joint_transitions"]!=1600 or p["validation_bank_joint_transitions"]!=6400 or
                p["bank_policy_forward_rows"]!=5*n*8000 or p["potential_forward_rows"]!=8000 or
                p["fit_rows"]!=p["stage"]*16*25 or p["evaluation_used_by_fitting"]):raise AssertionError("exact bank/fitting accounting")
        scores=[]
        for r in p["candidates"]:
            x=np.asarray(r["score"]);y=np.asarray(r["validation"]);z=np.asarray(r["direct_branch"])
            if len(x)!=64 or len(y)!=256 or len(z)!=32 or not all(np.isfinite(v).all() for v in (x,y,z)):
                raise AssertionError("sample count/finiteness")
            cost=r["direct_ledger"]
            direct_cost=sum(25*(3 if a["weight"]>0 else 1) for a in r["direct_aux"])
            forwards=sum(2*n*25+(2*n*(25-a["branch_time"]) if a["weight"]>0 else 0) for a in r["direct_aux"])
            if cost["joint_transitions"]!=direct_cost or cost["policy_forward_rows"]!=forwards:raise AssertionError("exact branch accounting")
            xm,ym,zm=map(lambda v:float(v.mean()),(x,y,z))
            se,yse,zse=map(lambda v:float(v.std(ddof=1)/math.sqrt(len(v))),(x,y,z))
            ratio=(float(x.var(ddof=1))*25/3)/(max(float(z.var(ddof=1))*direct_cost/32,1e-30))
            ratios[p["task"]].append(max(ratio,1e-30))
            ok=abs(xm-ym)<=4*math.sqrt(se*se+yse*yse)+1e-10
            dok=abs(zm-ym)<=4*math.sqrt(zse*zse+yse*yse)+1e-10
            compatible &= ok;direct_compatible &= dok
            scores.append((xm-1.645*se,ym-1.645*yse))
            rows.append({"context":p["context"],"task":p["task"],"stage":p["stage"],"scale":r["scale"],
                         "mean":xm,"se":se,"validation_mean":ym,"validation_se":yse,"branch_mean":zm,"branch_se":zse,
                         "amortized_cost_variance_ratio":ratio,"compatible_4se":bool(ok),"direct_compatible_4se":bool(dok)})
        chosen=max(scores,key=lambda s:s[0]);positive+=int(chosen[0]>0 and chosen[1]>0)
    task_ratios={t:math.exp(float(np.log(v).mean())) for t,v in ratios.items()}
    replay_ok=bool(replay) and (root/"context_00.json").read_bytes()==(Path(replay)/"context_00.json").read_bytes()
    gates={"integrity_mapping_roles_accounting":True,"registered_payload_replay":replay_ok,
           "task_cost_variance":all(v<=.8 for v in task_ratios.values()),"score_truth_compatibility":bool(compatible),
           "different_estimator_truth_compatibility":bool(direct_compatible),"positive_validated_scores":positive>=4,
           "score_stage4_cost":1600/1600<=1}
    with (out/"metrics.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    result={"experiment":plan["id"],"pass":all(gates.values()),"gates":gates,"task_cost_variance_ratio":task_ratios,
            "positive_selected_contexts":positive,"score_native_transitions":12800,"validation_native_transitions":51200,
            "branch_comparator_native_transitions":sum(r["direct_ledger"]["joint_transitions"] for p in payloads for r in p["candidates"]),
            "formal_nonzero_simultaneous_certificate":False,"fixed_primary":"training-only finite-horizon Monte Carlo potential",
            "interpretation":"Fresh estimator qualification. Unbiasedness is structural, potential accuracy affects variance only. Normal-SE positive checks are not distribution-free confidence guarantees or MARL learning superiority."}
    dump(out/"summary.json",result);return result


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--output",required=True);ap.add_argument("--replay")
    args=ap.parse_args();print(json.dumps(analyze(args.root,args.output,args.replay),indent=2))
