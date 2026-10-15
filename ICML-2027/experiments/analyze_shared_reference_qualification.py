"""Fixed-primary shared-bank estimator decision; no best-method selection."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from run_relative_return_qualification import dump
from shared_reference_return import exact_identity_audit
import happo_batch_cpu as base


def analyze(root,original,out,replay=None):
    root,original,out=Path(root),Path(original),Path(out)
    plan=json.loads(Path(__file__).with_name("shared_reference_qualification.json").read_text())
    if any(json.loads((r/"execution.json").read_text())["state"]!="complete" for r in (root,original)):
        raise AssertionError("do not open incomplete outcomes")
    out.mkdir(parents=True,exist_ok=False)
    for r in (root,original):
        for name,h in json.loads((r/"hashes.json").read_text()).items():
            if base.sha(r/name)!=h:raise AssertionError("artifact hash mismatch")
    replay_ok=False
    if replay:
        replay_ok=all(p.read_bytes()==(Path(replay)/p.name).read_bytes() for p in sorted(root.glob("context_*.json")))
    ratios={t["name"]:[] for t in plan["tasks"]};compatible=True;positive=0;rows=[];covariances={}
    for index in range(8):
        p=json.loads((root/("context_%02d.json"%index)).read_text())
        old=json.loads((original/("context_%02d.json"%index)).read_text())
        if (p["reference_hash"]!=old["reference_hash"] or p["seed"]!=old["seed"] or p["stage"]!=old["stage"]):
            raise AssertionError("reference snapshot mismatch")
        ti=0 if p["task"]=="spread" else 1;n=3 if ti==0 else 2
        ids=[base.seed_for(p["seed"],ti,p["stage"],plan["bank_role"],e) for e in range(64)]
        forbidden={base.seed_for(p["seed"],ti,stage,110,e) for stage in range(1,17) for e in range(16)}
        forbidden.update(base.seed_for(p["seed"],ti,p["stage"],310,e) for e in range(256))
        forbidden.update(base.seed_for(p["seed"],ti,p["stage"],210+mi*10+ci,e) for mi in range(3) for ci in range(3) for e in range(64))
        if ids!=p["bank_identities"] or len(set(ids))!=64 or set(ids)&forbidden:
            raise AssertionError("bank seed mapping/leakage")
        if (p["bank_joint_transitions"]!=1600 or p["reference_policy_forward_rows"]!=2*n*1600 or
                p["critic_forward_rows"]!=1600 or p["validation_used_for_scoring"]):
            raise AssertionError("bank accounting")
        scores=[];primary_samples=[]
        for r,prior in zip(p["candidates"],old["candidates"]):
            if r["candidate_hash"]!=prior["candidate_hash"] or r["scale"]!=prior["scale"] or r["candidate_policy_forward_rows"]!=n*1600:
                raise AssertionError("candidate mismatch/accounting")
            y=np.asarray(prior["validation"]);ym=float(y.mean());yse=float(y.std(ddof=1)/16)
            prior_var=float(np.var(prior["samples"]["first_disagreement"],ddof=1))
            prior_cost=prior["ledgers"]["first_disagreement"]["joint_transitions"]/64
            for method,xs in r["samples"].items():
                x=np.asarray(xs)
                if len(x)!=64 or not np.isfinite(x).all():raise AssertionError("invalid sample")
                variance=float(x.var(ddof=1));mean=float(x.mean());se=math.sqrt(variance/64)
                ratio=(variance*(25/3))/(max(prior_var*prior_cost,1e-30))
                ok=abs(mean-ym)<=4*math.sqrt(se*se+yse*yse)+1e-10
                rows.append({"context":index,"task":p["task"],"stage":p["stage"],"scale":r["scale"],"method":method,
                             "mean":mean,"se":se,"validation_mean":ym,"validation_se":yse,
                             "variance":variance,"amortized_cost_variance_ratio_vs_first_disagreement":ratio,
                             "compatible_4se":bool(ok),"weight_min":r["weight_min"],"weight_max":r["weight_max"]})
                if method==plan["primary_method"]:
                    ratios[p["task"]].append(max(ratio,1e-30));compatible &= ok
                    scores.append((mean-1.645*se,ym-1.645*yse));primary_samples.append(x)
        covariances[str(index)]=np.cov(np.asarray(primary_samples),ddof=1).tolist()
        selected=max(scores,key=lambda s:s[0]);positive+=int(selected[0]>0 and selected[1]>0)
    task_ratios={t:math.exp(float(np.log(v).mean())) for t,v in ratios.items()}
    gates={"integrity_snapshots_disjoint_bank_accounting":True,"registered_full_payload_replay":replay_ok,
           "exact_unbiasedness":exact_identity_audit()["maximum_absolute_error"]<1e-12,
           "primary_task_cost_variance":all(v<=plan["gates"]["primary_task_geomean_amortized_cost_variance_ratio_vs_first_disagreement_max"] for v in task_ratios.values()),
           "primary_truth_compatibility":bool(compatible),"positive_validated_scores":positive>=plan["gates"]["positive_score_and_validation_contexts_min"]}
    with (out/"metrics.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    result={"experiment":plan["id"],"pass":all(gates.values()),"gates":gates,
            "primary_task_geomean_cost_variance_ratio":task_ratios,"positive_validated_selected_contexts":positive,
            "shared_bank_transitions":8*1600,"first_disagreement_three_candidate_transitions":8*3*64*75,
            "primary_score_covariance":covariances,"rigorous_nonzero_return_certificate":False,
            "validation_reused_from_original_prospective_bank":True,"estimator_family_registered_before_original_outcomes":True,
            "interpretation":"Fixed primary TD control variate. Other estimators are reported, never post-selected as a replacement primary. Empirical scores are not simultaneous certificates."}
    dump(out/"summary.json",result);return result


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--original",required=True)
    ap.add_argument("--output",required=True);ap.add_argument("--replay")
    args=ap.parse_args();print(json.dumps(analyze(args.root,args.original,args.output,args.replay),indent=2))
