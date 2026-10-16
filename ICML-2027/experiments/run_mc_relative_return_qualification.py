"""Fresh-context, fully charged qualification of a finite-horizon potential."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
os.environ.setdefault("SDL_VIDEODRIVER","dummy");os.environ.setdefault("SDL_AUDIODRIVER","dummy")
import numpy as np
import torch
import happo_batch_cpu as base
import relative_return_estimator as estimator
from relative_return_a1.stable_overlap import stable_overlap
from run_relative_return_qualification import dump,actor_hash,estimate
from run_shared_reference_qualification import collect,log_probabilities
from shared_reference_return import contrasts,exact_identity_audit
from finite_horizon_return_control import MonteCarloPotential

estimator.overlap=stable_overlap
HERE=Path(__file__).resolve().parent


def run(args):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    plan=json.loads((HERE/"mc_relative_return_qualification.json").read_text())
    lock=json.loads(Path(args.lock).read_text())
    for name,h in lock["files"].items():
        if base.sha(base.ROOT/name)!=h:raise AssertionError("source lock changed: "+name)
    if args.qualification:
        plan.update(seeds=[78109999],stages=[1],bank_samples=2,validation_samples=2,branch_samples=2)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    dump(out/"execution.json",{"state":"running","lock":lock})
    p=base.config();p.update(episode_length=25,episodes_per_training_block=16)
    started=time.process_time();index=0
    for ti,task in enumerate(plan["tasks"]):
        for seed in plan["seeds"]:
            actors,critic,normalizer=base.initialize(task,seed,p)
            history=[]
            for stage in range(1,max(plan["stages"])+1):
                data=base.rollouts(actors,task,ti,seed,stage,110,16,p,training=True)
                history.append({"states":data["states"].copy(),"rewards":data["rewards"].copy()})
                base.add_advantages(data,critic,normalizer,p)
                old=copy.deepcopy(actors) if stage in plan["stages"] else None
                base.sequential_update(actors,data,list(range(len(actors))),p)
                base.train_critic(critic,normalizer,data,p)
                if stage not in plan["stages"]:continue
                if args.context is not None and args.context!=index:
                    index+=1;continue
                potential=MonteCarloPotential(ridge=plan["ridge"]).fit(history)
                banks=[]
                for m,role in ((plan["bank_samples"],610),(plan["validation_samples"],710)):
                    bank=collect(actors,task,ti,seed,stage,{"bank_samples":m,"bank_role":role},p)
                    banks.append((bank,log_probabilities(actors,bank),potential.predict(bank["states"])))
                row={"context":index,"task":task["name"],"seed":seed,"stage":stage,"reference_hash":actor_hash(actors),
                     "bank_identities":[b[0]["identities"] for b in banks],
                     "score_bank_joint_transitions":plan["bank_samples"]*25,
                     "validation_bank_joint_transitions":plan["validation_samples"]*25,
                     "training_joint_transitions":stage*16*25,
                     "fit_rows":stage*16*25,"fit_feature_count":len(potential.coef),
                     "potential_coefficient_norm":float(np.linalg.norm(potential.coef)),
                     "evaluation_used_by_fitting":False,"candidates":[]}
                for ci,scale in enumerate(plan["candidate_scales"]):
                    candidate=copy.deepcopy(actors)
                    with torch.no_grad():
                        for a,b,c in zip(old,actors,candidate):
                            for x,y,z in zip(a.actor.parameters(),b.actor.parameters(),c.actor.parameters()):z.copy_(x+scale*(y-x))
                    cr={"scale":scale,"candidate_hash":actor_hash(candidate)}
                    for name,(bank,ref_log,values) in zip(("score","validation"),banks):
                        xx,w=contrasts(log_probabilities(candidate,bank)-ref_log,bank["rewards"],values)
                        cr[name]=xx["td_control_variate"].tolist()
                        cr[name+"_weight_max"]=float(w.max())
                    direct=[];aux=[];cost={"joint_transitions":0,"policy_forward_rows":0}
                    for e in range(plan["branch_samples"]):
                        key=base.seed_for(seed,ti,stage,810+ci,e)
                        x,ledger,extra=estimate(candidate,actors,task,key,"first_disagreement",25)
                        direct.append(x);aux.append(extra)
                        for k in cost:cost[k]+=ledger[k]
                    cr["direct_branch"]=direct;cr["direct_ledger"]=cost;cr["direct_aux"]=aux
                    row["candidates"].append(cr)
                n=len(actors);total=(plan["bank_samples"]+plan["validation_samples"])*25
                row["bank_policy_forward_rows"]=(2+len(plan["candidate_scales"]))*n*total
                row["potential_forward_rows"]=total
                dump(out/("context_%02d.json"%index),row)
                index+=1;print(json.dumps({"completed_context":index,"cpu_seconds":time.process_time()-started}),flush=True)
                if time.process_time()-started>plan["cpu_limit_seconds"]:raise TimeoutError("CPU limit")
                if args.context is not None:break
            if args.context is not None and index>args.context:break
        if args.context is not None and index>args.context:break
    dump(out/"exact_identity.json",exact_identity_audit())
    dump(out/"hashes.json",{x.name:base.sha(x) for x in sorted(out.glob("context_*.json"))})
    dump(out/"execution.json",{"state":"complete","lock":lock,"contexts":len(list(out.glob('context_*.json'))),"cpu_seconds":time.process_time()-started})


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);ap.add_argument("--lock",required=True)
    ap.add_argument("--qualification",action="store_true");ap.add_argument("--context",type=int)
    run(ap.parse_args())
