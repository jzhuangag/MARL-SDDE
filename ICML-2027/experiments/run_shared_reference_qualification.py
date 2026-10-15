"""Prospective shared-bank estimator extension on the same frozen contexts.

Registered before opening first-disagreement outcomes. New independent bank
seeds; no new candidate tuning or changes to the original qualification gates.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
os.environ.setdefault("SDL_AUDIODRIVER","dummy")
import numpy as np
import torch
import happo_batch_cpu as base
from shared_reference_return import contrasts,exact_identity_audit
from run_relative_return_qualification import actor_hash,dump

HERE=Path(__file__).resolve().parent


def collect(actors,task,ti,seed,stage,plan,p):
    count,h,n=plan["bank_samples"],p["episode_length"],len(actors)
    all_obs=[[] for _ in actors];all_actions=[[] for _ in actors]
    all_states=[];all_rewards=[];identities=[]
    for start in range(0,count,16):
        size=min(16,count-start)
        keys=[base.seed_for(seed,ti,stage,plan["bank_role"],e) for e in range(start,start+size)]
        envs=base.env_pool(task,h,size)
        obs=[env.reset(seed=key)[0] for env,key in zip(envs,keys)]
        names=list(envs[0].possible_agents)
        uniforms=np.stack([np.random.default_rng(key+1).random((h,n)) for key in keys])
        oo=[[] for _ in actors];aa=[[] for _ in actors];ss=[];rr=[]
        for t in range(h):
            ss.append(np.stack([env.state() for env in envs]))
            actions=[]
            for i,actor in enumerate(actors):
                ob=np.stack([o[names[i]] for o in obs]).astype(np.float32)
                with torch.no_grad():
                    pr=actor.actor.act.action_out(actor.actor.base(torch.as_tensor(ob))).probs.numpy().astype(float)
                pr/=pr.sum(axis=1,keepdims=True)
                ac=np.minimum((uniforms[:,t,i,None]>np.cumsum(pr,axis=1)).sum(axis=1),pr.shape[1]-1)
                oo[i].append(ob);aa[i].append(ac);actions.append(ac)
            next_obs=[];rew=[]
            for e,env in enumerate(envs):
                no,r,_,_,_=env.step({name:int(actions[i][e]) for i,name in enumerate(names)})
                next_obs.append(no);rew.append(float(np.mean(list(r.values()))))
            rr.append(rew);obs=next_obs
        all_states.append(np.asarray(ss).transpose(1,0,2));all_rewards.append(np.asarray(rr).T)
        for i in range(n):
            all_obs[i].append(np.asarray(oo[i]).transpose(1,0,2))
            all_actions[i].append(np.asarray(aa[i]).T)
        identities.extend(keys)
    return {"obs":[np.concatenate(x).reshape(count*h,-1) for x in all_obs],
            "actions":[np.concatenate(x).reshape(-1) for x in all_actions],
            "states":np.concatenate(all_states).reshape(count*h,-1),
            "rewards":np.concatenate(all_rewards),"identities":identities}


def log_probabilities(actors,data):
    result=np.zeros(len(data["states"]))
    with torch.no_grad():
        for i,actor in enumerate(actors):
            pr=actor.actor.act.action_out(actor.actor.base(torch.as_tensor(data["obs"][i]))).probs.numpy().astype(float)
            pr/=pr.sum(axis=1,keepdims=True)
            result+=np.log(pr[np.arange(len(pr)),data["actions"][i]])
    return result.reshape(data["rewards"].shape)


def run(args):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    plan=json.loads((HERE/"shared_reference_qualification.json").read_text())
    lock=json.loads(Path(args.lock).read_text())
    for name,h in lock["files"].items():
        if base.sha(base.ROOT/name)!=h:raise AssertionError("source changed: "+name)
    if args.qualification:
        plan.update(seeds=[75109996],stages=[1],bank_samples=2)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    dump(out/"execution.json",{"state":"running","lock":lock})
    p=base.config();p.update(episode_length=25,episodes_per_training_block=16)
    started=time.process_time();index=0
    for ti,task in enumerate(plan["tasks"]):
        for seed in plan["seeds"]:
            actors,critic,normalizer=base.initialize(task,seed,p)
            for stage in range(1,max(plan["stages"])+1):
                data=base.rollouts(actors,task,ti,seed,stage,110,16,p,training=True)
                base.add_advantages(data,critic,normalizer,p)
                old=copy.deepcopy(actors) if stage in plan["stages"] else None
                base.sequential_update(actors,data,list(range(len(actors))),p)
                base.train_critic(critic,normalizer,data,p)
                if stage not in plan["stages"]:continue
                bank=collect(actors,task,ti,seed,stage,plan,p)
                ref_log=log_probabilities(actors,bank)
                count=len(bank["states"])
                zeros=np.zeros((count,1,32),np.float32);ones=np.ones((count,1),np.float32)
                with torch.no_grad():
                    raw=critic.get_values(bank["states"],zeros,ones)[0].numpy()
                    values=np.asarray(normalizer.denormalize(raw)).reshape(bank["rewards"].shape)
                m=plan["bank_samples"]
                row={"context":index,"task":task["name"],"seed":seed,"stage":stage,
                     "reference_hash":actor_hash(actors),"bank_identities":bank["identities"],
                     "bank_joint_transitions":m*25,"reference_policy_forward_rows":2*len(actors)*m*25,
                     "critic_forward_rows":m*25,"training_joint_transitions":stage*16*25,
                     "validation_used_for_scoring":False,"candidates":[]}
                for scale in plan["candidate_scales"]:
                    candidate=copy.deepcopy(actors)
                    with torch.no_grad():
                        for a,b,c in zip(old,actors,candidate):
                            for x,y,z in zip(a.actor.parameters(),b.actor.parameters(),c.actor.parameters()):
                                z.copy_(x+scale*(y-x))
                    lr=log_probabilities(candidate,bank)-ref_log
                    estimates,w=contrasts(lr,bank["rewards"],values)
                    row["candidates"].append({"scale":scale,"candidate_hash":actor_hash(candidate),
                        "samples":{m:x.tolist() for m,x in estimates.items()},
                        "candidate_policy_forward_rows":len(actors)*m*25,
                        "weight_min":float(w.min()),"weight_max":float(w.max())})
                null,_=contrasts(ref_log-ref_log,bank["rewards"],values)
                if any(np.any(x!=0) for x in null.values()):raise AssertionError("identity failed")
                dump(out/("context_%02d.json"%index),row)
                index+=1;print(json.dumps({"completed_context":index,"cpu_seconds":time.process_time()-started}),flush=True)
                if time.process_time()-started>plan["active_cpu_limit_seconds"]:raise TimeoutError("CPU limit")
    dump(out/"exact_identity.json",exact_identity_audit())
    dump(out/"hashes.json",{p.name:base.sha(p) for p in sorted(out.glob("context_*.json"))})
    dump(out/"execution.json",{"state":"complete","lock":lock,"contexts":index,"cpu_seconds":time.process_time()-started})


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);ap.add_argument("--lock",required=True)
    ap.add_argument("--qualification",action="store_true")
    run(ap.parse_args())
