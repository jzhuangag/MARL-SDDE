"""Exact categorical behavior banks and factorized joint-return feedback.

Only the neural forwards are factorized. Joint trajectory likelihoods retain
all cross-agent products; enumeration remains exponential in team size.
"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import happo_batch_cpu as base
from shared_reference_return import contrasts


def probabilities(actor,obs):
    with torch.no_grad():
        p=actor.actor.act.action_out(actor.actor.base(torch.as_tensor(obs))).probs.numpy().astype(float)
    p/=p.sum(axis=1,keepdims=True)
    if not np.isfinite(p).all() or np.any(p<=0):raise FloatingPointError("categorical support lost")
    return p


def collect(actors,task,ti,seed,frame,role,count,h):
    n=len(actors);obs_all=[[] for _ in actors];act_all=[[] for _ in actors];logs_all=[[] for _ in actors]
    states_all=[];rewards_all=[];identities=[]
    for start in range(0,count,16):
        size=min(16,count-start);keys=[base.seed_for(seed,ti,frame,role,e) for e in range(start,start+size)]
        envs=base.env_pool(task,h,size);obs=[e.reset(seed=k)[0] for e,k in zip(envs,keys)]
        names=list(envs[0].possible_agents)
        uniforms=np.stack([np.random.default_rng(k+1).random((h,n)) for k in keys])
        oo=[[] for _ in actors];aa=[[] for _ in actors];ll=[[] for _ in actors];ss=[];rr=[]
        for t in range(h):
            ss.append(np.stack([e.state() for e in envs]));actions=[]
            for i,actor in enumerate(actors):
                o=np.stack([x[names[i]] for x in obs]).astype(np.float32);p=probabilities(actor,o)
                a=np.minimum((uniforms[:,t,i,None]>np.cumsum(p,axis=1)).sum(axis=1),p.shape[1]-1)
                actions.append(a);oo[i].append(o);aa[i].append(a);ll[i].append(np.log(p[np.arange(size),a]))
            following=[];rewards=[]
            for j,e in enumerate(envs):
                ob,r,term,trunc,_=e.step({name:int(actions[i][j]) for i,name in enumerate(names)})
                if t<h-1 and (any(term.values()) or any(trunc.values())):raise AssertionError("early episode termination")
                following.append(ob);rewards.append(float(np.mean(list(r.values()))))
            obs=following;rr.append(rewards)
        states_all.append(np.asarray(ss).transpose(1,0,2));rewards_all.append(np.asarray(rr).T)
        for i in range(n):
            obs_all[i].append(np.asarray(oo[i]).transpose(1,0,2))
            act_all[i].append(np.asarray(aa[i]).T);logs_all[i].append(np.asarray(ll[i]).T)
        identities.extend(keys)
    return {"obs":[np.concatenate(x).reshape(count*h,-1) for x in obs_all],
            "actions":[np.concatenate(x).reshape(-1,1) for x in act_all],
            "logs":[np.concatenate(x).reshape(-1,1) for x in logs_all],
            "states":np.concatenate(states_all).reshape(count*h,-1).astype(np.float32),
            "rewards":np.concatenate(rewards_all),"identities":identities}


def take(data,m,h):
    return {"obs":[x[:m*h] for x in data["obs"]],"actions":[x[:m*h] for x in data["actions"]],
            "logs":[x[:m*h] for x in data["logs"]],"states":data["states"][:m*h],
            "rewards":data["rewards"][:m],"identities":data["identities"][:m]}


def menu_feedback(actors,variants,actions,data,potential,*,current_bank):
    m,h=data["rewards"].shape;n=len(actors);ratios=[];rows=0
    for i,actor in enumerate(actors):
        ac=data["actions"][i].reshape(-1)
        if current_bank:ref=data["logs"][i].reshape(-1)
        else:
            p=probabilities(actor,data["obs"][i]);ref=np.log(p[np.arange(len(p)),ac]);rows+=m*h
        local={0.:np.zeros((m,h))}
        for alpha in (.5,1.):
            if i not in variants:local[alpha]=np.zeros((m,h));continue
            p=probabilities(variants[i][alpha],data["obs"][i]);rows+=m*h
            local[alpha]=(np.log(p[np.arange(len(p)),ac])-ref).reshape(m,h)
        ratios.append(local)
    values=np.zeros((m,h)) if potential is None else potential.predict(data["states"])
    samples=[];weight_max=1.
    for action in actions:
        lr=sum(ratios[i][alpha] for i,alpha in enumerate(action))
        estimate,w=contrasts(lr,data["rewards"],values)
        samples.append(estimate["td_control_variate"]);weight_max=max(weight_max,float(w.max()))
    result=np.stack(samples,axis=1)
    if not np.isfinite(result).all():raise FloatingPointError("nonfinite return feedback")
    return result,rows,weight_max
