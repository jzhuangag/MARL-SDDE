"""Multiple in-transit gradient packets, one-use receipt and server-owned Adam.

The asynchronous mechanism is network/transport delay, not a claim that one
physical compute worker simultaneously runs several unfinished gradient jobs.
Source gradients are computed before entering the independently delayed link.
"""
import copy
import hashlib
import numpy as np
import torch


def tensor_bytes(tensors):return sum(x.numel()*x.element_size() for x in tensors)


def digest(tensors):
    h=hashlib.sha256()
    for x in tensors:h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def source_gradient(actor,data,i):
    """One clipped PPO backpropagation, with no worker optimizer step."""
    count=len(data["obs"][i]);hidden=actor.actor.hidden_sizes[-1]
    zero=np.zeros((count,1,hidden),np.float32);ones=np.ones((count,1),np.float32)
    log,entropy,_=actor.evaluate_actions(data["obs"][i],zero,data["actions"][i],ones,None,ones)
    old=torch.as_tensor(data["logs"][i],dtype=torch.float32)
    adv=torch.as_tensor(data["advantages"],dtype=torch.float32)
    ratio=torch.exp(log-old)
    loss=-torch.minimum(ratio*adv,ratio.clamp(1-actor.clip_param,1+actor.clip_param)*adv).mean()-actor.entropy_coef*entropy
    actor.actor_optimizer.zero_grad();loss.backward()
    if actor.use_max_grad_norm:torch.nn.utils.clip_grad_norm_(actor.actor.parameters(),actor.max_grad_norm)
    gradient=[p.grad.detach().clone() for p in actor.actor.parameters()]
    if any(not torch.isfinite(g).all() for g in gradient):raise FloatingPointError("nonfinite raw gradient")
    return gradient


def adam_variant(server_actor,gradient,alpha):
    """Counterfactual Adam state derives only from the CURRENT server state."""
    if alpha==0:return server_actor
    result=copy.deepcopy(server_actor)
    original=[g["lr"] for g in result.actor_optimizer.param_groups]
    for group,lr in zip(result.actor_optimizer.param_groups,original):group["lr"]=lr*alpha
    result.actor_optimizer.zero_grad()
    for param,grad in zip(result.actor.parameters(),gradient):param.grad=grad.clone()
    result.actor_optimizer.step()
    for group,lr in zip(result.actor_optimizer.param_groups,original):group["lr"]=lr
    return result


class NetworkPackets:
    def __init__(self,seed,agents):
        self.seed=seed;self.agents=agents;self.jobs=[];self.used=set();self.events=[]
        self.ordinals=[0]*agents;self.versions=[0]*agents
        self.reserved_gradient_bytes=0;self.transmitted_gradient_bytes=0

    def launch(self,tick,agent,source_versions,gradient,delay_min,delay_max):
        ordinal=self.ordinals[agent];self.ordinals[agent]+=1
        word=hashlib.sha256(f"{self.seed}:{agent}:{ordinal}".encode()).digest()
        delay=delay_min+int.from_bytes(word[:8],"big")%(delay_max-delay_min+1)
        job={"id":len(self.jobs),"agent":agent,"source_versions":tuple(source_versions),"launch":tick,
             "due":tick+delay,"gradient":gradient,"bytes":tensor_bytes(gradient),"gradient_hash":digest(gradient)}
        if any(old>new for old,new in zip(source_versions,self.versions)):raise AssertionError("future source version")
        self.jobs.append(job);self.reserved_gradient_bytes+=job["bytes"]
        self.events.append({k:v for k,v in job.items() if k!="gradient"})
        return job

    def receive_latest(self,tick):
        arrived=[j for j in self.jobs if j["id"] not in self.used and j["due"]<=tick]
        for job in arrived:
            self.used.add(job["id"]);self.transmitted_gradient_bytes+=job["bytes"]
        selected={}
        for job in arrived:
            i=job["agent"]
            if i not in selected or (job["launch"],job["id"])>(selected[i]["launch"],selected[i]["id"]):selected[i]=job
        return selected,arrived

    def commit(self,action):
        for i,alpha in enumerate(action):
            if alpha>0:self.versions[i]+=1

    def drain(self):
        if self.jobs:self.receive_latest(max(j["due"] for j in self.jobs))
        if len(self.used)!=len(self.jobs) or self.reserved_gradient_bytes!=self.transmitted_gradient_bytes:
            raise AssertionError("packet transmission/use accounting")
