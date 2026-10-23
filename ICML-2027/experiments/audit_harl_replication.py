"""CPU-only operational audit of the pinned HARL critic's replication law.

Synthetic fixed buffers test an actual optimizer, without an environment,
benchmark return, model selection, or a claim of MARL performance.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import gym
import numpy as np
import torch
import yaml


PIN="b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"
ROOT=Path(__file__).resolve().parents[2]


def run(out:Path) -> None:
    upstream=ROOT/"tmp"/"HARL"
    head=subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(upstream),"status","--porcelain"],text=True).strip()
    if head!=PIN or dirty:
        raise RuntimeError("audit requires the exact clean pinned HARL source")
    if out.exists():
        raise FileExistsError("refusing to overwrite audit artifacts")
    sys.path.insert(0,str(upstream))
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.valuenorm import ValueNorm
    config=yaml.safe_load((upstream/"harl"/"configs"/"algos_cfgs"/"mappo.yaml").read_text())
    args={**config["model"],**config["algo"]}
    args.update({"hidden_sizes":[128,128],"activation_func":"tanh","critic_epoch":5})
    torch.set_num_threads(1)
    torch.manual_seed(61000001)
    rng=np.random.default_rng(61000001)
    obs=rng.uniform(-1,1,size=(200,12)).astype(np.float32)
    targets=(np.sin(obs[:,0])+0.2*obs[:,1]).reshape(-1,1).astype(np.float32)
    space=gym.spaces.Box(-1,1,shape=(12,),dtype=np.float32)
    original=VCritic(args,space,torch.device("cpu"))
    initial_hash=hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in original.critic.parameters())).hexdigest()
    reference=None
    rows=[]
    for q in (1,2,4,8):
        critic=copy.deepcopy(original)
        normalizer=ValueNorm(1,device=torch.device("cpu"))
        inputs=np.tile(obs,(q,1))
        returns=np.tile(targets,(q,1))
        states=np.zeros((len(inputs),1,128),dtype=np.float32)
        masks=np.ones((len(inputs),1),dtype=np.float32)
        with torch.no_grad():
            old_values=critic.get_values(inputs,states,masks)[0].cpu().numpy()
        losses=[]
        for _ in range(5):
            loss,norm=critic.update((inputs,states,old_values,returns,masks),normalizer)
            losses.append(float(loss.detach()))
        parameters=np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in critic.critic.parameters()])
        if reference is None:
            reference=parameters.copy()
        rows.append({"q":q,"base_observations":200,"stored_rows":len(inputs),"optimizer_steps":5,
                     "initial_parameters_sha256":initial_hash,"losses":losses,
                     "max_parameter_difference_from_q1":float(np.max(np.abs(parameters-reference)))})
    result={"role":"synthetic-buffer operational audit; not MARL return evidence", "harl_commit":head,
            "torch_version":torch.__version__,"device":"cpu","seed":61000001,
            "all_initializations_identical":len({x["initial_parameters_sha256"] for x in rows})==1,
            "replication_within_tolerance":all(x["max_parameter_difference_from_q1"]<=1e-5 for x in rows),
            "tolerance":1e-5,"rows":rows,
            "interpretation":"If replication is invariant, a shared-stream interior q optimum cannot be attributed to duplicated batch size improving this critic update alone. Actor dynamics, data differences, normalization history, or finite training horizon must be measured separately."}
    out.mkdir(parents=True)
    (out/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
