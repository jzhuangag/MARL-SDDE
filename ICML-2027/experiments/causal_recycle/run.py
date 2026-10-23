"""Locked one-worker execution; qualification outputs are outcome-ineligible."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
os.environ.setdefault("SDL_VIDEODRIVER","dummy");os.environ.setdefault("SDL_AUDIODRIVER","dummy")
import numpy as np
import torch
from engine import run_cell,base

HERE=Path(__file__).resolve().parent


def dump(path,obj):
    text=json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n"
    with Path(path).open("w",encoding="utf-8",newline="\n") as f:f.write(text)


def lock():
    files=list(HERE.glob("*.py"))+[HERE/"plan.json"]
    files += [HERE.parent/n for n in ("happo_batch_cpu.py","shared_reference_return.py","finite_horizon_return_control.py")]
    runtime=base.lock_state()
    return {"files":{str(p.relative_to(base.ROOT)).replace("\\","/"):base.sha(p) for p in sorted(files)},
        "harl_commit":runtime["harl_commit"],"mpe_runtime_tree_sha256":runtime["mpe_runtime_tree_sha256"],
        "numpy_version":np.__version__,"torch_version":torch.__version__}


def main(args):
    if args.create_lock:
        if Path(args.create_lock).exists():raise FileExistsError(args.create_lock)
        dump(args.create_lock,lock());return
    frozen=json.loads(Path(args.lock).read_text())
    if frozen!=lock():raise AssertionError("source lock mismatch")
    if subprocess.check_output(["git","-C",str(base.UPSTREAM),"status","--porcelain"],text=True).strip():raise AssertionError("dirty HARL")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    plan=json.loads((HERE/"plan.json").read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    started=time.process_time();complete=0
    execution={"state":"running","qualification":args.qualification,"lock":frozen,"pid":os.getpid(),
        "command":__import__("sys").argv,"started_unix":time.time(),"cells_complete":0}
    dump(out/"execution.json",execution)
    seeds=[plan["qualification_seed"]] if args.qualification else plan["seeds"]
    methods=plan["methods"]
    if args.replay:methods=["happo_1.0","async_latest","recycle"];seeds=seeds[:1]
    try:
        for ti,task in enumerate(plan["tasks"]):
            for seed in seeds:
                for method in methods:
                    row=run_cell(task,ti,seed,method,plan,qualification=args.qualification)
                    name=f'{task["name"]}_{seed}_{method}.json'
                    dump(out/name,row);complete+=1
                    dump(out/"progress.json",{"cells_complete":complete,"last_cell":name,"cpu_seconds":time.process_time()-started})
                    print(json.dumps({"cells_complete":complete,"last_cell":name,"cpu_seconds":time.process_time()-started}),flush=True)
                    limit=plan["qualification_cpu_seconds_limit"] if args.qualification else plan["cpu_seconds_limit"]
                    if time.process_time()-started>limit:raise TimeoutError("registered active CPU limit")
        dump(out/"hashes.json",{p.name:base.sha(p) for p in sorted(out.glob("*.json")) if p.name not in ("execution.json","progress.json","hashes.json")})
        execution.update(state="complete",cells_complete=complete,cpu_seconds=time.process_time()-started)
        dump(out/"execution.json",execution)
    except BaseException as exc:
        execution.update(state="failed",cells_complete=complete,error=repr(exc),cpu_seconds=time.process_time()-started)
        dump(out/"execution.json",execution);raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--create-lock");p.add_argument("--lock");p.add_argument("--output")
    p.add_argument("--qualification",action="store_true");p.add_argument("--replay",action="store_true");main(p.parse_args())
