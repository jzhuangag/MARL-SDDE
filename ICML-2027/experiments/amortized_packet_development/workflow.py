"""One bounded locked complete-algorithm development; never retries a cell."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
START_CPU=time.process_time()
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
os.environ["SDL_VIDEODRIVER"]="dummy";os.environ["SDL_AUDIODRIVER"]="dummy"
import numpy as np
import torch
from integration import run_cell,base
from amortized_packet_development.audit import check
from amortized_packet_development.analyze import analyze,plots

HERE=Path(__file__).resolve().parent;ROOT=base.ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    with Path(path).open("w",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")


def state():
    previous=json.loads((ROOT/"ICML-2027/internal/amortized_packet_native_lock_20261016.json").read_text())
    files={ROOT/p for p in previous["files"]}
    files.update(HERE.glob("*.py"));files.add(HERE/"plan.json")
    files.add(HERE.parent/"test_amortized_development_protocol.py")
    runtime=base.lock_state()
    return dict(files={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted(files)},
        harl_commit=runtime["harl_commit"],mpe_runtime_tree_sha256=runtime["mpe_runtime_tree_sha256"],
        numpy=np.__version__,torch=torch.__version__)


def main(args):
    plan=json.loads((HERE/"plan.json").read_text())
    assert np.__version__==plan["required_numpy"] and torch.__version__==plan["required_torch"]
    if args.freeze:
        if Path(args.freeze).exists():raise FileExistsError(args.freeze)
        dump(args.freeze,state());return
    lock=json.loads(Path(args.lock).read_text());assert state()==lock
    for relative in [*lock["files"],str(Path(args.lock).resolve().relative_to(ROOT)).replace("\\","/"),
                     "ICML-2027/internal/amortized_packet_development_preregistration_20261016.md"]:
        subprocess.check_call(["git","cat-file","-e",f"HEAD:{relative}"],stdout=subprocess.DEVNULL)
    qualified=Path(args.qualification)
    q=json.loads((qualified/"workflow_complete.json").read_text())
    assert q["decision"]=="OPERATIONALLY_QUALIFIED" and q["cells"]==q["replays"]==6 and q["analyzer_byte_replay"]
    eval_count=1+len([t for t in range(1,plan["frames"]+1) if t%plan["evaluation_every"]==0 or t==plan["frames"]])
    native_volume=plan["frames"]*plan["bank_episodes"]*plan["horizon"]+2*eval_count*plan["evaluation_episodes"]*plan["horizon"]
    estimate=q["cpu_seconds"]*(plan["development_cells"]+plan["replay_cells"])/12*native_volume/3500*plan["runtime_margin"]
    assert estimate<=plan["cpu_seconds_limit"]<=12000
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    replay=Path(str(root)+"_replay");a=Path(str(root)+"_analysis_a");b=Path(str(root)+"_analysis_b")
    execution=dict(state="running",phase="development",pid=os.getpid(),command=sys.argv,
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        lock_sha256=sha(args.lock),source_lock=lock,qualification_sha256=sha(qualified/"workflow_complete.json"),
        estimated_cpu_seconds=estimate,cpu_cap_seconds=plan["cpu_seconds_limit"],cells_complete=0)
    dump(root/"execution.json",execution)
    def budget():
        if time.process_time()-START_CPU>plan["cpu_seconds_limit"]:
            raise TimeoutError("registered complete development/replay active CPU bound")
    try:
        for directory,phase in ((root,"development"),(replay,"replay")):
            if directory!=root:directory.mkdir(parents=True,exist_ok=False)
            hashes={};execution.update(phase=phase,cells_complete=0);dump(root/"execution.json",execution)
            for ti,task in enumerate(plan["tasks"]):
                for seed in (plan["seeds"] if phase=="development" else [plan["seeds"][-1]]):
                    for method in (plan["methods"] if phase=="development" else plan["replay_methods"]):
                        budget();assert state()==lock
                        before=time.process_time();row=run_cell(task,ti,seed,method,plan,budget)
                        row["cpu_seconds"]=time.process_time()-before;row["source_lock_sha256"]=sha(args.lock)
                        name=f'{task["name"]}_{seed}_{method}.json'
                        dump(directory/name,row);hashes[name]=sha(directory/name)
                        check(row,plan)
                        execution["cells_complete"]+=1
                        progress=dict(phase=phase,cells_complete=execution["cells_complete"],last_cell=name,cpu_seconds=time.process_time()-START_CPU)
                        dump(directory/"progress.json",progress);print(json.dumps(progress),flush=True)
            dump(directory/"hashes.json",hashes)
        execution.update(phase="analysis");dump(root/"execution.json",execution)
        analyze(root,replay,plan,a,dump,sha);analyze(root,replay,plan,b,dump,sha)
        for name in ("metrics.csv","summary.json","integrity.json"):
            assert (a/name).read_bytes()==(b/name).read_bytes()
        plots(root,plan,a);budget();assert state()==lock
        decision=json.loads((a/"summary.json").read_text())["decision"]
        execution.update(state="complete",phase="complete",cpu_seconds=time.process_time()-START_CPU)
        dump(root/"execution.json",execution)
        dump(root/"workflow_complete.json",dict(decision=decision,cells=plan["development_cells"],replays=plan["replay_cells"],
            analyzer_byte_replay=True,cpu_seconds=time.process_time()-START_CPU,source_lock_sha256=sha(args.lock),
            manuscript_admission=False,icml_ready=False))
    except BaseException as error:
        execution.update(state="failed",error=repr(error),cpu_seconds=time.process_time()-START_CPU)
        dump(root/"execution.json",execution)
        dump(root/"stop_decision.json",dict(decision="NO_GO_PAYLOAD",error=repr(error),retry_permitted=False))
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--freeze");p.add_argument("--lock");p.add_argument("--qualification");p.add_argument("--output")
    main(p.parse_args())
