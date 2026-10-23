"""One-worker, locked operational qualification plus complete semantic replay."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
START_CPU = time.process_time()
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
import numpy as np
import torch
from integration import run_cell, base
from audit import analyze, check

HERE = Path(__file__).resolve().parent
ROOT = base.ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    with Path(path).open("w",encoding="utf-8",newline="\n") as file:
        file.write(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")


def state():
    files = list(HERE.glob("*.py")) + [HERE/"plan.json"]
    files += [HERE.parent/x for x in ("current_packet_value_control.py","amortized_packet_value.py","current_packet_native/integration.py","happo_batch_cpu.py",
        "happo_batch_cpu_preregistration.json","shared_reference_return.py","finite_horizon_return_control.py")]
    files += [HERE.parent/"causal_recycle"/x for x in ("native.py","packet.py")]
    runtime = base.lock_state()
    return dict(files={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted(files)},
        harl_commit=runtime["harl_commit"],mpe_runtime_tree_sha256=runtime["mpe_runtime_tree_sha256"],
        numpy=np.__version__,torch=torch.__version__)


def main(args):
    plan = json.loads((HERE/"plan.json").read_text())
    assert np.__version__ == plan["required_numpy"] and torch.__version__ == plan["required_torch"]
    if args.freeze:
        if Path(args.freeze).exists():
            raise FileExistsError(args.freeze)
        dump(args.freeze,state()); return
    lock = json.loads(Path(args.lock).read_text())
    assert state() == lock
    for relative in [*lock["files"], str(Path(args.lock).resolve().relative_to(ROOT)).replace("\\", "/"),
                     "ICML-2027/internal/amortized_packet_native_qualification_preregistration_20261016.md"]:
        subprocess.check_call(["git","cat-file","-e",f"HEAD:{relative}"],stdout=subprocess.DEVNULL)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    root = Path(args.output); root.mkdir(parents=True,exist_ok=False)
    replay = Path(str(root)+"_replay")
    a, b = Path(str(root)+"_analysis_a"), Path(str(root)+"_analysis_b")
    execution = dict(state="running",phase="qualification",pid=os.getpid(),command=sys.argv,
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_lock_sha256=sha(args.lock),source_lock=lock,started_unix=time.time(),cells_complete=0)
    dump(root/"execution.json",execution)

    def budget():
        if time.process_time()-START_CPU > plan["cpu_seconds_limit"]:
            raise TimeoutError("registered qualification+replay active CPU limit")

    try:
        for directory, phase in ((root,"qualification"),(replay,"replay")):
            if directory != root:
                directory.mkdir(parents=True,exist_ok=False)
            execution.update(phase=phase,cells_complete=0)
            dump(root/"execution.json",execution)
            hashes = {}
            for ti,task in enumerate(plan["tasks"]):
                for method in plan["methods"]:
                    budget(); assert state() == lock
                    before = time.process_time()
                    row = run_cell(task,ti,plan["seed"],method,plan,budget)
                    row["cpu_seconds"] = time.process_time()-before
                    name=f'{task["name"]}_{plan["seed"]}_{method}.json'
                    dump(directory/name,row);hashes[name]=sha(directory/name)
                    # Verify each complete payload without ranking any return.
                    check(row,plan)
                    execution["cells_complete"] += 1
                    progress=dict(phase=phase,cells_complete=execution["cells_complete"],
                                  last_cell=name,cpu_seconds=time.process_time()-START_CPU)
                    dump(directory/"progress.json",progress);print(json.dumps(progress),flush=True)
            dump(directory/"hashes.json",hashes)
        execution.update(phase="analysis");dump(root/"execution.json",execution)
        analyze(root,replay,plan,a,dump,sha);analyze(root,replay,plan,b,dump,sha)
        for name in ("decision.json","integrity.json","resource_counts.csv"):
            assert (a/name).read_bytes() == (b/name).read_bytes()
        budget();assert state() == lock
        execution.update(state="complete",phase="complete",cpu_seconds=time.process_time()-START_CPU)
        dump(root/"execution.json",execution)
        dump(a/"workflow_complete.json",dict(decision="OPERATIONALLY_QUALIFIED",source_lock_sha256=sha(args.lock),
            cpu_seconds=time.process_time()-START_CPU,cells=6,replays=6,analyzer_byte_replay=True,
            performance_evidence=False,precision_usefulness_established=False))
    except BaseException as error:
        execution.update(state="failed",error=repr(error),cpu_seconds=time.process_time()-START_CPU)
        dump(root/"execution.json",execution)
        dump(root/"stop_decision.json",dict(decision="NO_GO_PAYLOAD",error=repr(error),retry_permitted=False))
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--freeze");parser.add_argument("--lock");parser.add_argument("--output")
    main(parser.parse_args())
