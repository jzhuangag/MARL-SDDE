"""Frozen complete-learning development + selected replay + double analysis.

Reuses the qualified engine unchanged. Only this prospectively frozen stage's
seeds, evaluation cadence and technical CPU cap differ from qualification.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
os.environ.setdefault("SDL_VIDEODRIVER","dummy");os.environ.setdefault("SDL_AUDIODRIVER","dummy")
HERE=Path(__file__).resolve().parent
QUAL=HERE.parent/"causal_recycle"
sys.path.insert(0,str(QUAL))
import numpy as np
import torch
import run as qualified_run
import analyze as qualified_analysis
from engine import run_cell,base


def source_lock():
    return {"qualified":qualified_run.lock(),"stage_files":{str(p.relative_to(base.ROOT)).replace("\\","/"):base.sha(p) for p in sorted(HERE.glob("*")) if p.suffix in (".py",".json")}}


def execute(root,plan,lock,*,replay,started):
    root.mkdir(parents=True,exist_ok=False)
    execution={"state":"running","qualification":False,"replay":replay,"lock":lock,"pid":os.getpid(),
        "command":sys.argv,"started_unix":time.time(),"cells_complete":0,
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=base.ROOT,text=True).strip()}
    qualified_run.dump(root/"execution.json",execution);names=[]
    seeds=plan["seeds"][:1] if replay else plan["seeds"]
    methods=["happo_1.0","async_latest","recycle"] if replay else plan["methods"]
    try:
        for ti,task in enumerate(plan["tasks"]):
            for seed in seeds:
                for method in methods:
                    row=run_cell(task,ti,seed,method,plan)
                    if source_lock()!=lock:raise AssertionError("frozen sources changed during cell")
                    name=f'{task["name"]}_{seed}_{method}.json';qualified_run.dump(root/name,row);names.append(name)
                    progress={"cells_complete":len(names),"last_cell":name,"workflow_cpu_seconds":time.process_time()-started}
                    qualified_run.dump(root/"progress.json",progress);print(json.dumps(progress),flush=True)
                    if time.process_time()-started>plan["cpu_seconds_limit"]:raise TimeoutError("registered workflow CPU limit")
        qualified_run.dump(root/"hashes.json",{name:base.sha(root/name) for name in sorted(names)})
        execution.update(state="complete",cells_complete=len(names),workflow_cpu_seconds=time.process_time()-started)
        qualified_run.dump(root/"execution.json",execution)
    except BaseException as exc:
        execution.update(state="failed",cells_complete=len(names),error=repr(exc));qualified_run.dump(root/"execution.json",execution);raise


def analyze(root,replay,out,plan):
    # The qualified analyzer already takes its plan from HERE. Redirect just
    # that explicit configuration location; all validation/metric code is intact.
    qualified_analysis.HERE=HERE
    qualified_analysis.main(argparse.Namespace(root=str(root),replay_root=str(replay),output=str(out)))
    decision=json.loads((out/"decision.json").read_text())
    with (out/"metrics.csv").open(newline="",encoding="utf-8") as f:metrics=list(csv.DictReader(f))
    for task in decision["tasks"]:
        rows=[r for r in metrics if r["task"]==task["task"] and r["method"]==task["strongest_baseline"]]
        change=np.mean([(float(r["final_return"])-float(r["initial_return"]))/max(abs(float(r["initial_return"])),1) for r in rows])
        task["best_baseline_mean_gain_from_initial"]=float(change)
        task["baseline_learning_signal_pass"]=bool(change>=plan["mandatory_gates"]["both_tasks_baseline_gain_from_initial_min"])
        task["pass"]=task["pass"] and task["baseline_learning_signal_pass"]
    decision["decision"]="DEVELOPMENT_PASS" if all(t["pass"] for t in decision["tasks"]) and decision["replay_checked"]==6 else "NO_GO"
    qualified_run.dump(out/"decision.json",decision)
    return decision


def main(args):
    if args.create_lock:
        if Path(args.create_lock).exists():raise FileExistsError(args.create_lock)
        qualified_run.dump(args.create_lock,source_lock());return
    lock=json.loads(Path(args.lock).read_text());assert lock==source_lock(),"development source lock mismatch"
    # Require the completed qualification and its complete replay before science.
    qualification=json.loads(Path(args.qualification_decision).read_text())
    assert qualification["decision"]=="QUALIFIED" and qualification["replay_cells"]==20
    assert lock["qualified"]==json.loads((base.ROOT/"ICML-2027/internal/causal_recycle_qualification_lock_a1_20261016.json").read_text())
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    plan=json.loads((HERE/"plan.json").read_text());started=time.process_time()
    root=Path(args.output);replay=Path(args.output+"_replay");a=Path(args.output+"_analysis_a");b=Path(args.output+"_analysis_b")
    for path in (root,replay,a,b,Path(args.output+"_plots")):
        if path.exists():raise FileExistsError(path)
    execute(root,plan,lock,replay=False,started=started)
    execute(replay,plan,lock,replay=True,started=started)
    if source_lock()!=lock:raise AssertionError("frozen sources changed before analysis")
    decision=analyze(root,replay,a,plan);analyze(root,replay,b,plan)
    for name in ("metrics.csv","decision.json","integrity.json"):
        if (a/name).read_bytes()!=(b/name).read_bytes():raise AssertionError("analyzer byte replay failed: "+name)
    import plot as qualified_plot
    qualified_plot.HERE=HERE;qualified_plot.main(argparse.Namespace(root=str(root),output=args.output+"_plots"))
    lines=["# Complete causal-recycle development decision","","Material Passport: deterministic-replay-verified development, not confirmation.","",decision["decision"],""]
    for t in decision["tasks"]:
        lines.append(f'- {t["task"]}: strongest control {t["strongest_baseline"]}; mean native AUC gain {t["mean_auc_gain"]:.6%}; positive seed fraction {t["positive_seed_fraction"]:.3f}; final-return change {t["mean_final_return_change"]:.6%}; common-range communication AUC gain {t["mean_common_range_communication_auc_gain"]:.6%}; baseline learning gain {t["best_baseline_mean_gain_from_initial"]:.6%}; pass={t["pass"]}.')
    lines.extend(["","All 60 registered learning cells and six selected replay cells are preserved.","Both analyzers are byte-identical. All methods and complete resource costs remain visible.","A failure requires full-mechanism diagnosis, not relabeling estimator gains as learning success."])
    with (a/"decision.md").open("w",encoding="utf-8",newline="\n") as f:f.write("\n".join(lines)+"\n")
    qualified_run.dump(a/"workflow_complete.json",{"decision":decision["decision"],"analyzer_byte_replay":True,
        "workflow_cpu_seconds":time.process_time()-started,"source_lock":lock,"manuscript_admission":False})
    print(json.dumps({"complete_decision":decision["decision"],"tasks":decision["tasks"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--create-lock");p.add_argument("--lock");p.add_argument("--output");p.add_argument("--qualification-decision");main(p.parse_args())
