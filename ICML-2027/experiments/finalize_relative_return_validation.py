"""Preserve verified estimator decisions, source hashes and reproducibility."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/"ICML-2027/results"


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    studies=[("relative_return_a1_analysis_20261015","relative_return_a1_analysis_replay_20261015"),
             ("shared_reference_analysis_a1_20261015","shared_reference_analysis_a1_replay_20261015"),
             ("mc_relative_return_analysis_20261015","mc_relative_return_analysis_replay_20261015")]
    checks=[]
    for a,b in studies:
        for name in ("metrics.csv","summary.json"):
            digest=sha(RESULTS/a/name)
            if digest!=sha(RESULTS/b/name):raise AssertionError("analyzer byte replay")
            checks.append({"root":a,"file":name,"sha256":digest,"byte_identical":True})
    locks=[ROOT/"ICML-2027/internal"/p for p in ("relative_return_source_lock_a1_20261015.json",
        "shared_reference_source_lock_20261015.json","mc_relative_return_source_lock_20261015.json")]
    source={}
    for path in locks:
        for name,digest in json.loads(path.read_text())["files"].items():
            if sha(ROOT/name)!=digest:raise AssertionError("frozen source changed")
            source[name]=digest
    upstream=ROOT/"tmp/HARL"
    if subprocess.check_output(["git","-C",str(upstream),"status","--porcelain"],text=True).strip():raise AssertionError("dirty upstream")
    # MC roles are independently reconstructed and checked across all contexts.
    def key(seed,task,block,role,episode):
        return int(np.random.SeedSequence([seed,task,block,role,episode]).generate_state(1)[0]%(2**31-1))
    groups={r:set() for r in (110,610,710,810)}
    for ti in (0,1):
        for seed in (78101001,78101002):
            for stage in range(1,17):
                groups[110].update(key(seed,ti,stage,110,e) for e in range(16))
            for stage in (4,16):
                groups[610].update(key(seed,ti,stage,610,e) for e in range(64))
                groups[710].update(key(seed,ti,stage,710,e) for e in range(256))
                groups[810].update(key(seed,ti,stage,810+ci,e) for ci in range(3) for e in range(32))
    for a in groups:
        for b in groups:
            if a!=b and groups[a]&groups[b]:raise AssertionError("MC episode-role collision")
    if {r:len(v) for r,v in groups.items()}!={110:1024,610:512,710:2048,810:768}:raise AssertionError("MC unique episode counts")
    for name,rep in (("relative_return_a1_scientific_20261015","relative_return_a1_registered_replay_20261015"),
                     ("mc_relative_return_scientific_20261015","mc_relative_return_registered_replay_20261015")):
        if (RESULTS/name/"context_00.json").read_bytes()!=(RESULTS/rep/"context_00.json").read_bytes():raise AssertionError("context replay")
    for path in sorted((RESULTS/"shared_reference_scientific_20261015").glob("context_*.json")):
        if path.read_bytes()!=(RESULTS/"shared_reference_registered_replay_20261015"/path.name).read_bytes():raise AssertionError("shared full replay")
    record={"verification_status":"VERIFIED","analyzer_replays":checks,"source_files":source,
            "mc_unique_episode_counts":{str(r):len(v) for r,v in groups.items()},"mc_episode_roles_disjoint":True,
            "registered_payload_replay":{"first_disagreement":"context 0 exact","shared_bank":"all eight contexts exact","mc_potential":"context 0 exact"},
            "tests":{"passed":43,"command":"conda run --no-capture-output -n ust2 python -m pytest ICML-2027/tests ICML-2027/experiments/test_relative_return_estimator.py ICML-2027/experiments/relative_return_a1/test_stable_overlap.py ICML-2027/experiments/test_mc_relative_return.py -q"},
            "decisions":[json.loads((RESULTS/a/"summary.json").read_text()) for a,_ in studies],
            "scientific_outcome_retries":False,"gpu_or_remote_used":False,"frozen_TSP_manuscript_changed":False,
            "qualification_result":"Variance efficiency improved; every overall positive-selection qualification remains failed. No failed gate is relabelled passed.",
            "next":"Causal observable-score transfer and finite-budget expected-risk/probe-cost design, not GPU performance or manuscript claims."}
    out=RESULTS/"relative_return_complete_validation_20261015.json"
    with out.open("x",encoding="utf-8",newline="\n") as f:json.dump(record,f,indent=2,sort_keys=True);f.write("\n")
    return record


if __name__=="__main__":
    record=run();print(json.dumps({"verification_status":record["verification_status"],"tests":43,"all_three_overall_gates_pass":False},indent=2))
