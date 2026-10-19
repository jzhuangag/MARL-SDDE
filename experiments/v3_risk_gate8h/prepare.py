"""Record every toy-test cost, then freeze a pushed source/runtime admission."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from guard import (ROOT,HERE,PROTOCOL,SPEC,LOCK,ADMISSION,RESULT,RECEIPT,PREDECESSOR,
                   child_cpu,dump,sha,runtime,source_paths,run_command)

PREP = HERE/"preparation"


def source_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in source_paths()}


def numeric_overlap(text,targets):
    """Conservative literal/range screen, supplemented by structured seed ranges."""
    numbers = {int(x.replace("_","")) for x in re.findall(r"(?<![\w.])\d[\d_]{5,11}(?![\w.])",text)}
    found = numbers&targets
    for a,b in re.findall(r"(?<!\d)(\d{7,10})\s*(?:--?|\.\.)\s*(\d{7,10})(?!\d)",text):
        lo,hi = int(a),int(b)
        if lo<=hi:
            found.update(t for t in targets if lo<=t<=hi)
    return sorted(found)


def structured_overlap(obj,targets):
    found = set()
    if isinstance(obj,dict):
        for key,value in obj.items():
            if "seed" in key.lower() and key.endswith("start") and isinstance(value,int):
                count = obj.get(key[:-5]+"count")
                if isinstance(count,int):
                    found.update(t for t in targets if value<=t<value+count)
            found.update(structured_overlap(value,targets))
    elif isinstance(obj,list):
        for value in obj:
            found.update(structured_overlap(value,targets))
    return found


def seed_registry_audit(protocol):
    tracked,_ = run_command(["git","ls-files"])
    candidates = []
    for rel in tracked.decode("utf-8").splitlines():
        low = rel.lower()
        if "/results/" in low or low.startswith("experiments/v3_risk_gate8h/"):
            continue
        metadata = low.endswith(".json") and any(s in Path(low).name for s in ("seed","registry","protocol","lock","config"))
        source = low.endswith(".py") and low.startswith("experiments/dependence_delay_linear/")
        if metadata or source:
            candidates.append(rel)
    targets = set(range(protocol["formal_seeds"]["start"],protocol["formal_seeds"]["end"]+1))
    targets.add(protocol["analysis"]["analysis_rng_seed"])
    checked,conflicts = {},{}
    for rel in candidates:
        p = ROOT/rel
        content = p.read_text(encoding="utf-8-sig")
        if "seed" not in content.lower():
            continue
        checked[rel] = sha(p)
        found = set(numeric_overlap(content,targets))
        if p.suffix==".json":
            found.update(structured_overlap(json.loads(content),targets))
        if found:
            conflicts[rel] = sorted(found)
    required = ["docs/exp016a_seed_registry.json","docs/exp016b_seed_registry.json",
                "docs/exp016b_formal_registry.json","docs/exp017a_pilot_registry.json",
                "docs/exp018a_seed_registry.json","docs/exp018b_seed_registry.json"]
    if not all(p in checked for p in required):
        raise AssertionError("Historical seed registry missing from screen")
    if conflicts:
        raise AssertionError("Proposed seed overlap: "+json.dumps(conflicts))
    return dict(status="PASS",method="all historical tracked seed/registry/protocol/lock/config JSON plus old dependence-delay Python; conservative literals and encoded ranges",
                old_metadata_sha256=checked,old_files_checked=len(checked),required_registries=required,
                formal_seed_count=128,formal_start=20571001,formal_end=20571128,
                analysis_seed=20579901,overlap=[],scientific_outcomes_read=False)


def tests():
    if RESULT.exists():
        raise RuntimeError("Preparation prohibited after any formal data exist")
    PREP.mkdir(exist_ok=True)
    paths = sorted(PREP.glob("test_attempt_*.json"))
    number = len(paths)+1
    snapshot = source_hashes()
    child = subprocess.Popen([sys.executable,str(HERE/"tests.py")],cwd=ROOT,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err = child.communicate()
    cpu = child_cpu(child._handle)
    try:
        result = json.loads(out.decode("utf-8"))
    except Exception:
        result = dict(successful=False,stdout=out.decode(errors="replace"),unexpected_output=True)
    report = dict(attempt=number,utc=datetime.now(timezone.utc).isoformat(),source_sha256=snapshot,
                  test_result=result,exit_code=child.returncode,exact_child_cpu_seconds=cpu,
                  preparation_parent_cpu_seconds=time.process_time(),
                  stderr=err.decode(errors="replace"),all_failed_test_costs_charged=True,
                  native_scientific_attempt=False)
    target = PREP/f"test_attempt_{number:03d}.json"
    dump(target,report)
    print(json.dumps(dict(report=str(target),tests=result.get("tests_run"),passed=result.get("successful"),exact_test_cpu_seconds=cpu)))
    return child.returncode


def freeze():
    if RESULT.exists() or LOCK.exists() or ADMISSION.exists():
        raise RuntimeError("No overwriting a registered lock or data root")
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PREP.glob("test_attempt_*.json"))]
    if not reports or reports[-1]["exit_code"]!=0 or reports[-1]["test_result"].get("successful") is not True:
        raise AssertionError("Latest immutable toy-test attempt must PASS")
    hashes = source_hashes()
    if reports[-1]["source_sha256"]!=hashes:
        raise AssertionError("Source changed since latest PASS; rerun toy tests with cost recorded")
    head,_ = run_command(["git","rev-parse","HEAD"])
    ahead,_ = run_command(["git","rev-list","--count","@{upstream}..HEAD"])
    if ahead.strip()!=b"0":
        raise AssertionError("Push implementation and test evidence before freezing admission")
    for rel,digest in hashes.items():
        data,_ = run_command(["git","show",f"HEAD:{rel}"])
        if hashlib.sha256(data).hexdigest()!=digest:
            raise AssertionError("Source not committed: "+rel)
    plan = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    seeds = seed_registry_audit(plan)
    seedpath = PREP/"seed_registry_audit.json"
    dump(seedpath,seeds)
    for rel,digest in seeds["old_metadata_sha256"].items():
        hashes[rel] = digest
    hashes[seedpath.relative_to(ROOT).as_posix()] = sha(seedpath)
    for p in sorted(PREP.glob("test_attempt_*.json")):
        hashes[p.relative_to(ROOT).as_posix()] = sha(p)
    locked_runtime = runtime()
    if time.process_time()>=60:
        raise TimeoutError("Registration exceeded conservative60CPU-second auxiliary charge")
    measured = sum(r["exact_child_cpu_seconds"] for r in reports)
    lock = dict(schema=1,status="FROZEN_BEFORE_ANY_NEW_FORMAL_DATA",source_commit=head.decode().strip(),
                source_sha256=hashes,protocol=plan,execution_spec=json.loads(SPEC.read_text(encoding="utf-8")),runtime=locked_runtime,
                preparation_measured_cpu_seconds=measured,all_test_attempts=len(reports),
                latest_test_count=reports[-1]["test_result"]["tests_run"],
                registration_auxiliary_charge_seconds=60.,
                preparation_parent_snapshots_seconds=[r["preparation_parent_cpu_seconds"] for r in reports],
                freeze_parent_cpu_snapshot_seconds=time.process_time(),
                earlier_charge_seconds=measured+60.,cap_seconds=28800,
                predecessor=dict(receipt=RECEIPT.relative_to(ROOT).as_posix(),native_root=PREDECESSOR.relative_to(ROOT).as_posix(),
                                 driver_exit_AND_independent_audit=True),
                human_authority="Separate8CPUh approved; user renewed autonomous implementation/testing/lock and automatic launch ONLY after140h exit AND final audit; no further confirmation",
                generated_formal_trajectories=0,results_root_must_not_exist=True)
    dump(LOCK,lock)
    dump(ADMISSION,dict(status="REGISTERED_QUEUED_UNTIL_AUDITED_PREDECESSOR_EXIT",lock_sha256=sha(LOCK),
                        protocol_sha256=sha(PROTOCOL),source_commit=lock["source_commit"],
                        full_science_rows=196608,full_replay_rows=196608,policies=6,learners=2,seeds=128,
                        cap_seconds=28800,earlier_charge_seconds=measured+60.,
                        predecessor_audit_required=True,no_retry=True,no_shrinkage=True,extra_cap=False,
                        must_commit_and_push_this_lock_and_admission_before_launch=True))
    print(json.dumps(dict(lock_sha256=sha(LOCK),admission_sha256=sha(ADMISSION),
                          tests=lock["latest_test_count"],old_seed_files=seeds["old_files_checked"],
                          measured_tests_cpu_seconds=measured,charged_cpu_seconds=measured+60.,
                          remaining_cpu_seconds=28800-measured-60.,formal_trajectories_generated=0)))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--test",action="store_true")
    options.add_argument("--freeze",action="store_true")
    args = parser.parse_args()
    raise SystemExit(tests() if args.test else freeze())
