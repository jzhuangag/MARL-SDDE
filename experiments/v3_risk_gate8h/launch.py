"""Audited140h handoff, single owner, whole8h guard, post-exit read-back audit."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from guard import (ADMISSION, HERE, LOCK, RESULT, RECEIPT, Budget, SingleOwner,
                   child_cpu, dump, predecessor_ready, sha, verify_registration,verify_sources)


def final_readback(budget,lock,worker_cpu):
    from analysis import load_rows,audit_row
    if not (RESULT/"workflow_complete.json").exists():
        raise AssertionError("Worker exited without complete marker")
    completed = json.loads((RESULT/"workflow_complete.json").read_text(encoding="utf-8"))
    if completed["science_rows"]!=196608 or completed["replay_rows"]!=196608:
        raise AssertionError("Terminal count mismatch")
    manifests = []
    for phase in ("science","replay"):
        budget()
        manifest = json.loads((RESULT/phase/"hashes.json").read_text(encoding="utf-8"))
        if len(manifest)!=128:
            raise AssertionError("Missing seed payloads")
        for name,digest in manifest.items():
            budget()
            if sha(RESULT/phase/name)!=digest:
                raise AssertionError("Payload SHA mismatch")
        manifests.append(manifest)
    if manifests[0]!=manifests[1]:
        raise AssertionError("Full replay digest mismatch")
    frame = load_rows(RESULT/"science",budget)
    if len(frame)!=196608:
        raise AssertionError("Read-back row count mismatch")
    for i,row in enumerate(frame.to_dict("records")):
        if i%512==0:
            budget()
        audit_row(row)
    for name,digest in completed["analysis_sha256"].items():
        budget()
        if sha(RESULT/"analysis_a"/name)!=digest or sha(RESULT/"analysis_b"/name)!=digest:
            raise AssertionError("Post-exit double-analysis mismatch")
    if len(list((RESULT/"plots").glob("*.png")))!=4:
        raise AssertionError("Missing all-policy risk/cost plots")
    verify_sources(lock)
    budget()
    dump(RESULT/"audited_terminal.json",dict(status="FULL_WORKFLOW_AND_POST_EXIT_READBACK_VERIFIED_VISUAL_REVIEW_PENDING",
         worker_exited=True,science_rows=196608,replay_rows=196608,analysis_copies=2,
         native_worker_cpu_seconds=worker_cpu,supervisor_cpu_seconds=time.process_time(),
         cumulative_charged_cpu_seconds=budget.cpu,
         tracked_measured_cpu_seconds=budget.cpu-60.,registration_auxiliary_allowance_charged_seconds=60.,
         full_replay_byte_identical=True,all_sha_finiteness_and_ledgers=True,
         all_analysis_bytes_identical=True,all_negative_results_retained=True,
         plots_visual_review="requires agent inspection after supervisor exit",manuscript_admission=False,
         workflow_complete_sha256=sha(RESULT/"workflow_complete.json")))


def run(execute):
    ready,reason = predecessor_ready()
    if not ready:
        print(json.dumps(dict(status="WAITING_NO_NEW_DATA",reason=reason),ensure_ascii=False))
        return
    if not execute:
        lock,admission,cost = verify_registration()
        print(json.dumps(dict(status="READY_TO_LAUNCH",lock_sha256=sha(LOCK),verification_child_cpu_seconds=cost)))
        return
    with SingleOwner():
        # Recheck after mutex acquisition; no data root is created on a blocked gate.
        ready,reason = predecessor_ready()
        if not ready:
            raise RuntimeError(reason)
        lock,admission,verification_child_cost = verify_registration()
        if RESULT.exists():
            raise FileExistsError("Existing native root preserved; no resume or retry")
        import shutil
        if shutil.disk_usage(HERE).free<2*1024**3:
            raise RuntimeError("Insufficient free local space for preserved science/replay")
        budget = Budget(lock["preparation_measured_cpu_seconds"]+60.,verification_child_cost)
        budget()
        RESULT.mkdir(parents=True,exist_ok=False)
        dump(RESULT/"execution.json",dict(supervisor_pid=os.getpid(),worker_authorized=True,
             started_utc=datetime.now(timezone.utc).isoformat(),lock_sha256=sha(LOCK),
             admission_sha256=sha(ADMISSION),predecessor_audit_receipt_sha256=sha(RECEIPT),
             cpu_cap_seconds=28800,preparation_measured_cpu_seconds=lock["preparation_measured_cpu_seconds"],
             registration_auxiliary_allowance_charged_seconds=60.,retry_authorized=False))
        worker = None
        worker_cost = 0.
        worker_cost_added = False
        try:
            with (RESULT/"stdout.log").open("x") as out,(RESULT/"stderr.log").open("x") as err:
                command = [sys.executable,str(HERE/"worker.py"),"--earlier-cpu",str(budget.cpu),"--parent-pid",str(os.getpid())]
                worker = subprocess.Popen(command,cwd=HERE,stdout=out,stderr=err)
                dump(RESULT/"worker_launch.json",dict(pid=worker.pid,command=command))
                while worker.poll() is None:
                    worker_cost = child_cpu(worker._handle)
                    if budget.cpu+worker_cost>=28800:
                        worker.terminate()
                        worker.wait()
                        worker_cost = child_cpu(worker._handle)
                        raise TimeoutError("Supervisor cumulative8h CPU guard; active payload preserved")
                    time.sleep(.5)
                worker_cost = child_cpu(worker._handle)
            budget.child += worker_cost
            worker_cost_added = True
            dump(RESULT/"worker_exit.json",dict(pid=worker.pid,exit_code=worker.returncode,
                 exact_worker_cpu_seconds=worker_cost,cumulative_charged_cpu_seconds=budget.cpu))
            if worker.returncode!=0:
                raise RuntimeError("Native worker failed; no retry")
            budget()
            final_readback(budget,lock,worker_cost)
        except BaseException as error:
            if worker is not None and worker.poll() is None:
                worker.terminate()
                worker.wait()
                worker_cost = child_cpu(worker._handle)
            dump(RESULT/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                 exact_worker_cpu_seconds=worker_cost,supervisor_cpu_seconds=time.process_time(),
                 cumulative_charged_cpu_seconds=budget.cpu+(0. if worker_cost_added else worker_cost),
                 all_partial_artifacts_preserved=True,retry_authorized=False))
            raise


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",action="store_true")
    run(parser.parse_args().run)
