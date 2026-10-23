"""One native owner; 7200s total includes preparation, worker, supervisor and audit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import json
import subprocess
import sys
import time
import traceback
from support import (HERE,RESULT,LOCK,ADMISSION,SingleOwner,child_cpu,dump,sha,
                     verify_registration,verify_sources,prerequisites,plan)


def run():
    prerequisites()
    with SingleOwner():
        prerequisites();lock=verify_registration();cfg=plan()
        if RESULT.exists():raise FileExistsError('Existing root preserved; no resume or retry')
        import shutil
        if shutil.disk_usage(HERE).free<3*1024**3:raise RuntimeError('Insufficient local preservation space')
        earlier=lock['preparation_measured_cpu_seconds']+cfg['auxiliary_charge_seconds']
        if earlier+time.process_time()>=300:raise TimeoutError('Setup profile allowance already exhausted')
        RESULT.mkdir(parents=True,exist_ok=False)
        dump(RESULT/'execution.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
            supervisor_pid=os.getpid(),cap_seconds=7200,preparation_charge=earlier,
            registration_lock_sha256=sha(LOCK),admission_sha256=sha(ADMISSION),worker_authorized=True,
            no_retry=True,no_shrinkage=True))
        worker=None;cost=0.
        try:
            with (RESULT/'stdout.log').open('x') as out,(RESULT/'stderr.log').open('x') as err:
                command=[sys.executable,str(HERE/'worker.py'),'--earlier',str(earlier+time.process_time()),'--parent',str(os.getpid())]
                worker=subprocess.Popen(command,cwd=HERE,stdout=out,stderr=err)
                dump(RESULT/'worker_launch.json',dict(pid=worker.pid,command=command))
                while worker.poll() is None:
                    cost=child_cpu(worker._handle)
                    if earlier+time.process_time()+cost>=7200:
                        worker.terminate();worker.wait();cost=child_cpu(worker._handle)
                        raise TimeoutError('Whole2h supervisor CPU cap; partial data preserved')
                    time.sleep(.5)
                cost=child_cpu(worker._handle)
            dump(RESULT/'worker_exit.json',dict(pid=worker.pid,exit_code=worker.returncode,
                exact_worker_cpu_seconds=cost,cumulative_charged_cpu_seconds=earlier+cost+time.process_time()))
            if worker.returncode!=0:raise RuntimeError('Native failure; no retry')
            marker=json.loads((RESULT/'workflow_complete.json').read_text())
            if marker['science_rows']!=294912 or marker['replay_rows']!=294912:raise AssertionError('Incomplete workflow')
            manifests=[]
            for phase in ('science','replay'):
                manifest=json.loads((RESULT/phase/'hashes.json').read_text())
                if len(manifest)!=128:raise AssertionError('Missing payloads')
                for name,digest in manifest.items():
                    if sha(RESULT/phase/name)!=digest:raise AssertionError('Post-exit payload hash')
                manifests.append(manifest)
            if manifests[0]!=manifests[1]:raise AssertionError('Post-exit replay mismatch')
            for name,digest in marker['analysis_sha256'].items():
                if sha(RESULT/'analysis_a'/name)!=digest or sha(RESULT/'analysis_b'/name)!=digest:
                    raise AssertionError('Post-exit double analysis mismatch')
            for name,digest in marker['plots'].items():
                if sha(RESULT/'plots'/name)!=digest:raise AssertionError('Plot hash')
            verify_sources(lock)
            charge=earlier+cost+time.process_time()
            if charge>=7200:raise TimeoutError('Post-exit CPU cap')
            dump(RESULT/'audited_terminal.json',dict(status='POST_EXIT_INTEGRITY_VERIFIED_INDEPENDENT_SCIENTIFIC_VISUAL_AUDIT_PENDING',
                worker_exited=True,science_rows=294912,replay_rows=294912,analyses=2,full_replay_byte_equal=True,
                all_seven_analysis_files_byte_equal=True,exact_worker_cpu_seconds=cost,
                supervisor_cpu_seconds=time.process_time(),preparation_charge=earlier,
                cumulative_charged_cpu_seconds=charge,workflow_complete_sha256=sha(RESULT/'workflow_complete.json'),
                manuscript_admission=False))
        except BaseException as exc:
            if worker is not None and worker.poll() is None:
                worker.terminate();worker.wait();cost=child_cpu(worker._handle)
            dump(RESULT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
                worker_cpu_seconds=cost,supervisor_cpu_seconds=time.process_time(),
                cumulative_charged_cpu_seconds=earlier+cost+time.process_time(),partial_work_preserved=True,retry=False))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:raise RuntimeError('Explicit admitted execution required')
    run()
