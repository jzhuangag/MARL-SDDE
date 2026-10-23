"""Run a frozen independent audit once, then independently recheck runtime bytes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def run(study):
    if study not in ('v3_conditional2h','v3_observable_gate1h'):raise ValueError('Unknown study')
    package=ROOT/'experiments'/study;sys.path.insert(0,str(package))
    import support
    if (support.RESULT/'independent_scientific_audit.json').exists():raise RuntimeError('Existing independent audit preserved')
    spec=importlib.util.spec_from_file_location('terminal_frozen_audit',package/'audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.run()
    expected=json.loads(support.LOCK.read_text())['runtime'];actual=support.runtime()
    if actual!=expected:raise AssertionError('Independent terminal runtime mismatch')
    support.dump(support.RESULT/'independent_runtime_readback.json',dict(status='PASS',runtime=actual,lock_sha256=support.sha(support.LOCK),independent_scientific_audit_sha256=support.sha(support.RESULT/'independent_scientific_audit.json'),wrapper_source_sha256=support.sha(Path(__file__)),wrapper_process_cpu_seconds=time.process_time(),runtime_checked_after_scientific_readback=True))
    print(json.dumps(dict(status='RUNTIME_AND_SCIENTIFIC_READBACK_PASS',cpu_snapshot=time.process_time())))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');a=p.parse_args();run(a.study)
