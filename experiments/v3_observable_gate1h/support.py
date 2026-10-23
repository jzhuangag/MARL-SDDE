"""Separate observable-probe study; no changes to the preceding frozen study."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'experiments/v3_conditional2h'
LEGACY=ROOT/'experiments/v3_risk_gate8h'
RESEARCH=ROOT/'TSP-V3-revised-20261018/manuscript/internal/conditional_risk_20261021'
sys.path.insert(0,str(LEGACY));sys.path.insert(0,str(RESEARCH))
from guard import SingleOwner,child_cpu,pid_alive,runtime,sha,dump,historical_blob_matches
PLAN=HERE/'protocol.json'
LOCK=HERE/'registration/lock.json'
ADMISSION=HERE/'registration/admission.json'
RESULT=HERE/'results/formal_20261021'

def previous_module(name):
    spec=importlib.util.spec_from_file_location('observable_reused_'+name,PREVIOUS/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class Budget:
    def __init__(self,earlier=0.):self.earlier=float(earlier)
    @property
    def cpu(self):return self.earlier+time.process_time()
    def __call__(self,reserve=0.):
        if self.cpu+reserve>=3600:raise TimeoutError('Separate1h CPU ceiling; no retry')

def plan():return json.loads(PLAN.read_text(encoding='utf-8'))

def source_paths():
    return sorted([*HERE.glob('*.py'),PLAN,HERE/'.gitattributes',HERE/'planning_amendment.json',
        *[PREVIOUS/(n+'.py') for n in ('model','analyze','prepare','tests','worker')],
        LEGACY/'core.py',LEGACY/'guard.py',LEGACY/'protocol.json',LEGACY/'prepare.py',
        RESEARCH/'controller.py',ROOT/'TSP-V3-revised-20261018/manuscript/tools/risk_gate_core.py'])

def verify_sources(lock,executed_only=False):
    selected={p.relative_to(ROOT).as_posix() for p in source_paths()} if executed_only else None
    for rel,digest in lock['sources'].items():
        if selected is not None and rel not in selected:continue
        if sha(ROOT/rel)!=digest:raise AssertionError('Source changed: '+rel)

def verify_registration():
    lock=json.loads(LOCK.read_text());admission=json.loads(ADMISSION.read_text())
    verify_sources(lock)
    if runtime()!=lock['runtime']:raise AssertionError('Runtime changed')
    if sha(LOCK)!=admission['lock_sha256']:raise AssertionError('Admission mismatch')
    for rel,digest in lock['sources'].items():
        raw=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
        expected=lock.get('historical_blobs',{}).get(rel,digest)
        if hashlib.sha256(raw).hexdigest()!=expected:raise AssertionError('Uncommitted input: '+rel)
    for path in (LOCK,ADMISSION):
        raw=subprocess.check_output(['git','show','HEAD:'+path.relative_to(ROOT).as_posix()],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=sha(path):raise AssertionError('Uncommitted registration')
    if subprocess.check_output(['git','rev-list','--count','@{upstream}..HEAD'],cwd=ROOT).strip()!=b'0':raise AssertionError('Registration not pushed')
    return lock

def prerequisites():
    terminal=PREVIOUS/'results/formal_20261021'
    execution=json.loads((terminal/'execution.json').read_text())
    worker=json.loads((terminal/'worker_launch.json').read_text())
    if pid_alive(execution['supervisor_pid']) or pid_alive(worker['pid']):raise RuntimeError('Predecessor still live')
    receipt=json.loads((terminal/'independent_audited_terminal.json').read_text())
    if not all(receipt.get(k) is True for k in ('audit_complete','all_preserved_evidence_reviewed','plots_visually_reviewed')):raise RuntimeError('Independent predecessor audit incomplete')
    report=ROOT/receipt['audit_report']
    if sha(report)!=receipt['audit_report_sha256']:raise RuntimeError('Predecessor report changed')
    if not receipt.get('terminal_decision'):raise RuntimeError('Missing predecessor decision')
