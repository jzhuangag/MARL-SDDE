"""Frozen study support; shared native mutex and exact Windows CPU accounting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
LEGACY=ROOT/'experiments/v3_risk_gate8h'
RESEARCH=ROOT/'TSP-V3-revised-20261018/manuscript/internal/conditional_risk_20261021'
sys.path.insert(0,str(LEGACY))
sys.path.insert(0,str(RESEARCH))
from guard import SingleOwner,child_cpu,pid_alive,runtime,sha,dump,historical_blob_matches
PLAN=HERE/'protocol.json'
LOCK=HERE/'registration/lock.json'
ADMISSION=HERE/'registration/admission.json'
RESULT=HERE/'results/formal_20261021'


class Budget:
    def __init__(self,earlier=0.):self.earlier=float(earlier)
    @property
    def cpu(self):return self.earlier+time.process_time()
    def __call__(self,reserve=0.):
        if self.cpu+reserve>=7200:raise TimeoutError('Separate2h CPU ceiling; no retry')


def plan():return json.loads(PLAN.read_text(encoding='utf-8'))


def source_paths():
    return sorted([*HERE.glob('*.py'),PLAN,HERE/'.gitattributes',
       LEGACY/'core.py',LEGACY/'guard.py',LEGACY/'protocol.json',LEGACY/'prepare.py',
       RESEARCH/'controller.py',RESEARCH/'proposed_validation.json',
       ROOT/'TSP-V3-revised-20261018/manuscript/tools/risk_gate_core.py'])


def verify_sources(lock):
    for rel,digest in lock['sources'].items():
        if sha(ROOT/rel)!=digest:raise AssertionError('Source changed: '+rel)


def verify_registration():
    lock=json.loads(LOCK.read_text(encoding='utf-8'))
    admission=json.loads(ADMISSION.read_text(encoding='utf-8'))
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
    if subprocess.check_output(['git','rev-list','--count','@{upstream}..HEAD'],cwd=ROOT).strip()!=b'0':
        raise AssertionError('Registration not pushed')
    return lock


def prerequisites():
    from guard import predecessor_ready
    ready,reason=predecessor_ready()
    if not ready:raise RuntimeError(reason)
    terminal=LEGACY/'results/formal_20261019/audited_terminal.json'
    if not terminal.exists():raise RuntimeError('Prior V3 post-exit audit missing')
    old=json.loads((LEGACY/'results/formal_20261019/worker_launch.json').read_text())
    if pid_alive(old['pid']):raise RuntimeError('Prior V3 owner is live')
    if not (ROOT/'ICML-2027/internal/v3_risk_gate8h_terminal_20261020/REPORT.md').exists():
        raise RuntimeError('Prior V3 independent scientific audit missing')
