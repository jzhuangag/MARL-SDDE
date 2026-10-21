"""Meter deterministic attempts and freeze exact bytes before any formal data."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from support import *
PREP=HERE/'preparation'
base=previous_module('prepare')
tests=base.tests
overlap=base.overlap
source_hashes=base.source_hashes

def seed_screen(cfg):
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    targets=set(range(cfg['formal_seed_start'],cfg['formal_seed_end']+1))|{cfg['analysis_seed']}
    checked={};conflicts={};excluded='experiments/v3_observable_gate1h/'
    for rel in tracked:
        low=rel.lower()
        if '/results/' in low or rel.startswith(excluded):continue
        metadata=low.endswith('.json') and any(x in Path(low).name for x in ('seed','registry','protocol','lock','config'))
        source=low.endswith('.py') and (low.startswith('experiments/') or low.startswith('icml-2027/experiments/'))
        if not (metadata or source):continue
        content=(ROOT/rel).read_text(encoding='utf-8-sig')
        if 'seed' not in content.lower():continue
        found={int(x.replace('_','')) for x in re.findall(r'(?<![\w.])\d[\d_]{5,11}(?![\w.])',content)}&targets
        for a,b in re.findall(r'(?<!\d)(\d{7,10})\s*(?:--?|\.\.)\s*(\d{7,10})(?!\d)',content):
            lo,hi=int(a),int(b)
            if lo<=hi:found.update(t for t in targets if lo<=t<=hi)
        if low.endswith('.json'):found.update(overlap(json.loads(content),targets))
        checked[rel]=sha(ROOT/rel)
        if found:conflicts[rel]=sorted(found)
    required=['experiments/v3_conditional2h/protocol.json','experiments/v3_conditional2h/registration/seed_registry_audit.json','experiments/v3_risk_gate8h/protocol.json','docs/exp016a_seed_registry.json','docs/exp016b_formal_registry.json','docs/exp018b_seed_registry.json']
    if not all(p in checked for p in required):raise AssertionError('Historical registries missing')
    if conflicts:raise AssertionError('Historical seed overlap '+json.dumps(conflicts))
    return dict(status='PASS',historical_files_checked=len(checked),source_sha256=checked,required=required,overlap=[],scientific_outcomes_read=False,formal_start=cfg['formal_seed_start'],formal_end=cfg['formal_seed_end'],analysis_seed=cfg['analysis_seed'],excluded_planned_self_provenance=[excluded],method='Historical source and structured seed metadata, including preceding conditional2h registry')

def freeze():
    prerequisites()
    if RESULT.exists() or LOCK.exists() or ADMISSION.exists():raise RuntimeError('Existing registration/data preserved')
    reports=[json.loads(p.read_text()) for p in sorted(PREP.glob('test_attempt_*.json'))]
    if not reports or reports[-1]['exit_code'] or reports[-1]['result'].get('successful') is not True:raise AssertionError('Latest tests not PASS')
    designs=json.loads((HERE/reports[-1]['result']['fixtures']/'data_free_designs.json').read_text())
    if designs['certificate_admissions']==0:raise RuntimeError('Structural no-go: gate identically strongest fixed prefix; preserve and do not sample redundant data')
    sources=source_hashes()
    if reports[-1]['sources']!=sources:raise AssertionError('Changed source since PASS')
    if subprocess.check_output(['git','rev-list','--count','@{upstream}..HEAD'],cwd=ROOT).strip()!=b'0':raise AssertionError('Push implementation first')
    for rel,digest in sources.items():
        if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)).hexdigest()!=digest:raise AssertionError('Uncommitted source '+rel)
    cfg=plan();screen=seed_screen(cfg);blobs={}
    for rel,digest in screen['source_sha256'].items():
        raw=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
        if not historical_blob_matches((ROOT/rel).read_bytes(),raw):raise AssertionError('Historical non-newline mismatch '+rel)
        blobs[rel]=hashlib.sha256(raw).hexdigest();sources[rel]=digest
    LOCK.parent.mkdir(exist_ok=False);dump(LOCK.parent/'seed_registry_audit.json',screen)
    sources[(LOCK.parent/'seed_registry_audit.json').relative_to(ROOT).as_posix()]=sha(LOCK.parent/'seed_registry_audit.json')
    for path in sorted(PREP.glob('test_attempt_*.json')):sources[path.relative_to(ROOT).as_posix()]=sha(path)
    measured=sum(r['exact_test_parent_and_child_cpu_seconds'] for r in reports)
    rt=runtime()
    if time.process_time()>=60:raise TimeoutError('Preparation allowance exceeded')
    lock=dict(status='FROZEN_PRE_DATA',protocol=cfg,sources=sources,historical_blobs=blobs,runtime=rt,preparation_measured_cpu_seconds=measured,auxiliary_charge_seconds=60.,auxiliary_is_charged_not_claimed_measured=True,freeze_parent_cpu_snapshot=time.process_time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),formal_science_generated=0,attempts=len(reports),latest_checks=reports[-1]['result']['checks'])
    dump(LOCK,lock);dump(ADMISSION,dict(status='REGISTERED_BEFORE_DATA',lock_sha256=sha(LOCK),protocol_sha256=sha(PLAN),cap_seconds=3600,earlier_charge=measured+60.,science_rows=294912,replay_rows=294912,no_retry=True,no_shrinkage=True,must_push_before_launch=True))
    print(json.dumps(dict(lock_sha256=sha(LOCK),checks=lock['latest_checks'],historical_files=len(blobs),charged_cpu_seconds=measured+60.,formal_science=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--test',action='store_true');g.add_argument('--freeze',action='store_true');a=p.parse_args();sys.exit(tests() if a.test else freeze())
