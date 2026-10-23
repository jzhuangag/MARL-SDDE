"""Meter every deterministic attempt, audit historical seeds, freeze before data."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from support import (HERE,ROOT,RESEARCH,RESULT,PLAN,LOCK,ADMISSION,source_paths,sha,dump,
    runtime,child_cpu,historical_blob_matches,plan)
PREP=HERE/'preparation'


def source_hashes():return {p.relative_to(ROOT).as_posix():sha(p) for p in source_paths()}


def overlap(obj,targets,context=''):
    found=set()
    if isinstance(obj,dict):
        for key,value in obj.items():
            path=context+'/'+key
            if key.endswith('start') and ('seed' in path.lower()) and isinstance(value,int):
                count=obj.get(key[:-5]+'count',obj.get('count'))
                end=obj.get(key[:-5]+'end',obj.get('end'))
                if isinstance(count,int):found.update(t for t in targets if value<=t<value+count)
                if isinstance(end,int):found.update(t for t in targets if value<=t<=end)
            found.update(overlap(value,targets,path))
    elif isinstance(obj,list):
        for value in obj:found.update(overlap(value,targets,context))
    return found


def seed_screen(cfg):
    # Conservative scan over historical source/metadata, never outcome payloads.
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    targets=set(range(cfg['formal_seed_start'],cfg['formal_seed_end']+1))|{cfg['analysis_seed']}
    checked={};conflicts={}
    excluded_prefixes=('experiments/v3_conditional2h/',
       'TSP-V3-revised-20261018/manuscript/internal/conditional_risk_20261021/')
    for rel in tracked:
        low=rel.lower()
        if '/results/' in low or rel.startswith(excluded_prefixes):continue
        metadata=low.endswith('.json') and any(x in Path(low).name for x in ('seed','registry','protocol','lock','config'))
        source=low.endswith('.py') and (low.startswith('experiments/') or low.startswith('icml-2027/experiments/'))
        if not (metadata or source):continue
        path=ROOT/rel;content=path.read_text(encoding='utf-8-sig')
        if 'seed' not in content.lower():continue
        literals={int(x.replace('_','')) for x in re.findall(r'(?<![\w.])\d[\d_]{5,11}(?![\w.])',content)}
        found=literals&targets
        for a,b in re.findall(r'(?<!\d)(\d{7,10})\s*(?:--?|\.\.)\s*(\d{7,10})(?!\d)',content):
            lo,hi=int(a),int(b)
            if lo<=hi:found.update(t for t in targets if lo<=t<=hi)
        if path.suffix=='.json':found.update(overlap(json.loads(content),targets))
        checked[rel]=sha(path)
        if found:conflicts[rel]=sorted(found)
    required=['docs/exp016a_seed_registry.json','docs/exp016b_seed_registry.json',
        'docs/exp016b_formal_registry.json','docs/exp017a_pilot_registry.json',
        'docs/exp018a_seed_registry.json','docs/exp018b_seed_registry.json',
        'experiments/v3_risk_gate8h/protocol.json']
    if not all(p in checked for p in required):raise AssertionError('Historical registries missing')
    if conflicts:raise AssertionError('Historical seed conflict '+json.dumps(conflicts))
    return dict(status='PASS',historical_files_checked=len(checked),source_sha256=checked,
        required=required,overlap=[],scientific_outcomes_read=False,
        formal_start=cfg['formal_seed_start'],formal_end=cfg['formal_seed_end'],analysis_seed=cfg['analysis_seed'],
        excluded_planned_self_provenance=list(excluded_prefixes),
        method='Tracked historical seed/registry/protocol/lock/config JSON and all experiments Python mentioning seed; literals, encoded ranges and structured seed ranges')


def tests():
    if RESULT.exists():raise RuntimeError('No pre-data tests after formal root')
    PREP.mkdir(exist_ok=True);number=len(list(PREP.glob('test_attempt_*.json')))+1
    sources=source_hashes()
    child=subprocess.Popen([sys.executable,str(HERE/'tests.py')],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err=child.communicate();cpu=child_cpu(child._handle)
    try:result=json.loads(out.decode('utf-8'))
    except Exception:result=dict(successful=False,raw_stdout=out.decode(errors='replace'))
    total=cpu+result.get('test_child_cpu_seconds',0.)
    report=dict(attempt=number,utc=datetime.now(timezone.utc).isoformat(),sources=sources,
        result=result,exit_code=child.returncode,exact_test_parent_and_child_cpu_seconds=total,
        exact_test_parent_cpu_seconds=cpu,wrapper_cpu_snapshot_seconds=time.process_time(),
        stderr=err.decode(errors='replace'),all_failed_costs_charged=True,scientific_attempt=False)
    dump(PREP/f'test_attempt_{number:03d}.json',report)
    print(json.dumps(dict(attempt=number,successful=result.get('successful'),checks=result.get('checks'),cpu_seconds=total)))
    return child.returncode


def freeze():
    if RESULT.exists() or LOCK.exists() or ADMISSION.exists():raise RuntimeError('Existing registration/data preserved')
    reports=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(PREP.glob('test_attempt_*.json'))]
    if not reports or reports[-1]['exit_code'] or reports[-1]['result'].get('successful') is not True:
        raise AssertionError('Latest recorded tests not PASS')
    sources=source_hashes()
    if reports[-1]['sources']!=sources:raise AssertionError('Source changed since PASS')
    if subprocess.check_output(['git','rev-list','--count','@{upstream}..HEAD'],cwd=ROOT).strip()!=b'0':
        raise AssertionError('Push implementation/tests before freezing')
    for rel,digest in sources.items():
        raw=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=digest:raise AssertionError('Uncommitted source '+rel)
    cfg=plan();screen=seed_screen(cfg);historical_blobs={}
    for rel,digest in screen['source_sha256'].items():
        data=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
        if not historical_blob_matches((ROOT/rel).read_bytes(),data):raise AssertionError('Historical bytes differ beyond newlines '+rel)
        historical_blobs[rel]=hashlib.sha256(data).hexdigest();sources[rel]=digest
    LOCK.parent.mkdir(exist_ok=False)
    dump(LOCK.parent/'seed_registry_audit.json',screen)
    sources[(LOCK.parent/'seed_registry_audit.json').relative_to(ROOT).as_posix()]=sha(LOCK.parent/'seed_registry_audit.json')
    for path in sorted(PREP.glob('test_attempt_*.json')):sources[path.relative_to(ROOT).as_posix()]=sha(path)
    measured=sum(r['exact_test_parent_and_child_cpu_seconds'] for r in reports)
    locked_runtime=runtime()
    if time.process_time()>=60:raise TimeoutError('Preparation helper exceeded60s charged allowance')
    lock=dict(status='FROZEN_PRE_DATA',protocol=cfg,sources=sources,historical_blobs=historical_blobs,
        runtime=locked_runtime,preparation_measured_cpu_seconds=measured,
        auxiliary_charge_seconds=60.,auxiliary_is_charged_not_claimed_measured=True,
        freeze_parent_cpu_snapshot=time.process_time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        formal_science_generated=0,attempts=len(reports),latest_checks=reports[-1]['result']['checks'])
    dump(LOCK,lock)
    dump(ADMISSION,dict(status='REGISTERED_BEFORE_DATA',lock_sha256=sha(LOCK),protocol_sha256=sha(PLAN),
        cap_seconds=7200,earlier_charge=measured+60.,science_rows=294912,replay_rows=294912,
        no_retry=True,no_shrinkage=True,must_push_before_launch=True))
    print(json.dumps(dict(lock_sha256=sha(LOCK),tests=lock['latest_checks'],historical_files=len(historical_blobs),
        charged_cpu_seconds=measured+60.,formal_science=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--test',action='store_true');mode.add_argument('--freeze',action='store_true')
    args=parser.parse_args();sys.exit(tests() if args.test else freeze())
