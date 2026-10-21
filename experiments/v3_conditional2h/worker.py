"""Single native worker; first science+replay part is the retained cost profile."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import json
import time
import traceback
from support import HERE,RESULT,LOCK,Budget,dump,sha,verify_sources,plan
from model import old,rows,audit_row
from analyze import analyze,plots


def run(earlier,parent_pid):
    if os.getppid()!=parent_pid:raise RuntimeError('Admitted supervisor required')
    execution=json.loads((RESULT/'execution.json').read_text(encoding='utf-8'))
    if execution['supervisor_pid']!=parent_pid:raise RuntimeError('Parent identity mismatch')
    cfg=plan();lock=json.loads(LOCK.read_text(encoding='utf-8'));budget=Budget(earlier)
    parts=[(learner,seed) for learner in cfg['learners'] for seed in range(cfg['formal_seed_start'],cfg['formal_seed_end']+1)]
    manifests={'science':{},'replay':{}};counts={'science':0,'replay':0};index=0;active=None
    for phase in manifests:(RESULT/phase).mkdir(exist_ok=False)
    with (RESULT/'cost_events.jsonl').open('x',encoding='utf-8',newline='\n') as ledger:
        def part(phase,number):
            nonlocal active,index
            budget(900);verify_sources(lock)
            learner,seed=parts[number];start=time.process_time()
            path=RESULT/phase/f'part_{number:03d}_{learner}_{seed}.jsonl';count=0
            with path.open('x',encoding='utf-8',newline='\n') as stream:
                for family in old.families(cfg):
                    for regime in cfg['regimes']:
                        active=dict(phase=phase,part=number,learner=learner,seed=seed,family=family.key,regime=regime)
                        for row in rows(seed,learner,family,regime,cfg['public_grid']['scales'],lambda:budget(900)):
                            stream.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=True)+'\n')
                            audit_row(row);count+=1
                        stream.flush();budget(900)
                stream.flush();os.fsync(stream.fileno())
            if count!=cfg['rows_per_part']:raise AssertionError('Incomplete learner/seed part')
            digest=sha(path);manifests[phase][path.name]=digest;counts[phase]+=count
            if phase=='replay' and digest!=sha(RESULT/'science'/path.name):raise AssertionError('Semantic replay mismatch')
            cost=time.process_time()-start
            ledger.write(json.dumps(dict(phase=phase,part=number,rows=count,part_cpu_seconds=cost,
                cumulative_charged_cpu_seconds=budget.cpu),sort_keys=True)+'\n');ledger.flush()
            index+=1
            dump(RESULT/f'progress_{index:04d}.json',dict(phase=phase,science_rows=counts['science'],replay_rows=counts['replay'],
                parts={k:len(v) for k,v in manifests.items()},cumulative_charged_cpu_seconds=budget.cpu,
                worker_pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat()))
            return cost
        try:
            first_science=part('science',0);first_replay=part('replay',0)
            projection=budget.cpu+cfg['profile_multiplier']*127*(first_science+first_replay)+cfg['analysis_audit_reserve_seconds']
            admitted=projection<=cfg['cap_seconds'] and budget.cpu<=cfg['profile_setup_ceiling_seconds']
            dump(RESULT/'cost_admission.json',dict(cost_only=True,first_science_cpu_seconds=first_science,
                first_replay_cpu_seconds=first_replay,charged_at_profile=budget.cpu,
                projected_total_cpu_seconds=projection,admitted=admitted,scientific_outcomes_inspected=False,
                complete_prefix_retained=True,remaining_seed_learner_parts=127))
            if not admitted:raise TimeoutError('Fixed full-suite cost projection/setup limit; no shrinkage or retry')
            for phase in ('science','replay'):
                for number in range(1,len(parts)):part(phase,number)
                if counts[phase]!=cfg['trajectories_per_phase']:raise AssertionError('Phase count mismatch')
                dump(RESULT/phase/'hashes.json',manifests[phase])
                dump(RESULT/(phase+'_complete.json'),dict(rows=counts[phase],parts=len(manifests[phase]),cpu=budget.cpu))
            for name in ('analysis_a','analysis_b'):
                analyze(RESULT/'science',RESULT/'replay',RESULT/name,cfg,budget)
                dump(RESULT/(name+'_complete.json'),dict(cpu=budget.cpu))
            digests={}
            for name in cfg['analysis_outputs']:
                digests[name]=sha(RESULT/'analysis_a'/name)
                if digests[name]!=sha(RESULT/'analysis_b'/name):raise AssertionError('Double-analysis byte mismatch')
            plots(RESULT/'analysis_a',RESULT/'plots',cfg,budget)
            verify_sources(lock);budget()
            dump(RESULT/'workflow_complete.json',dict(science_rows=counts['science'],replay_rows=counts['replay'],
                analyses=2,analysis_sha256=digests,plots={n:sha(RESULT/'plots'/n) for n in cfg['plots']},
                cumulative_charged_cpu_seconds=budget.cpu,worker_pid=os.getpid(),supervisor_pid=parent_pid,
                manuscript_admission=False))
        except BaseException as exc:
            dump(RESULT/'worker_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),active=active,
                counts=counts,cumulative_charged_cpu_seconds=budget.cpu,all_partial_work_preserved=True,retry=False))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--earlier',type=float,required=True);parser.add_argument('--parent',type=int,required=True)
    args=parser.parse_args();run(args.earlier,args.parent)
