"""Single-worker original/replay/audit owner for a frozen native qualification."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
from sealed_bridge.native import base, FARAMA, run_cell
from sealed_bridge.core import Option, RiskQueueController


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(path):
    entries={str(p.relative_to(path)).replace('\\','/'):sha(p) for p in sorted(Path(path).rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    return hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()


def fingerprint():
    import gymnasium, gymnasium_robotics, mujoco
    names=['core.py','native.py','qualify.py','qualification.json','test_core.py']
    files={str((HERE/n).relative_to(ROOT)).replace('\\','/'):sha(HERE/n) for n in names}
    for p in [HERE.parent/'happo_batch_cpu.py',HERE.parent/'happo_batch_cpu_preregistration.json']:
        files[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    hc=subprocess.check_output(['git','-C',str(base.UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(base.UPSTREAM),'status','--porcelain'],text=True)
    if dirty:raise AssertionError('HARL source is dirty')
    return dict(files=files,harl_commit=hc,pettingzoo_tree=tree_hash(base.RUNTIME),
                farama_tree=tree_hash(FARAMA),python=platform.python_version(),numpy=np.__version__,
                torch=torch.__version__,gymnasium=gymnasium.__version__,
                robotics=gymnasium_robotics.__version__,mujoco=mujoco.__version__)


def check(row,plan):
    n=len(row['launches']);h=plan['horizon'];N=plan['source_episodes']
    assert n==len(row['receipts'])==plan['source_cohorts']
    assert row['evaluations_collected']==0 and row['qualification_returns_recorded'] is False
    assert row['performance_evidence'] is False and row['all_source_cohorts_consumed']
    assert len(set(row['episode_ids']))==len(row['episode_ids'])
    launches={x['id']:x for x in row['launches']}
    seen=set(); source_total=bridge_total=0; age=False; previous=row['initial_hash']
    agents=3 if row['task']=='spread' else 2
    P=row['policy_parameter_bytes']; C=row['baseline_parameter_bytes']
    ebytes=row['per_episode_payload_bytes']
    c=row['costs']; expected={k:0 for k in c}
    expected['source_transitions']=n*N*h
    expected['actor_sampling_rows']=agents*n*N*h
    expected['source_payload_bytes']=n*N*ebytes
    expected['policy_broadcast_bytes']=n*P;expected['baseline_broadcast_bytes']=n*C
    expected['model_copy_bytes']=n*(P+C); expected['messages']=n*(2*agents+1)
    options=[Option(m,b) for m in plan['bridge_episodes'] for b in plan['betas']]
    ctrl=RiskQueueController(options,step=plan['meta_step'],variance_weight=plan['variance_weight'],targets=plan['resource_targets'])
    for r in row['receipts']:
        assert r['source_id'] not in seen;seen.add(r['source_id'])
        s=launches[r['source_id']];d=r['decision'];m=d['episodes']
        assert s['source_version']==r['source_version'] and s['source_hash']==r['source_hash']
        assert s['data_hash']==r['data_hash'] and s['source_payload_bytes']==N*ebytes
        word=hashlib.sha256(f"{row['seed']}:source:{s['id']}".encode()).digest()
        delay=plan['delay_min']+int.from_bytes(word[:8],'big')%(plan['delay_max']-plan['delay_min']+1)
        assert s['due']==s['launch_tick']+delay and r['tick']==s['due']
        assert d['before_anchor_and_bridge'] and d['current_hash']==r['target_before']==previous
        previous=r['target_after']
        assert r['target_version']==len(seen)-1 and r['actual_version_age']==r['target_version']-r['source_version']>=0
        age|=r['actual_version_age']>0
        assert r['source_cohort_consumed_once'] and not r['source_cohort_reused']
        assert not r['current_return_ranked'] and not r['evaluation_used']
        assert r['actor_optimizer']=='server_sgd' and not r['ordinary_adam_theorem_claimed']
        assert r['target_weight_max']<=2+1e-12 and r['source_weight_max']<=2+1e-12
        assert len(r['source_ids'])==N and len(r['bridge_ids'])==m and len(r['complete_episode_modes'])==m
        assert set(r['source_ids']).isdisjoint(r['bridge_ids'])
        assert r['source_payload_bytes']==N*ebytes and r['bridge_payload_bytes']==m*ebytes
        assert r['exact_native_transitions']==(N+m)*h
        source_total+=N*h;bridge_total+=m*h
        expected['bridge_transitions']+=m*h;expected['actor_sampling_rows']+=agents*m*h
        expected['gradient_forward_rows']+=agents*(N+2*m)*h
        expected['gradient_backward_rows']+=agents*(N+2*m)*h
        expected['critic_value_rows']+=(N+3*m)*h
        expected['critic_backward_rows']+=plan['critic_epochs']*m*h
        expected['actor_sgd_commits']+=agents;expected['critic_adam_calls']+=plan['critic_epochs']
        expected['bridge_payload_bytes']+=m*ebytes;expected['messages']+=agents+1
        # Float64 episode vectors, separately from float32 policy wire/storage arrays.
        parameter_count=P//4
        expected['gradient_array_bytes']+=(N+2*m)*parameter_count*8
        expected['moment_scalar_entries']+=len(options)*m*parameter_count
        if row['method']=='bridge_control':
            x=r['control'];assert x['source_id']==str(r['source_id'])
            np.testing.assert_array_equal(x['probabilities_before'],ctrl.p)
            np.testing.assert_array_equal(x['queues_before'],ctrl.q)
            costs=np.asarray([[(N+o.episodes)*h,(N+o.episodes)*ebytes+P+C] for o in options])/row['charge_scales']
            ctrl.pending=dict(index=d['index'],costs=costs,queues=ctrl.q.copy(),probabilities=ctrl.p.copy(),
                              source_id=str(r['source_id']),current_hash=r['target_before'])
            replayed=ctrl.close(x['risk_feedback'],costs[d['index']])
            assert x==replayed
    assert previous==row['final_hash'] and source_total==c['source_transitions'] and bridge_total==c['bridge_transitions']
    assert expected==c and age
    return dict(task=row['task'],method=row['method'],payload_pass=True,actual_staleness=True,
                source_transitions=source_total,bridge_transitions=bridge_total,all_costs=c)


def analyze(root,replay,plan,output):
    output.mkdir(exist_ok=False)
    checks=[]
    hashes=json.loads((root/'hashes.json').read_text());other=json.loads((replay/'hashes.json').read_text())
    expected={f"{t['name']}__{m}.json" for t in plan['tasks'] for m in plan['methods']}
    assert set(hashes)==set(other)==expected
    initials={}
    for name in sorted(expected):
        assert sha(root/name)==hashes[name] and sha(replay/name)==other[name]
        a=json.loads((root/name).read_text());b=json.loads((replay/name).read_text())
        assert {k:v for k,v in a.items() if k!='cpu_seconds'}=={k:v for k,v in b.items() if k!='cpu_seconds'}
        checks.append(check(a,plan));check(b,plan)
        initials.setdefault(a['task'],set()).add(a['initial_hash'])
    assert all(len(v)==1 for v in initials.values())
    dump(output/'integrity.json',dict(cells=checks,originals=len(expected),replays=len(expected),returns_ranked=False))
    dump(output/'decision.json',dict(decision='OPERATIONALLY_QUALIFIED',performance_evidence=False,
        practical_precision_established=False,novelty_established=False,manuscript_admission=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',type=Path);ap.add_argument('--lock',type=Path);ap.add_argument('--root',type=Path)
    args=ap.parse_args();plan=json.loads((HERE/'qualification.json').read_text());fp=fingerprint()
    for key,name in [('numpy','required_numpy'),('torch','required_torch'),('gymnasium','required_gymnasium'),
                     ('robotics','required_robotics'),('mujoco','required_mujoco'),('harl_commit','required_harl')]:
        assert fp[key]==plan[name],(key,fp[key],plan[name])
    if args.freeze:
        if args.freeze.exists():raise FileExistsError(args.freeze)
        dump(args.freeze,dict(experiment=plan['experiment'],fingerprint=fp,plan=plan));return
    lock=json.loads(args.lock.read_text());assert lock['fingerprint']==fp and lock['plan']==plan
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for rel,h in fp['files'].items():
        content=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)
        assert hashlib.sha256(content).hexdigest()==h,rel
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    start=time.process_time();root=args.root;root.mkdir(exist_ok=False)
    execution=dict(experiment=plan['experiment'],pid=__import__('os').getpid(),source_commit=commit,
                   lock_sha256=sha(args.lock),state='running',completed=0)
    dump(root/'execution.json',execution)
    def budget():
        if time.process_time()-start>plan['cpu_seconds_limit']:raise TimeoutError('Frozen qualification CPU cap')
    try:
        siblings=[root,Path(str(root)+'_replay')]
        for index,directory in enumerate(siblings):
            if index:directory.mkdir(exist_ok=False)
            hashes={}
            for task in plan['tasks']:
                for method in plan['methods']:
                    budget();t=time.process_time();row=run_cell(task,plan['seed'],method,plan,budget)
                    row['cpu_seconds']=time.process_time()-t;check(row,plan)
                    name=f"{task['name']}__{method}.json";dump(directory/name,row);hashes[name]=sha(directory/name)
                    dump(directory/'hashes.json',hashes)
                    execution.update(completed=execution['completed']+1,phase='original' if index==0 else 'replay',cpu_seconds=time.process_time()-start)
                    dump(root/'execution.json',execution)
            dump(directory/'fingerprint.json',fp)
        a,b=Path(str(root)+'_analysis_a'),Path(str(root)+'_analysis_b')
        analyze(root,siblings[1],plan,a);analyze(root,siblings[1],plan,b)
        for name in ['integrity.json','decision.json']:assert (a/name).read_bytes()==(b/name).read_bytes()
        execution.update(state='complete',phase='complete',cpu_seconds=time.process_time()-start)
        dump(root/'execution.json',execution);dump(root/'workflow_complete.json',dict(**execution,byte_identical_analyses=True))
        print(json.dumps(dict(state='complete',cpu_seconds=execution['cpu_seconds'],performance_evidence=False)))
    except BaseException as e:
        execution.update(state='failed',error_type=type(e).__name__,error=str(e),cpu_seconds=time.process_time()-start)
        dump(root/'execution.json',execution);raise


if __name__=='__main__':main()
