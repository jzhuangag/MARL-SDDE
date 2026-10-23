"""Read-only independent terminal audit. Never invokes the frozen analyzer/runner."""
import time
import json
import csv
import math
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT/'ICML-2027/results/sealed_bridge_learning_dev140h_20261018'
OLD = ROOT/'ICML-2027/results/sealed_bridge_learning_dev40h_20261016'
OUT = ROOT/'ICML-2027/internal/sealed_bridge_terminal_audit_20261020'
CONTROLS = ['fresh_happo_half','fresh_happo','fresh_happo_double','fresh_mappo',
            'delayed_happo','online_joint_kl','joint_ma_trace_adaptation',
            'defensive_page_pg_adaptation','fresh_pg']

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')

def finite(v):
    if isinstance(v,dict):
        return all(finite(x) for x in v.values())
    if isinstance(v,list):
        return all(finite(x) for x in v)
    return not isinstance(v,float) or math.isfinite(v)

def same(a,b):
    assert np.allclose(a,b,rtol=1e-10,atol=1e-10), (a,b)

def key(seed,ti,ordinal,role,e):
    return int.from_bytes(hashlib.sha256(f'{seed}:{ti}:{ordinal}:{role}:{e}'.encode()).digest()[:8],'big') & ((1<<63)-1)

def check_ids(ids,r,ordinal,role):
    assert ids == [key(r['seed'],r['task_index'],ordinal,role,e) for e in range(len(ids))]

def simplex(x):
    ordered=np.sort(x)[::-1]
    thresholds=(np.cumsum(ordered)-1)/np.arange(1,len(x)+1)
    return np.maximum(x-thresholds[np.where(ordered>thresholds)[0][-1]],0)

def series(r,axis):
    pairs={c['training_transitions'] if axis=='native' else c['costs'][axis]: c['mean_return'] for c in r['curves']}
    return list(zip(*sorted(pairs.items())))

def area(r,axis,lo,hi):
    xs,ys=series(r,axis)
    # Independent scalar segment integration; do not call frozen auc/analyze.
    grid=[lo]+[x for x in xs if lo<x<hi]+[hi]
    vals=np.interp(grid,xs,ys)
    return sum((grid[i+1]-grid[i])*(vals[i]+vals[i+1])/2 for i in range(len(grid)-1))/(hi-lo)

def inspect(r,plan):
    task=plan['tasks'][r['task_index']]; h=task['horizon']; costs=r['costs']; roles=plan['roles']
    assert finite(r) and r['training_transitions']==task['training_transitions']
    assert r['task']==task['name'] and r['all_sources_drained']
    assert not any(r[x] for x in ['evaluation_used_by_controller','current_return_ranked','physical_parallel_speedup_claimed'])
    sums={k:0 for k in costs if k not in ['logical_bytes','work_rows']}
    for event in r['charge_event_totals'].values():
        assert event['calls']>0
        for k,v in event.items():
            assert isinstance(v,int) and v>=0
            if k!='calls': sums[k]+=v
    assert all(sums[k]==costs[k] for k in sums)
    assert costs['logical_bytes']==costs['program_upload_bytes']+costs['policy_broadcast_bytes']
    assert costs['work_rows']==sum(costs[k] for k in ['sampling_actor_rows','actor_forward_rows','actor_backward_rows','critic_forward_rows','critic_backward_rows'])
    paid=0; reservations={}; closed=set(); eval_paid={'benchmark':0,'frozen_policy':0}
    for e in r['native_budget_events']:
        kind=e['kind']; sid=e.get('source_id')
        if kind=='reserve_launch':
            assert sid not in reservations and sid not in closed
            assert e['source']>=2*h and e['receipt']>=e['minimum_receipt']>=2*h
            reservations[sid]=[e['source'],e['receipt'],e['minimum_receipt'],False,False]
        elif kind=='source_charge':
            s=reservations[sid]; assert not s[3] and e['transitions']==s[0]
            paid+=s[0]; s[0]=0; s[3]=True
        elif kind=='source_complete':
            s=reservations[sid]; assert s[3] and not s[4]; s[4]=True
        elif kind=='receipt_charge':
            s=reservations.pop(sid); assert s[4] and e['transitions']>=s[2]
            paid+=e['transitions']; closed.add(sid)
        elif kind=='evaluation_charge': eval_paid[e['role']]+=e['transitions']
        else:
            assert not reservations and e['transitions']>0 and e['transitions']%h==0
            paid+=e['transitions']
        assert paid+sum(s[0]+s[1] for s in reservations.values())<=task['training_transitions']
    assert not reservations and paid==task['training_transitions']==costs['native_training']
    expected=(plan['evaluation_intervals']+1)*plan['evaluation_episodes']*h
    assert eval_paid=={'benchmark':expected,'frozen_policy':expected}
    assert costs['native_benchmark']==costs['native_frozen_policy']==expected
    training=[]; evaluation=[]; launches={}; seen=set(); ages=[]; version=0; prior=r['initial_hash']
    for ordinal,l in enumerate(r['launches']):
        assert l['id']==ordinal
        delay=plan['delay_min']+int.from_bytes(hashlib.sha256(f'{r["seed"]}:sealed_learning_service:{ordinal}'.encode()).digest()[:8],'big')%(plan['delay_max']-plan['delay_min']+1)
        assert l['due']==l['launch_tick']+delay
        check_ids(l['source_ids'],r,ordinal,roles['source']); training+=l['source_ids']; launches[ordinal]=l
    controller_rng=np.random.default_rng(int(np.random.SeedSequence([r['seed'],r['task_index'],0,roles['controller'],0]).generate_state(1)[0]%(2**31-1)))
    qstate=None; ordinal_fresh=0; ordinal_terminal=0; diagnostics=[]
    for t in r['transactions']:
        kind=t['kind']; assert not t.get('evaluation_used',False) and not t.get('current_return_ranked',False)
        if 'before' in t:
            assert t['before']==prior; prior=t['after']
        if 'data_ids' in t:
            if kind=='charged_calibration': ordinal,role=0,roles['calibration']
            elif kind=='terminal_fresh': ordinal,role=ordinal_terminal,roles['terminal']; ordinal_terminal+=1
            else: ordinal,role=ordinal_fresh,roles['fresh']; ordinal_fresh+=1
            check_ids(t['data_ids'],r,ordinal,role); training+=t['data_ids']
        if kind=='charged_calibration':
            assert not t['actor_updated'] and len(t['data_ids'])==plan['calibration_episodes']; continue
        if kind in ['bridge_receipt','delayed_reference_receipt']:
            sid=t['source_id']; l=launches[sid]
            assert sid not in seen; seen.add(sid)
            assert t['source_version']==l['source_version'] and t['target_version']==version
            assert t['tick']>=l['due'] and t['actual_version_age']==version-l['source_version']>=0
            ages.append(t['actual_version_age'])
        if kind=='bridge_receipt':
            assert t['source_hash']==l['source_hash'] and t['data_hash']==l['data_hash']
            assert t['source_consumed_once'] and t['decision']['before_source_and_bridge']
            assert t['actor_step']['before_source_and_bridge'] and not t['actor_step']['neural_smoothness_certificate']
            n=plan['source_episodes']+t['decision']['episodes']
            same(t['actor_step']['alpha'],t['actor_step']['alpha_reference']*n/plan['fresh_batch_episodes'])
            assert t['actor_step']['paid_episode_exposure']==n
            check_ids(t['bridge_ids'],r,sid,roles['bridge']); training+=t['bridge_ids']
            modes=[int(np.random.default_rng(key(r['seed'],r['task_index'],sid,roles['bridge']+200000,e)).random()<plan['source_fraction']) for e in range(len(t['bridge_ids']))]
            assert t['complete_policy_modes']==modes and len(modes)==t['decision']['episodes']
            options=[(m,b) for m in plan['bridge_episodes'] for b in ([0.] if r['method']=='no_transport' else plan['betas'])]
            ix=t['decision']['index']; assert options[ix]==(t['decision']['episodes'],t['decision']['beta'])
            resources=np.asarray(t['resources']); durations=np.array([plan['source_episodes']+m for m,b in options],float)
            A=3 if r['task_index']==0 else 2; E=plan['critic_epochs']
            assert all(resources[j,0]==h*((3*A+1)*plan['source_episodes']+(5*A+3+2*E)*m) for j,(m,b) in enumerate(options))
            epbytes=(resources[-1,1]-resources[0,1])/(options[-1][0]-options[0][0])
            assert all(resources[j,1]==(plan['source_episodes']+m)*epbytes+3*r['policy_parameter_bytes'] for j,(m,b) in enumerate(options))
            before=t['costs_before_receipt']; after=t['costs']
            actual=[after['work_rows']-before['work_rows']+A*plan['source_episodes']*h,after['logical_bytes']-before['logical_bytes']+plan['source_episodes']*epbytes+r['policy_parameter_bytes']]
            assert actual==t['actual_resources']==resources[ix].tolist()
            if r['method']=='fixed_allocation':
                assert options[ix]==(plan['fixed_bridge_episodes'],plan['fixed_beta'])
            else:
                scales=(resources/durations[:,None]).max(axis=0); norm=resources/scales
                ends=[int(durations.argmin()),int(durations.argmax())]
                witness=np.mean(norm[ends]/durations[ends,None],axis=0); target=(1+witness)/2
                debts=norm-durations[:,None]*target
                if qstate is None: qstate=(np.ones(len(options))/len(options),np.zeros(2),0.)
                w,q,energy=qstate; inv=w/durations; probs=inv/inv.sum(); expected_duration=1/inv.sum()
                assert int(controller_rng.choice(len(options),p=probs))==ix
                losses=t['risk_weight']*durations*np.array(t['risk_feedback'])
                if r['method']!='no_price': losses+=(debts@q+.5*np.square(debts).sum(-1))/durations
                gradient=expected_duration*(losses-losses.mean()); energy+=float(gradient@gradient)
                wnext=simplex(w-math.sqrt(2)/math.sqrt(1e-12+energy)*gradient); qnext=np.maximum(0,q+debts[ix])
                f=t['feedback']
                for field,value in [('exposure_before',w),('receipt_probabilities',probs),('exposure_after',wnext),('queues_before',q),('queues_after',qnext),('normalized_risk_losses',losses),('resource_scales',scales),('resource_rate_targets',target),('static_slater_slack',target-witness)]: same(f[field],value)
                same(f['full_feedback_energy'],energy); assert not f['current_data_selected_action'] and not f['variance_upper_certificate']
                qstate=(wnext,qnext,energy)
            diagnostics.append(dict(m=t['decision']['episodes'],beta=t['decision']['beta'],alpha=t['actor_step']['alpha'],staleness=t['actual_version_age'],risk_feedback_selected=t['risk_feedback'][ix],overlap=np.mean(t['prefix_overlap_mean_by_time'])))
        version+=int(not(kind=='delayed_reference_receipt' and not t['details'].get('accepted',True)))
    assert prior==r['final_hash'] and seen==set(launches)
    assert not launches or max(ages)>0
    assert len(training)==len(set(training)) and len(training)*h==costs['native_training']
    assert len(r['curves'])==plan['evaluation_intervals']+1
    for i,c in enumerate(r['curves']):
        assert c['index']==i and not c['evaluation_used_by_controller']
        check_ids(c['evaluation_ids'],r,i,roles['evaluation']); evaluation+=c['evaluation_ids']
        assert len(c['benchmark_returns'])==len(c['frozen_returns'])==plan['evaluation_episodes']
        same(c['mean_return'],np.mean(c['benchmark_returns'])); same(c['frozen_mean_return'],np.mean(c['frozen_returns']))
    assert len(evaluation)==len(set(evaluation)) and set(training).isdisjoint(evaluation)
    assert r['curves'][-1]['training_transitions']==task['training_transitions'] and r['curves'][-1]['policy_version']==version
    assert r['curves'][-1]['costs']==costs
    return dict(task=r['task'],seed=r['seed'],method=r['method'],costs=costs,cpu_seconds=r['cpu_seconds'],max_actual_staleness=max(ages,default=0),policy_versions=version,training_ids=len(training),evaluation_ids=len(evaluation),bridge_diagnostics=diagnostics)

def main():
    OUT.mkdir(exist_ok=False)
    marker=read(NATIVE/'workflow_complete.json'); lock=read(ROOT/'ICML-2027/internal/sealed_bridge_learning_140h_lock_20261018.json'); plan=lock['plan']
    assert marker['driver_pid']==19816 and marker['state']=='complete'
    # Independent OS liveness check, not an assumption from the completion marker.
    import ctypes
    k=ctypes.WinDLL('kernel32',use_last_error=True); k.OpenProcess.restype=ctypes.c_void_p
    handle=k.OpenProcess(0x1000,False,19816)
    if handle:
        code=ctypes.c_ulong(); k.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code)); k.CloseHandle(ctypes.c_void_p(handle)); assert code.value!=259
    assert not (NATIVE/'workflow_failed.json').exists()
    execution=read(NATIVE/'execution.json'); expected={f'cell_{ti}_{s}_{m}.json':(ti,s,m) for ti in range(3) for s in plan['seeds'] for m in plan['methods']}
    repeats={f'cell_{ti}_{plan["seeds"][0]}_{m}.json':(ti,plan['seeds'][0],m) for ti in range(3) for m in ['fresh_happo','full']}
    data={}; checks=[]; manifests={}
    for directory,mapping in [(NATIVE,expected),(Path(str(NATIVE)+'_replay'),repeats)]:
        manifest=read(directory/'hashes.json'); assert set(manifest)==set(mapping)=={p.name for p in directory.glob('cell_*.json')}
        manifests[directory.name]=manifest
        for name,identity in mapping.items():
            assert sha(directory/name)==manifest[name]
            r=read(directory/name); assert (r['task_index'],r['seed'],r['method'])==identity
            result=inspect(r,plan); result['phase']='science' if directory==NATIVE else 'replay'; checks.append(result)
            if directory==NATIVE: data[identity]=r
            else: assert {k:v for k,v in r.items() if k!='cpu_seconds'}=={k:v for k,v in data[identity].items() if k!='cpu_seconds'}
    prefix=lock['preserved_prefix']; assert prefix['completed']==80 and len(prefix['manifest'])==80
    assert read(NATIVE/'imported_payload_hashes.json')==prefix['manifest']
    for name,digest in prefix['manifest'].items(): assert sha(OLD/name)==sha(NATIVE/name)==digest
    for field,name in [('failure_sha256','workflow_failed.json'),('partial_costs_sha256','failed_cell_partial_costs.json'),('execution_sha256','execution.json'),('prefix_manifest_sha256','hashes_0080.json')]: assert sha(OLD/name)==prefix[field]
    assert read(OLD/'workflow_failed.json')['cpu_seconds']==prefix['prior_cpu_seconds']==144002.671875
    assert read(OLD/'failed_cell_partial_costs.json')['cpu_seconds']==prefix['unfinished_cpu_seconds']==344.171875
    source_hashes={**lock['fingerprint']['files'],**lock['extension_files']}
    for name,digest in source_hashes.items():
        assert sha(ROOT/name)==digest
        assert hashlib.sha256(subprocess.check_output(['git','show',execution['source_commit']+':'+name],cwd=ROOT)).hexdigest()==digest
    sys.path.insert(0,str(ROOT/'ICML-2027/experiments'))
    from sealed_bridge_learning.guard import fingerprint
    assert fingerprint()==lock['fingerprint']
    for ti in range(3):
        for s in plan['seeds']:
            rows=[data[ti,s,m] for m in plan['methods']]
            assert len({r['initial_hash'] for r in rows})==1
            assert all(r['curves'][0]['benchmark_returns']==r['curves'][0]['frozen_returns'] for r in rows)
            assert all([c['frozen_returns'] for c in r['curves']]==[c['frozen_returns'] for c in rows[0]['curves']] for r in rows)
    a,b=[Path(str(NATIVE)+'_analysis_'+tag) for tag in ['a','b']]
    for name in ['metrics.csv','summary.json','integrity.json']: assert (a/name).read_bytes()==(b/name).read_bytes()
    reported=list(csv.DictReader((a/'metrics.csv').open(encoding='utf-8'))); assert len(reported)==117
    summary=read(a/'summary.json'); task_results=[]; aggregate=[]
    for ti,task in enumerate(plan['tasks']):
        group=[r for ident,r in data.items() if ident[0]==ti]; axes={'native':(0,task['training_transitions'])}
        for axis in ['messages','logical_bytes']: axes[axis]=(max(series(r,axis)[0][0] for r in group),min(series(r,axis)[0][-1] for r in group))
        values={m:np.array([area(data[ti,s,m],'native',*axes['native']) for s in plan['seeds']]) for m in plan['methods']}
        strongest=max(CONTROLS,key=lambda m:values[m].mean()); floor=plan['gates']['normalization_floor']
        gains=100*(values['full']-values[strongest])/max(abs(values[strongest].mean()),floor)
        final=[]; learning=[]; work=[]; byte=[]
        for s in plan['seeds']:
            f=data[ti,s,'full']; ref=data[ti,s,strongest]; rf=ref['curves'][-1]['mean_return']; frozen=ref['curves'][-1]['frozen_mean_return']
            final.append(100*(f['curves'][-1]['mean_return']-rf)/max(abs(rf),floor)); learning.append(100*(rf-frozen)/max(abs(frozen),floor))
            work.append(f['costs']['work_rows']/ref['costs']['work_rows']); byte.append(f['costs']['logical_bytes']/ref['costs']['logical_bytes'])
        g=plan['gates']; gates=dict(native_auc=float(gains.mean())>=g['native_auc_gain_percent_min'],paired_seed_count=int((gains>0).sum())>=g['positive_seed_count_min'],final_return=float(np.mean(final))>=g['final_change_percent_min'],strong_control_learning=float(np.mean(learning))>=g['strong_control_learning_percent_min'],work=float(np.mean(work))<=g['work_ratio_max'],logical_bytes=float(np.mean(byte))<=g['logical_byte_ratio_max'])
        result=dict(task=task['name'],strongest_control=strongest,mean_native_auc_gain_percent=float(gains.mean()),positive_seeds=int((gains>0).sum()),paired_native_auc_gain_percent=gains.tolist(),mean_final_change_percent=float(np.mean(final)),strongest_control_learning_percent=float(np.mean(learning)),mean_work_ratio=float(np.mean(work)),mean_logical_byte_ratio=float(np.mean(byte)),gates=gates)
        prior=summary['tasks'][ti]
        for field,value in result.items():
            if isinstance(value,(int,float,list)): same(value,prior[field])
            else: assert value==prior[field]
        task_results.append(result)
        for m in plan['methods']:
            rows=[data[ti,s,m] for s in plan['seeds']]
            ag=dict(task=task['name'],method=m)
            for axis in axes: ag[axis+'_auc_mean']=float(np.mean([area(r,axis,*axes[axis]) for r in rows]))
            ag['final_return_mean']=float(np.mean([r['curves'][-1]['mean_return'] for r in rows]))
            for cost in rows[0]['costs']: ag[cost+'_mean']=float(np.mean([r['costs'][cost] for r in rows]))
            ag['cpu_seconds_sum']=sum(r['cpu_seconds'] for r in rows); aggregate.append(ag)
            for r in rows:
                csvrow=next(x for x in reported if x['task']==r['task'] and int(x['seed'])==r['seed'] and x['method']==m)
                for axis,col in [('native','native_auc'),('messages','message_auc'),('logical_bytes','logical_byte_auc')]: same(float(csvrow[col]),area(r,axis,*axes[axis]))
                for cost,v in r['costs'].items(): assert int(csvrow[cost])==v
    assert not all(all(t['gates'].values()) for t in task_results)
    assert marker['decision']==summary['decision']=='DEVELOPMENT_NO_GO'
    # Read and inventory ALL preserved profile/old-stop/current artifacts, not only survivors.
    inventory={}
    for directory in (ROOT/'ICML-2027/results').glob('sealed_bridge_learning_*'):
        if directory.is_dir():
            for p in directory.rglob('*'):
                if p.is_file(): inventory[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'preserved_inventory.json',inventory)
    save(OUT/'payload_checks.json',checks)
    save(OUT/'all_method_costs_and_auc.json',aggregate)
    save(OUT/'task_gates.json',task_results)
    plot_paths=list(Path(str(NATIVE)+'_plots').glob('*')); assert len(plot_paths)==6
    save(OUT/'audit.json',dict(status='MACHINE_CHECKS_COMPLETE_AGENT_SCIENTIFIC_REVIEW_REQUIRED',independent_implementation_no_frozen_analyzer_invoked=True,originals=117,replays=6,imported_byte_identical=80,source_files=27,runtime_fingerprint_match=True,semantic_replay=True,double_analysis_byte_equal=True,initial_and_frozen_policies_matched=True,all_ledgers_roles_versions_service_and_controller_reconstructed=True,task_results=task_results,terminal_marker_sha256=sha(NATIVE/'workflow_complete.json'),native_recorded_cpu_seconds=marker['cpu_seconds'],native_shutdown_cpu_not_observed_after_last_marker=True,audit_process_cpu_seconds=time.process_time(),auxiliary_and_unmeasured_shutdown_conservative_charge_seconds=60,cumulative_charged_cpu_seconds=marker['cpu_seconds']+time.process_time()+60,cap_seconds=504000,inventory_files=len(inventory),plots={p.name:sha(p) for p in plot_paths},payloads_sha256=sha(OUT/'payload_checks.json'),inventory_sha256=sha(OUT/'preserved_inventory.json')))
    print(json.dumps(dict(status='audit machine checks complete',out=str(OUT),audit_cpu=time.process_time(),tasks=task_results),allow_nan=False))

if __name__=='__main__': main()
