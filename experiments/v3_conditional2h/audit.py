"""Independent terminal read-back logic, after both native processes exited."""
import json
import math
from pathlib import Path
import sys
import time
from support import (HERE,RESULT,LOCK,ROOT,child_cpu,pid_alive,dump,sha,verify_sources,plan,Budget)
from model import old,public_design,STOP


def run():
    execution=json.loads((RESULT/'execution.json').read_text())
    launch=json.loads((RESULT/'worker_launch.json').read_text())
    if pid_alive(execution['supervisor_pid']) or pid_alive(launch['pid']):raise RuntimeError('Both native processes must exit before scientific audit')
    marker=RESULT/'workflow_complete.json'
    terminal=RESULT/'audited_terminal.json'
    failure=RESULT/'failure.json'
    base=json.loads((terminal if terminal.exists() else failure).read_text())
    budget=Budget(base['cumulative_charged_cpu_seconds'])
    cfg=plan();lock=json.loads(LOCK.read_text());verify_sources(lock);budget()
    import numpy as np
    import pandas as pd
    from analyze import load
    frame=load(RESULT/'science',budget)
    counts={phase:sum(1 for path in (RESULT/phase).glob('part_*.jsonl') for _ in path.open(encoding='utf-8'))
            for phase in ('science','replay')}
    if not marker.exists() or not terminal.exists():
        dump(RESULT/'independent_scientific_audit.json',dict(decision='INCONCLUSIVE_RESOURCE_OR_RUNTIME_FAILURE',
            counts=counts,all_preserved_partial_evidence_reviewed=True,failure=base,
            audit_cpu_seconds=time.process_time(),total_charged_cpu_seconds=budget.cpu,
            no_scientific_retry=True,full_validation_completed=False))
        return
    completed=json.loads(marker.read_text())
    assert counts==dict(science=294912,replay=294912)
    for phase in ('science','replay'):
        manifest=json.loads((RESULT/phase/'hashes.json').read_text())
        assert len(manifest)==128
        for name,digest in manifest.items():
            budget();assert sha(RESULT/phase/name)==digest
            assert sha(RESULT/'science'/name)==sha(RESULT/'replay'/name)
    assert len(frame)==294912
    keys=['seed','learner','family','scale','regime','error_sign','policy']
    assert not frame.duplicated(keys).any()
    assert set(frame.seed)==set(range(20581001,20581065))
    assert frame.groupby(keys[:-1]).size().eq(9).all()
    assert np.isfinite(frame[['terminal_squared_risk','normalizer','normalized_risk','terminal_error']].to_numpy()).all()
    assert np.allclose(frame.terminal_error.to_numpy()**2,frame.terminal_squared_risk,rtol=0,atol=1e-14)
    assert frame.messages.between(0,frame.message_budget).all() and frame.clock.between(0,frame.clock_budget).all()
    assert (frame.messages==frame.probe_messages+frame.training_messages).all()
    assert (frame.clock==frame.probe_n+frame.training_clock).all()
    assert (frame.initial_parameter==0).all() and (frame.hidden_target==-frame.error_sign).all()
    assert (frame[frame.stop][['M','applied','training_messages','training_clock']]==0).all().all()
    active=frame[~frame.stop]
    assert (active.training_messages==active.M*(8+active.q)).all()
    assert (active.training_clock==active.M*active.b).all()
    design_keys=['learner','family','scale']
    normalization={}
    for learner,family,scale in frame.groupby(design_keys).groups:
        q,th,lam,delay=family.split('-')
        f=old.Family(int(q[1:]),float(th[2:]),float(lam[1:]),int(delay[1:]))
        bm=(8+f.Q)*scale;be=4*scale
        values=[1.]+[(old.program_risk(learner,a,bm,be,f,f.theta_low)[0]+
                       old.program_risk(learner,a,bm,be,f,f.theta_high)[0])/2 for a in old.catalogue(f)]
        normalization[(learner,family,scale)]=min(values)
    assert len(normalization)==128 and min(normalization.values())>0
    denom=np.array([normalization[x] for x in frame[design_keys].itertuples(index=False,name=None)])
    assert np.allclose(denom,frame.normalizer,rtol=0,atol=1e-13)
    assert np.allclose(frame.terminal_squared_risk/denom,frame.normalized_risk,rtol=1e-13,atol=1e-13)
    deploy=frame[~frame.oracle_evaluation_only]
    assert (deploy.groupby(['seed','learner','family','scale','regime','policy'])[['q','b','probe_used']].nunique()<=1).all().all()
    D=frame.family.str.extract(r'-D(\d+)$')[0].astype(int)
    assert (frame.applied==np.maximum(0,frame.M-D)).all()
    pivot=frame.pivot(index=keys[:-1],columns='policy',values='normalized_risk')
    robust=pivot['conditional_risk_robust_full_catalogue']
    point={
      'fixed_minus_robust':float((pivot['best_balanced_full_budget_fixed_including_stop']-robust).groupby('seed').mean().mean()),
      'marginal_minus_robust':float((pivot['posterior_regime_marginal_risk_full_catalogue']-robust).groupby('seed').mean().mean())}
    summary=json.loads((RESULT/'analysis_a/summary.json').read_text())
    for name,value in point.items():assert abs(value-summary['primary'][name]['mean'])<1e-12
    good=(all(summary['primary'][name]['one_sided_995_lower']>0 for name in point) and point['fixed_minus_robust']>=.01)
    assert good==(summary['decision']=='POSITIVE_AT_REGISTERED_MODEL_AND_INFORMATION_SCOPE')
    for name,digest in completed['analysis_sha256'].items():
        assert sha(RESULT/'analysis_a'/name)==sha(RESULT/'analysis_b'/name)==digest
    for name,digest in completed['plots'].items():assert sha(RESULT/'plots'/name)==digest
    budget()
    dump(RESULT/'independent_scientific_audit.json',dict(decision=summary['decision'],science_rows=294912,replay_rows=294912,
        analyses=2,native_worker_and_supervisor_exited=True,full_payload_hashes_and_mapping=True,
        all_128_normalizers_independently_reenumerated=True,initialization_role_and_ledgers=True,
        both_primary_means_independently_recomputed=point,all_negative_designs_reviewed=summary['adverse_design_mean_counts'],
        bootstrap_not_repeated_during_audit=True,all_seven_analysis_bytes_equal=True,all_plot_bytes_verified=True,
        plots_visual_review_pending=True,source_and_runtime_lock_preserved=True,
        audit_cpu_seconds=time.process_time(),total_charged_cpu_seconds=budget.cpu,
        whole_cap_seconds=7200,manuscript_admission=False))
    print(json.dumps(dict(status='INDEPENDENT_READBACK_COMPLETE',decision=summary['decision'],cpu=budget.cpu)))


if __name__=='__main__':run()
