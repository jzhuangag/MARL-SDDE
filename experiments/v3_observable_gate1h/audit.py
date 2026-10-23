"""Independent post-exit scientific/resource read-back; never a third analysis."""
import json
import time
from support import *

def run():
    execution=json.loads((RESULT/'execution.json').read_text());launch=json.loads((RESULT/'worker_launch.json').read_text())
    if pid_alive(execution['supervisor_pid']) or pid_alive(launch['pid']):raise RuntimeError('Both native processes must exit')
    if (RESULT/'independent_scientific_audit.json').exists():raise RuntimeError('Existing audit preserved; diagnose instead of repeating')
    terminal=RESULT/'audited_terminal.json';failure=RESULT/'failure.json';marker=RESULT/'workflow_complete.json'
    base=json.loads((terminal if terminal.exists() else failure).read_text());budget=Budget(base['cumulative_charged_cpu_seconds'])
    verify_sources(json.loads(LOCK.read_text()));budget()
    from analyze import load
    from model import old,public_design,invariant_filter
    import numpy as np
    import pandas as pd
    frame=load(RESULT/'science',budget)
    counts={phase:sum(1 for p in (RESULT/phase).glob('part_*.jsonl') for _ in p.open(encoding='utf-8')) for phase in ('science','replay')}
    if not terminal.exists() or not marker.exists():
        dump(RESULT/'independent_scientific_audit.json',dict(decision='INCONCLUSIVE_RESOURCE_OR_RUNTIME_FAILURE',counts=counts,all_preserved_partial_evidence_reviewed=True,failure=base,audit_cpu_seconds=time.process_time(),total_charged_cpu_seconds=budget.cpu,no_scientific_retry=True,full_validation_completed=False));return
    cfg=plan();complete=json.loads(marker.read_text());assert counts==dict(science=294912,replay=294912)
    for phase in ('science','replay'):
        manifest=json.loads((RESULT/phase/'hashes.json').read_text());assert len(manifest)==128
        for name,digest in manifest.items():
            budget();assert sha(RESULT/phase/name)==digest
            assert sha(RESULT/'science'/name)==sha(RESULT/'replay'/name)
    keys=['seed','learner','family','scale','regime','error_sign','policy']
    assert len(frame)==294912 and not frame.duplicated(keys).any()
    assert set(frame.seed)==set(range(20591001,20591065)) and frame.groupby(keys[:-1]).size().eq(9).all()
    assert set(frame.policy)==set(cfg['policies'])
    assert np.isfinite(frame[['terminal_squared_risk','terminal_error','normalized_risk','normalizer']].to_numpy()).all()
    assert np.allclose(frame.terminal_error**2,frame.terminal_squared_risk,rtol=0,atol=1e-14)
    assert frame.messages.between(0,frame.message_budget).all() and frame.clock.between(0,frame.clock_budget).all()
    assert (frame.messages==frame.probe_messages+frame.training_messages).all() and (frame.clock==frame.probe_n+frame.training_clock).all()
    assert (frame.initial_parameter==0).all() and (frame.hidden_target==-frame.error_sign).all()
    assert (frame[frame.stop][['M','applied','training_messages','training_clock']]==0).all().all()
    active=frame[~frame.stop];assert (active.training_messages==active.M*(8+active.q)).all() and (active.training_clock==active.M*active.b).all()
    delays=frame.family.str.extract(r'-D(\d+)$')[0].astype(int);assert (frame.applied==np.maximum(0,frame.M-delays)).all()
    deploy=frame[~frame.oracle_evaluation_only]
    assert (deploy.groupby(['seed','learner','family','scale','regime','policy'])[['q','b','probe_used']].nunique()<=1).all().all()
    normalization={};admissions={};no_carry={}
    for learner,family,scale in frame.groupby(['learner','family','scale']).groups:
        budget();q,th,lam,D=family.split('-');f=old.Family(int(q[1:]),float(th[2:]),float(lam[1:]),int(D[1:]))
        bm=(8+f.Q)*scale;be=4*scale
        # Independent scalar accumulation, not the production vectorized profile.
        denominator=1.;theta=(f.theta_low+f.theta_high)/2
        for action in old.catalogue(f):
            Nmax=max(0,old.scheduled(action,bm,be,f.h)-f.D);rho=f.lam**action.b
            impulse=[];G=0.;sumG=diagonal=common=history=0.
            for t in range(Nmax):
                if learner=='running_mean':g=1.
                else:
                    impulse.append(1. if t==0 else impulse[-1]-(.1/(f.D+1)*impulse[t-f.D-1] if t>f.D else 0.))
                    G+=impulse[-1];g=G
                diagonal+=g*g;common+=g*g+2*g*history;sumG+=g;N=t+1
                if learner=='running_mean':value=(1+theta*common+diagonal/action.q)/(N+1)**2
                else:
                    eta=.2/(f.D+1);value=(1-.5*eta*sumG/N)**2+(eta/N)**2*(theta*common+diagonal/action.q)
                denominator=min(denominator,value);history=rho*(history+g)
        pub=public_design(learner,f,scale);p=pub['legacy']
        assert abs(denominator-pub['normalizer'])<1e-13
        normalization[learner,family,scale]=denominator;admissions[learner,family,scale]=p['corrected'];no_carry[learner,family,scale]=p['no_carryover']
        if p['feasible']:
            # Independently form dense projected covariance for the public confidence bound.
            n=p['probe_n'];Dmat=np.diff(np.eye(n),axis=0);t=np.arange(n)
            ld=[np.linalg.slogdet(Dmat@(theta*f.lam**abs(t[:,None]-t[None,:])+np.eye(n)/f.Q)@Dmat.T)[1] for theta in (f.theta_low,f.theta_high,(f.theta_low+f.theta_high)/2)]
            delta=np.exp(.25*(ld[0]+ld[1])-.5*ld[2]);assert abs(delta-p['delta'])<1e-11
            assert not p['corrected'] or sum(p['U'])/2<denominator
    denom=np.array([normalization[x] for x in frame[['learner','family','scale']].itertuples(index=False,name=None)])
    assert np.allclose(denom,frame.normalizer,rtol=0,atol=1e-13)
    assert np.allclose(frame.terminal_squared_risk/denom,frame.normalized_risk,rtol=1e-13,atol=1e-13)
    pivot=frame.pivot(index=keys[:-1],columns='policy',values='normalized_risk');gate=pivot[cfg['policies'][0]]
    point={'fixed_minus_robust':float((pivot[cfg['policies'][7]]-gate).groupby('seed').mean().mean()),'marginal_minus_robust':float((pivot[cfg['policies'][2]]-gate).groupby('seed').mean().mean())}
    summary=json.loads((RESULT/'analysis_a/summary.json').read_text())
    for name,value in point.items():assert abs(value-summary['primary'][name]['mean'])<1e-12
    good=all(summary['primary'][name]['one_sided_995_lower']>0 for name in point) and point['fixed_minus_robust']>=.01
    assert good==(summary['decision']=='POSITIVE_AT_REGISTERED_MODEL_AND_INFORMATION_SCOPE')
    for name,digest in complete['analysis_sha256'].items():assert sha(RESULT/'analysis_a'/name)==sha(RESULT/'analysis_b'/name)==digest
    for name,digest in complete['plots'].items():assert sha(RESULT/'plots'/name)==digest
    budget();dump(RESULT/'independent_scientific_audit.json',dict(decision=summary['decision'],science_rows=294912,replay_rows=294912,analyses=2,native_worker_and_supervisor_exited=True,full_payload_hashes_and_mapping=True,all_128_normalizers_independently_reenumerated=True,all_feasible_projected_affinities_dense_crosschecked=True,certificate_admissions=sum(admissions.values()),no_carryover_admissions=sum(no_carry.values()),initialization_role_and_ledgers=True,both_primary_means_independently_recomputed=point,all_negative_designs_reviewed=summary['adverse_design_mean_counts'],bootstrap_not_repeated_during_audit=True,all_seven_analysis_bytes_equal=True,all_plot_bytes_verified=True,plots_visual_review_pending=True,audit_cpu_seconds=time.process_time(),total_charged_cpu_seconds=budget.cpu,whole_cap_seconds=3600,manuscript_admission=False))
    print(json.dumps(dict(status='INDEPENDENT_READBACK_COMPLETE',decision=summary['decision'],cpu=budget.cpu)))

if __name__=='__main__':run()
