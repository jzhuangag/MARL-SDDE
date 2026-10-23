"""Independent post-exit read-back; no trajectories or third bootstrap analysis."""
import time
import json
import hashlib
import math
import ctypes
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'experiments/v3_risk_gate8h'
DATA=PKG/'results/formal_20261019'
OUT=ROOT/'ICML-2027/internal/v3_risk_gate8h_terminal_20261020'

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,obj):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')

def main():
    OUT.mkdir(exist_ok=False)
    sys.path.insert(0,str(PKG))
    from guard import pid_alive
    from core import Family,design,POLICIES,LEARNERS
    complete=read(DATA/'workflow_complete.json'); terminal=read(DATA/'audited_terminal.json')
    assert not pid_alive(13756) and not pid_alive(23016)
    assert read(DATA/'worker_exit.json')['exit_code']==0
    assert complete['science_rows']==complete['replay_rows']==196608 and complete['analysis_copies']==2
    assert terminal['worker_exited'] and terminal['all_analysis_bytes_identical']
    assert sha(DATA/'workflow_complete.json')==terminal['workflow_complete_sha256']
    assert not (DATA/'worker_failure.json').exists() and not (DATA/'failure.json').exists()
    lock=read(PKG/'registration_v2/lock.json'); admission=read(PKG/'registration_v2/admission.json')
    assert sha(PKG/'registration_v2/lock.json')==admission['lock_sha256']
    for rel,digest in lock['source_sha256'].items(): assert sha(ROOT/rel)==digest
    manifests=[]; frames=[]; expected={f'seed_{s}.jsonl' for s in range(20571001,20571129)}
    for phase in ['science','replay']:
        directory=DATA/phase; manifest=read(directory/'hashes.json')
        assert set(manifest)==expected=={p.name for p in directory.glob('seed_*.jsonl')}
        for name,digest in manifest.items(): assert sha(directory/name)==digest
        manifests.append(manifest)
        if phase=='science':
            for name in sorted(manifest):
                part=pd.read_json(directory/name,lines=True,precise_float=True)
                assert len(part)==1536 and part.seed.nunique()==1 and part.seed.iloc[0]==int(name[5:-6])
                frames.append(part)
    assert manifests[0]==manifests[1]
    for name,digest in complete['analysis_sha256'].items():
        a,b=[DATA/('analysis_'+tag)/name for tag in ['a','b']]
        assert sha(a)==sha(b)==digest and a.read_bytes()==b.read_bytes()
    assert len(complete['analysis_sha256'])==7
    frame=pd.concat(frames,ignore_index=True)
    K=['seed','family','regime','learner','scale']; cellK=['family','regime','learner','scale','policy']
    assert len(frame)==196608 and not frame.duplicated(K+['policy']).any()
    assert set(frame.seed)==set(range(20571001,20571129)) and frame.family.nunique()==16
    assert set(frame.learner)==set(LEARNERS) and set(frame.policy)==set(POLICIES)
    assert set(frame.scale)=={32,128,512,2048} and set(frame.regime)=={'low','high'}
    assert (frame.groupby(K).size()==6).all()
    numeric=['terminal_estimate','terminal_squared_risk','scheduled','usable','messages_used','environment_used','message_budget','environment_budget']
    assert np.isfinite(frame[numeric].to_numpy()).all() and (frame.terminal_squared_risk>=0).all()
    assert np.allclose(frame.terminal_estimate**2,frame.terminal_squared_risk,rtol=1e-13,atol=1e-14)
    assert (frame.initialization==1).all() and (frame.physical_startup_charge==0).all()
    pm=frame.probe_n*(8+frame.probe_q); pe=frame.probe_n*frame.probe_b
    assert (frame.messages_used==pm+frame.scheduled*(8+frame.q)).all()
    assert (frame.environment_used==pe+frame.scheduled*frame.b).all()
    assert (frame.messages_used<=frame.message_budget).all() and (frame.environment_used<=frame.environment_budget).all()
    M=np.minimum((frame.message_budget-pm)//(8+frame.q),(frame.environment_budget-pe)//frame.b)
    assert (M==frame.scheduled).all() and (frame.usable==np.maximum(0,M-frame.D)).all()
    # Verify exact public tuple coverage and shared potential streams without generating observations.
    public=frame[['family','Q','theta_high','lambda_public','D','learner','scale','candidate_feasible','corrected_admit','zero_allowance_admit']].drop_duplicates()
    assert len(public)==128
    designs={}
    for r in public.itertuples(index=False):
        family=Family(int(r.Q),float(r.theta_high),float(r.lambda_public),int(r.D))
        assert family.key==r.family
        d=design(r.learner,family,int(r.scale)); designs[r.family,r.learner,r.scale]=d
        assert d['feasible']==r.candidate_feasible and d['corrected']==r.corrected_admit and d['zero_allowance']==r.zero_allowance_admit
    for (seed,family,regime,learner),part in frame.groupby(['seed','family','regime','learner']):
        stream=hashlib.sha256(f'V3-RISK-GATE-8H:{seed}:{family}:{regime}:{learner}'.encode()).hexdigest()
        assert (part.stream_seed==stream).all()
    # Check action/upper-bound role logic for all rows; oracle is the only truth-aware exception.
    for row in frame.itertuples(index=False):
        d=designs[row.family,row.learner,row.scale]
        assert row.message_budget==d['bm'] and row.environment_budget==d['be']
        if row.policy=='known_hypothesis_oracle_evaluation_only':
            assert not row.probe_used and row.probe_n==0
            continue  # Public oracle source is separately labeled, never passed to deployment selection.
        if row.policy=='fixed_q2_b1': wants=False; action=(2,1)
        else:
            wants=d['feasible'] and (row.policy=='information_only' or (row.policy=='risk_corrected_gate' and d['corrected']) or (row.policy=='no_carryover_gate' and d['no_carryover']))
            a=d[row.identified] if wants else d['fallback']; action=(a.q,a.b)
        assert bool(row.probe_used)==bool(wants) and (row.q,row.b)==action
        assert row.probe_n==(d['probe_n'] if wants else 0)
        if wants:
            assert row.probe_q==row.Q and row.probe_b==1
            assert bool(row.direction_error)==(row.identified!=row.regime)
            assert row.directional_bound==d['delta']
        if row.policy=='risk_corrected_gate':
            U=d['U0' if row.regime=='low' else 'U1'] if wants else row.fallback_population_risk
            assert math.isclose(row.certified_population_upper,U,rel_tol=1e-13,abs_tol=1e-14)
            assert U<=row.fallback_population_risk+(0.1 if row.regime=='low' else 0)+1e-12
    base=frame[frame.policy=='always_all'][K+['terminal_squared_risk']].rename(columns={'terminal_squared_risk':'fallback_observed_risk'})
    paired=frame.merge(base,on=K,validate='many_to_one'); paired['gain']=paired.fallback_observed_risk-paired.terminal_squared_risk
    a=DATA/'analysis_a'
    cells=pd.read_csv(a/'cell_results.csv'); macros=pd.read_csv(a/'macro_results.csv'); summary=read(a/'summary.json')
    assert len(cells)==1536 and len(macros)==96
    actual=paired.groupby(cellK).agg(mean_risk=('terminal_squared_risk','mean'),fallback_mean_risk=('fallback_observed_risk','mean'),paired_improvement=('gain','mean'),mean_messages=('messages_used','mean'),mean_environment=('environment_used','mean'),probe_rate=('probe_used','mean'))
    target=cells.set_index(cellK).sort_index(); actual=actual.sort_index()
    assert actual.index.equals(target.index)
    for col in actual: assert np.allclose(actual[col],target[col],rtol=1e-12,atol=1e-14)
    perseed=paired.groupby(['regime','learner','scale','policy','seed']).gain.mean()
    macro_actual=perseed.groupby(['regime','learner','scale','policy']).mean().sort_index()
    assert np.allclose(macro_actual,macros.set_index(['regime','learner','scale','policy']).sort_index().mean_improvement,rtol=1e-12,atol=1e-14)
    # Saved zero-slack reconstruction must exactly select a preserved branch, not a new run.
    stress=pd.read_csv(a/'zero_allowance_stress.csv'); assert len(stress)==32768 and not stress.duplicated(K).any()
    source=stress.merge(frame,on=K,left_index=False,right_index=False,suffixes=('','_raw'))
    source=source[source.source_policy==source.policy]
    assert len(source)==32768
    for col in ['terminal_squared_risk','messages_used','environment_used']: assert np.allclose(source[col],source[col+'_raw'],rtol=1e-13,atol=1e-14)
    corrected=cells[cells.policy=='risk_corrected_gate']
    assert int((corrected.paired_improvement<0).sum())==summary['corrected_adverse_cell_means']
    # Direction errors are deduplicated over shared policy/budget decisions, not counted24 times.
    probes=frame[(frame.policy=='information_only') & frame.probe_used].drop_duplicates(['seed','family','regime','learner'])
    errors=probes.groupby(['family','regime','learner']).direction_error.agg(['sum','count']).sort_index()
    directional=pd.read_csv(a/'directional_errors.csv').set_index(['family','regime','learner']).sort_index()
    assert (errors['sum'].to_numpy()==directional.errors.to_numpy()).all() and (errors['count'].to_numpy()==directional.seeds.to_numpy()).all()
    corrected_raw=paired[paired.policy=='risk_corrected_gate']; nocarry=frame[frame.policy=='no_carryover_gate']
    compare=corrected_raw.merge(nocarry,on=K,suffixes=('','_nc'),validate='one_to_one')
    gate_differences=int((compare.probe_used!=compare.probe_used_nc).sum())
    action_differences=int(((compare.q!=compare.q_nc)|(compare.b!=compare.b_nc)).sum())
    risk_differences=int((compare.terminal_squared_risk!=compare.terminal_squared_risk_nc).sum())
    empirical_calibration=cells[cells.policy=='risk_corrected_gate'].upper_minus_observed_mean
    result=dict(status='INDEPENDENT_READBACK_COMPLETE_SCIENTIFIC_JUDGMENT_REQUIRED',science=196608,replay=196608,source_hashes=len(lock['source_sha256']),all256_seed_parts_sha_and_full_replay_verified=True,all7_analysis_bytes_equal=True,all_rows_paid_ledgers_finite_initialization_FIFO_and_role_checks=True,all1536_cell_means_costs_and96_macro_means_recomputed=True,bootstrap_not_rerun='Reviewed frozen2000x128 paired-seed algorithm and exact two-copy outputs, no third analyzer execution',saved_zero_slack_rows=32768,public_designs=128,corrected_admitted_designs=int(public.corrected_admit.sum()),zero_slack_admitted_designs=int(public.zero_allowance_admit.sum()),no_carryover_admitted_designs=sum(bool(d['no_carryover']) for d in designs.values()),corrected_adverse_cells=int((corrected.paired_improvement<0).sum()),corrected_positive_interval_cells=int((corrected.ci_low>0).sum()),corrected_negative_interval_cells=int((corrected.ci_high<0).sum()),corrected_cell_total=len(corrected),corrected_vs_no_carryover_probe_row_differences=gate_differences,corrected_vs_no_carryover_action_row_differences=action_differences,corrected_vs_no_carryover_risk_row_differences=risk_differences,unique_probes=len(probes),probe_errors=int(probes.direction_error.sum()),empirical_cell_means_above_model_upper=int((empirical_calibration<0).sum()),empirical_calibration_is_not_a_finite_sample_guarantee=True,native_charged_cpu_seconds=terminal['cumulative_charged_cpu_seconds'],audit_observed_process_cpu_seconds=time.process_time(),no_new_trajectory_or_bootstrap_analysis=True)
    write('checks.json',result)
    write('corrected_macro_results.json',json.loads(macros[macros.policy=='risk_corrected_gate'].to_json(orient='records',double_precision=15)))
    write('all_policy_macro_results.json',json.loads(macros.to_json(orient='records',double_precision=15)))
    write('probe_results.json',json.loads(directional.reset_index().to_json(orient='records',double_precision=15)))
    print(json.dumps(result,allow_nan=False)); print('FINAL_OBSERVED_AUDIT_CPU_SECONDS',time.process_time())

if __name__=='__main__': main()
