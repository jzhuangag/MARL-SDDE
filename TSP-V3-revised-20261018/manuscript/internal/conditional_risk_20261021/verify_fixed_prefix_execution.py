"""Additional closed-data audit of executed fixed prefixes, not another analysis."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'experiments/v3_observable_gate1h'))
from support import RESULT,plan,dump,sha,pid_alive
from model import old

def run():
    execution=json.loads((RESULT/'execution.json').read_text());launch=json.loads((RESULT/'worker_launch.json').read_text())
    if pid_alive(execution['supervisor_pid']) or pid_alive(launch['pid']):raise RuntimeError('Closed data only')
    if not (RESULT/'independent_scientific_audit.json').exists():raise RuntimeError('Frozen independent read-back required first')
    output=RESULT/'independent_fixed_prefix_execution.json'
    if output.exists():raise RuntimeError('Existing audit preserved')
    cfg=plan();fixed=cfg['policies'][7];gate=cfg['policies'][0];uncorrected=cfg['policies'][3]
    groups={};fixed_rows={};gate_rows={};uncorrected_rows={};fallback_checks=0
    with (RESULT/'analysis_a/metrics.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['policy'] not in (fixed,gate,uncorrected):continue
            key=(row['learner'],row['family'],int(row['scale']))
            value=(int(row['q']),int(row['b']),int(row['M']),float(row['normalizer']))
            paired=tuple(row[k] for k in ('seed','learner','family','scale','regime','error_sign'))
            if row['policy'] in (gate,uncorrected):
                saved=gate_rows if row['policy']==gate else uncorrected_rows
                assert paired not in saved
                saved[paired]=tuple(row[k] for k in ('q','b','M','probe_used','probe_n','terminal_error','terminal_squared_risk','messages','clock'))
            if row['policy']==fixed:
                assert row['probe_used']=='False' and row['probe_n']=='0'
                if key in groups:assert value==groups[key]
                else:groups[key]=value
                paired=tuple(row[k] for k in ('seed','learner','family','scale','regime','error_sign'))
                assert paired not in fixed_rows
                fixed_rows[paired]=tuple(row[k] for k in ('terminal_error','terminal_squared_risk','messages','clock'))
    assert len(groups)==128
    assert len(gate_rows)==len(uncorrected_rows)==32768 and gate_rows==uncorrected_rows
    for (learner,family,scale),(q,b,M,normalizer) in groups.items():
        qkey,thkey,lkey,dkey=family.split('-')
        f=old.Family(int(qkey[1:]),float(thkey[2:]),float(lkey[1:]),int(dkey[1:]))
        if q==0:assert b==M==0;actual_risk=1.
        else:
            actual_risk=old.program_risk(learner,old.Action(q,b),M*(f.h+q),M*b,f,(f.theta_low+f.theta_high)/2)[0]
            assert M*(f.h+q)<=(f.h+f.Q)*scale and M*b<=4*scale
        assert abs(actual_risk-normalizer)<1e-11
    with (RESULT/'analysis_a/metrics.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['policy']!=gate:continue
            key=(row['learner'],row['family'],int(row['scale']))
            if row['candidate_admit']=='False':
                assert row['probe_used']=='False' and row['probe_n']=='0'
                assert (int(row['q']),int(row['b']),int(row['M']),float(row['normalizer']))==groups[key]
                paired=tuple(row[k] for k in ('seed','learner','family','scale','regime','error_sign'))
                assert tuple(row[k] for k in ('terminal_error','terminal_squared_risk','messages','clock'))==fixed_rows[paired]
                fallback_checks+=1
            else:assert row['candidate_admit']=='True' and row['probe_used']=='True'
    dump(output,dict(status='PASS',all_128_executed_fixed_prefixes_match_population_normalizers=True,
        fixed_actions_and_lengths_constant_across_seeds_regimes_and_signs=True,
        nonadmitted_gate_rows_exactly_equal_fixed_prefix=fallback_checks,
        nonadmitted_gate_errors_and_costs_byte_equal_paired_fixed_rows=True,
        all_32768_corrected_and_no_carryover_rows_exactly_equal=True,
        admitted_gate_rows_all_pay_probe=True,scientific_trajectories_generated=0,
        no_bootstrap_or_primary_endpoint_recomputed=True,source_sha256=sha(Path(__file__)),
        cpu_snapshot_seconds=time.process_time()))
    print(json.dumps(dict(status='PASS',fixed_designs=128,fallback_rows=fallback_checks,cpu_snapshot=time.process_time())))

if __name__=='__main__':run()
