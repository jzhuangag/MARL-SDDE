"""Read-only cost-evidence reconciliation; no native calls or science rerun.

Writes a new exclusive audit directory. Lives OUTSIDE the frozen learning
package; it neither mutates that package nor reruns its profile/analyzers.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'ICML-2027/experiments'))
from sealed_bridge_learning.guard import dump, fingerprint, sha
from sealed_bridge_learning.control import NativeBudget
from sealed_bridge_learning.native import Charges
from sealed_bridge_learning.analyze import finite


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(output):
    root = ROOT/'ICML-2027/results/sealed_bridge_learning_cost_20261016'
    lock_path = ROOT/'ICML-2027/internal/sealed_bridge_learning_lock_20261016.json'
    lock = load(lock_path); plan = lock['plan']
    execution, final, decision = [load(root/name) for name in
                                 ('execution.json', 'workflow_complete.json', 'decision.json')]
    assert execution['phase'] == 'profile' and execution['earlier_cpu_seconds'] == 0
    assert execution['fingerprint'] == lock['fingerprint'] == fingerprint()
    assert execution['lock_sha256'] == sha(lock_path)
    assert final['state'] == 'complete' and final['decision'] == decision['decision'] == 'RESOURCE_NO_GO'
    assert final['driver_pid'] == execution['pid']
    assert final['cpu_seconds'] <= plan['profile']['cpu_seconds_limit'] <= plan['cpu_seconds_limit']
    assert not (root/'workflow_failed.json').exists()
    for rel, digest in lock['fingerprint']['files'].items():
        content = subprocess.check_output(['git', 'show', f'{execution["source_commit"]}:{rel}'], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == digest
    old_lock = load(ROOT/'ICML-2027/internal/sealed_bridge_qualification_lock_20261016.json')
    for rel, digest in old_lock['fingerprint']['files'].items():
        assert sha(ROOT/rel) == digest

    manifest = load(root/'hashes.json')
    expected = {f'profile_{ti}_{method}.json': (ti, method) for ti in range(len(plan['tasks']))
                for method in plan['methods']}
    assert set(manifest) == set(expected) == {p.name for p in root.glob('profile_*.json')}
    assert len(expected) == decision['profile_cells'] == 39
    allowed = {'task_index', 'task', 'method', 'seed', 'training_transitions', 'cpu_seconds', 'costs',
               'charge_event_totals', 'native_budget_events', 'finite_payload',
               'return_ranking_performed', 'performance_evidence',
               'evaluations_checked_for_finiteness_only', 'full_payload_digest'}
    cells, forecasts = [], []
    for name, (ti, method) in expected.items():
        assert sha(root/name) == manifest[name]
        row = load(root/name)
        assert set(row) == allowed and finite(row)
        task = plan['tasks'][ti]; h = task['horizon']
        assert (row['task_index'], row['task'], row['method'], row['seed']) == (
            ti, task['name'], method, plan['profile']['seed'])
        assert row['finite_payload'] and row['evaluations_checked_for_finiteness_only']
        assert not row['return_ranking_performed'] and not row['performance_evidence']
        assert len(row['full_payload_digest']) == 64
        limit = plan['profile']['episodes']*h
        assert row['training_transitions'] == row['costs']['native_training'] == limit
        costs = row['costs']; aggregate = dict.fromkeys(Charges.FIELDS, 0)
        for event in row['charge_event_totals'].values():
            assert event['calls'] >= 1
            assert all(isinstance(v, int) and v >= 0 for v in event.values())
            for k, v in event.items():
                if k != 'calls':
                    aggregate[k] += v
        assert all(aggregate[k] == costs[k] for k in aggregate)
        assert costs['logical_bytes'] == costs['program_upload_bytes']+costs['policy_broadcast_bytes']
        assert costs['work_rows'] == sum(costs[k] for k in ('sampling_actor_rows', 'actor_forward_rows',
            'actor_backward_rows', 'critic_forward_rows', 'critic_backward_rows'))
        eval_count = (plan['profile']['evaluation_intervals']+1)*plan['profile']['evaluation_episodes']*h
        assert costs['native_benchmark'] == costs['native_frozen_policy'] == eval_count
        agents = 3 if task['name'] == 'spread' else 2
        assert costs['sampling_actor_rows'] == agents*(limit+2*eval_count)
        for role in ('benchmark', 'frozen_policy'):
            event = row['charge_event_totals'][role+':return_upload']
            assert event['program_upload_bytes'] == 8*eval_count//h
        fit = row['charge_event_totals']['critic:fit']
        assert fit['critic_adam_calls'] == fit['calls']*plan['critic_epochs']

        ledger = NativeBudget(limit, h)
        for event in row['native_budget_events']:
            kind = event['kind']
            if kind == 'reserve_launch':
                ledger.launch(event['source_id'], event['source']//h,
                              event['minimum_receipt']//h, event['receipt']//h)
            elif kind == 'source_charge':
                ledger.start_source(event['source_id'])
            elif kind == 'source_complete':
                ledger.finish_source(event['source_id'])
            elif kind == 'receipt_charge':
                ledger.start_receipt(event['source_id'], event['transitions']//h)
            elif kind == 'evaluation_charge':
                ledger.charge_evaluation(event['transitions'], event['role'])
            else:
                ledger.charge_unreserved(event['transitions']//h, kind)
            assert ledger.free >= 0
        ledger.assert_exhausted()
        assert ledger.events == row['native_budget_events']
        assert ledger.evaluation_paid == costs['native_benchmark']+costs['native_frozen_policy']
        repetitions = len(plan['seeds'])+int(method in plan['replay']['methods'])
        forecasts.append(dict(task=task['name'], method=method, repetitions=repetitions,
                              projected_cpu_seconds=row['cpu_seconds']*task['training_transitions']/limit*repetitions))
        cells.append(dict(name=name, sha256=manifest[name], task=task['name'], method=method,
                          cpu_seconds=row['cpu_seconds'], costs=costs, exact_costs=True,
                          exact_native_ledger=True, outcome_free_persisted_schema=True))
    assert forecasts == decision['forecasts']
    raw = sum(x['projected_cpu_seconds'] for x in forecasts)
    projected = (raw*plan['profile']['forecast_multiplier']+decision['profile_cpu_seconds']+
                 plan['profile']['analysis_reserve_seconds'])
    assert projected == decision['total_projected_cpu_seconds'] > plan['cpu_seconds_limit']
    assert decision['forecast_is_proof'] is False and decision['science_started'] is False
    assert not decision['performance_evidence'] and not decision['manuscript_admission']
    progress = sorted(root.glob('progress_*.json'))
    assert len(progress) == 39
    prior_cpu = -1
    for i, path in enumerate(progress, 1):
        record = load(path)
        assert record['completed'] == i and record['pid'] == execution['pid']
        assert record['phase'] == 'cost_profile' and record['cpu_seconds'] >= prior_cpu
        prior_cpu = record['cpu_seconds']
    assert final['cpu_seconds'] >= prior_cpu

    totals = {k: sum(c['costs'][k] for c in cells) for k in cells[0]['costs']}
    by_task = {}
    for task in plan['tasks']:
        name = task['name']
        by_task[name] = dict(profile_cpu_seconds=sum(c['cpu_seconds'] for c in cells if c['task'] == name),
            projected_science_replay_cpu_seconds=sum(f['projected_cpu_seconds'] for f in forecasts if f['task'] == name))
    audit_report = dict(decision='RESOURCE_NO_GO', evidence_checks_pass=True,
        source_commit=execution['source_commit'], lock_sha256=sha(lock_path),
        runtime_source_lock_pass=True, unchanged_old_frozen_sources=len(old_lock['fingerprint']['files']),
        profile_cells=39, cells=cells, totals=totals, by_task=by_task,
        raw_scaled_cpu_seconds=raw, projected_total_cpu_seconds=projected,
        projected_total_cpu_hours=projected/3600, registered_cap_cpu_hours=plan['cpu_seconds_limit']/3600,
        projected_to_cap_ratio=projected/plan['cpu_seconds_limit'],
        measured_profile_cpu_seconds=final['cpu_seconds'],
        profile_native_replay='NOT_REGISTERED_NOT_RUN', scientific_cells=0, scientific_replay_cells=0,
        return_superiority='NOT_TESTED', paper_admission=False,
        raw_return_and_gradient_values_inspected=False,
        validation_scope='independent cost/ledger/hash/forecast reconciliation; not a native replay or performance validation',
        driver_runtime_full_payload_audit='attested by locked driver before outcome-free stripping; full payload not retained',
        forecast_is_statistical_bound=False, required_next_authority='resource-envelope amendment before science',
        record_hashes={p.name: sha(p) for p in sorted(root.glob('*.json'))})
    output.mkdir(exist_ok=False)
    dump(output/'integrity.json', audit_report)
    dump(output/'decision.json', dict(decision='RESOURCE_NO_GO', evidence_checks_pass=True,
        actual_profile_cpu_seconds=final['cpu_seconds'], projected_cpu_seconds=projected,
        registered_cpu_limit=plan['cpu_seconds_limit'], science_started=False,
        scientific_failure=False, performance_evidence=False, unchanged_scientific_gates=True,
        retry_authorized=False, new_resource_authorization_required=True))
    print(json.dumps({k: v for k, v in audit_report.items() if k in (
        'decision', 'profile_cells', 'totals', 'by_task', 'raw_scaled_cpu_seconds',
        'projected_total_cpu_hours', 'registered_cap_cpu_hours', 'projected_to_cap_ratio',
        'measured_profile_cpu_seconds')}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True, type=Path)
    verify(parser.parse_args().output)
