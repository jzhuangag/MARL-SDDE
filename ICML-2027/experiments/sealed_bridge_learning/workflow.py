"""One owner: locked outcome-free cost profile, or separately admitted science.

No resume/retry mode. Existing roots are never overwritten. All native calls
belong to this owner; profile feedback/returns are never ranked or tuned.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sealed_bridge_learning.guard import Budget, SingleOwner, ROOT, PLAN, dump, fingerprint, profile_plan, sha
from sealed_bridge_learning.benchmark import run_cell, METHODS
from sealed_bridge_learning.analyze import audit, analyze, mapping, plots


def write_manifest(root, hashes):
    # Append-only generations; final hashes.json is created exactly once.
    dump(root/f'hashes_{len(hashes):04d}.json', hashes)


def profile(plan, budget):
    root = budget.root
    actual = profile_plan(plan)
    costs, hashes = [], {}
    for ti, task in enumerate(plan['tasks']):
        for method in plan['methods']:
            budget.active_cell = ti, plan['profile']['seed'], method
            budget()
            row = run_cell(*budget.active_cell, actual, budget)
            audit(row, actual, require_staleness=False)
            # Outcome-free persisted material: costs/shape/protocol only. Native
            # rewards/gradients are used by the fixed algorithm, but never inspected,
            # ranked, displayed or retained as a performance experiment here.
            stripped = dict(task_index=ti, task=task['name'], method=method,
                seed=plan['profile']['seed'], training_transitions=row['training_transitions'],
                cpu_seconds=row['cpu_seconds'], costs=row['costs'],
                charge_event_totals=row['charge_event_totals'], native_budget_events=row['native_budget_events'],
                finite_payload=True, return_ranking_performed=False, performance_evidence=False,
                evaluations_checked_for_finiteness_only=True,
                full_payload_digest=hashlib.sha256(json.dumps({k: v for k, v in row.items() if k != 'cpu_seconds'},
                                                            sort_keys=True, allow_nan=False).encode()).hexdigest())
            name = f'profile_{ti}_{method}.json'
            dump(root/name, stripped); hashes[name] = sha(root/name)
            costs.append(stripped)
            budget.progress('cost_profile', len(costs))
            budget.verify_sources()
    dump(root/'hashes.json', hashes)
    # Every method/task is profiled. Conservative linear scaling retains init and
    # all profile evaluation overhead per extrapolated block (does not subtract it).
    # It is a planning forecast, not a statistical confidence bound or impossibility
    # theorem. Multiplier and analysis reserve were frozen before native data.
    forecasts = []
    for row in costs:
        target = plan['tasks'][row['task_index']]['training_transitions']
        repetitions = len(plan['seeds'])+int(row['method'] in plan['replay']['methods'])
        estimate = row['cpu_seconds']*target/row['training_transitions']*repetitions
        forecasts.append(dict(task=row['task'], method=row['method'], repetitions=repetitions,
                              projected_cpu_seconds=estimate))
    projected = (sum(x['projected_cpu_seconds'] for x in forecasts)*plan['profile']['forecast_multiplier']+
                 budget.cpu+plan['profile']['analysis_reserve_seconds'])
    go = projected <= plan['cpu_seconds_limit']
    decision = dict(decision='RESOURCE_FEASIBLE' if go else 'RESOURCE_NO_GO',
        profile_cells=len(costs), profile_cpu_seconds=budget.cpu, forecasts=forecasts,
        total_projected_cpu_seconds=projected, cpu_cap=plan['cpu_seconds_limit'],
        forecast_is_proof=False, performance_evidence=False, manuscript_admission=False,
        science_started=False, need_separate_science_admission_commit=go,
        native_science_transitions=sum(t['training_transitions'] for t in plan['tasks'])*len(plan['seeds'])*len(METHODS),
        native_replay_transitions=sum(t['training_transitions'] for t in plan['tasks'])*len(plan['replay']['methods']))
    dump(root/'decision.json', decision)
    return decision


def science(plan, budget):
    root = budget.root
    replay = Path(str(root)+'_replay'); replay.mkdir(exist_ok=False)
    for repeat, directory in [(False, root), (True, replay)]:
        hashes = {}
        for name, cell in mapping(plan, repeat).items():
            budget.active_cell = cell
            budget()
            row = run_cell(*cell, plan, budget)
            # Preserve a completed payload BEFORE judging it, including failures.
            dump(directory/name, row)
            hashes[name] = sha(directory/name)
            write_manifest(directory, hashes)
            audit(row, plan)
            budget.verify_sources()
            budget.progress('replay' if repeat else 'science', len(hashes))
        dump(directory/'hashes.json', hashes)
    a, b = Path(str(root)+'_analysis_a'), Path(str(root)+'_analysis_b')
    budget()
    rows, go = analyze(root, replay, plan, a)
    budget()
    analyze(root, replay, plan, b)
    for name in ('metrics.csv', 'summary.json', 'integrity.json'):
        if (a/name).read_bytes() != (b/name).read_bytes():
            raise AssertionError(f'Analyzer replay mismatch {name}')
    budget()
    plots(rows, plan, Path(str(root)+'_plots'))
    budget()
    return dict(decision='DEVELOPMENT_GO' if go else 'DEVELOPMENT_NO_GO',
                analysis_sha256={n: sha(a/n) for n in ('metrics.csv', 'summary.json', 'integrity.json')},
                byte_identical_analysis=True, manuscript_admission=False, independent_confirmation_required=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--phase', choices=('profile', 'science'))
    parser.add_argument('--profile-root', type=Path)
    parser.add_argument('--admission', type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if tuple(plan['methods']) != METHODS or len(set(plan['seeds'])) != len(plan['seeds']):
        raise AssertionError('Incomplete catalogue or seed mapping')
    if args.freeze:
        dump(args.freeze, dict(plan=plan, fingerprint=fingerprint()))
        return
    if not all((args.lock, args.root, args.phase)):
        parser.error('Execution requires lock/root/phase')
    earlier = 0.
    if args.phase == 'science':
        if not args.profile_root or not args.admission:
            raise RuntimeError('Science requires separate committed admission after resource decision')
        decision = json.loads((args.profile_root/'decision.json').read_text())
        admission = json.loads(args.admission.read_text())
        assert decision['decision'] == 'RESOURCE_FEASIBLE'
        assert admission['profile_decision_sha256'] == sha(args.profile_root/'decision.json')
        assert admission['lock_sha256'] == sha(args.lock) and admission['science_admitted'] is True
        content = subprocess.check_output(['git', 'show', f'HEAD:{args.admission.resolve().relative_to(ROOT).as_posix()}'], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == sha(args.admission)
        earlier = json.loads((args.profile_root/'workflow_complete.json').read_text())['cpu_seconds']
    with SingleOwner():
        budget = Budget(plan, args.lock.resolve(), args.root.resolve(), args.phase, earlier)
        budget.root.mkdir(exist_ok=False)
        dump(budget.root/'execution.json', dict(experiment=plan['experiment'], state='running', phase=args.phase,
            pid=os.getpid(), source_commit=budget.source_commit, command=sys.argv,
            lock_sha256=sha(args.lock), fingerprint=budget.fingerprint, earlier_cpu_seconds=earlier))
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        try:
            decision = profile(plan, budget) if args.phase == 'profile' else science(plan, budget)
            budget.verify_sources()
            if fingerprint() != budget.fingerprint:
                raise AssertionError('Post-run runtime/source mismatch')
            budget()
            dump(budget.root/'workflow_complete.json', dict(state='complete', decision=decision['decision'],
                                                           cpu_seconds=budget.cpu, driver_pid=os.getpid()))
            print(json.dumps(dict(state='complete', decision=decision['decision'], cpu_seconds=budget.cpu)))
        except BaseException as error:
            dump(budget.root/'workflow_failed.json', dict(state='stopped', exception=repr(error),
                traceback=traceback.format_exc(), cpu_seconds=budget.cpu, retry_authorized=False))
            raise


if __name__ == '__main__':
    main()
