"""Human-authorized resource-only continuation; never mutate the frozen study.

Only a CPU-cap stop may be continued. Complete cells are imported byte-for-byte;
the unfinished cell has no checkpoint and is regenerated, with its lost CPU paid.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from sealed_bridge_learning.guard import Budget, SingleOwner, ROOT, dump, fingerprint, sha
from sealed_bridge_learning.analyze import mapping, audit, analyze, plots
from sealed_bridge_learning.benchmark import run_cell

OLD = ROOT/'ICML-2027/results/sealed_bridge_learning_dev40h_20261016'
OLD_LOCK = ROOT/'ICML-2027/internal/sealed_bridge_learning_40h_lock_20261016.json'
MEMO = ROOT/'ICML-2027/internal/sealed_bridge_learning_140h_amendment_20261018.md'
CAP = 504000


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def extension_files():
    paths = [Path(__file__).resolve(), MEMO]
    paths += [ROOT/rel for rel in load(OLD_LOCK)['amendment_files']]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def stopped_prefix():
    failure = load(OLD/'workflow_failed.json')
    assert failure['exception'] == "TimeoutError('Whole registered profile+science+replay+analysis CPU cap')"
    assert not (OLD/'workflow_complete.json').exists()
    plan = load(OLD_LOCK)['plan']
    expected = mapping(plan)
    progress = sorted(OLD.glob('progress_*.json'))
    last = load(progress[-1])
    assert last['phase'] == 'science'
    count = last['completed']
    manifest_path = OLD/f'hashes_{count:04d}.json'
    manifest = load(manifest_path)
    assert list(sorted(manifest)) == sorted(list(expected)[:count])
    assert {p.name for p in OLD.glob('cell_*.json')} == set(manifest)
    for name, digest in manifest.items():
        assert sha(OLD/name) == digest
    partial = load(OLD/'failed_cell_partial_costs.json')
    assert tuple(partial['cell']) == expected[list(expected)[count]]
    assert not list(Path(str(OLD)+'_replay').glob('cell_*.json'))
    return dict(completed=count, manifest=manifest,
                failure_sha256=sha(OLD/'workflow_failed.json'),
                partial_costs_sha256=sha(OLD/'failed_cell_partial_costs.json'),
                execution_sha256=sha(OLD/'execution.json'),
                prefix_manifest_sha256=sha(manifest_path),
                prior_cpu_seconds=failure['cpu_seconds'],
                unfinished_cell=partial['cell'],
                unfinished_cpu_seconds=partial['cpu_seconds'])


class ContinuationBudget(Budget):
    def __init__(self, lock_path, root):
        lock = load(lock_path)
        self.extension_hashes = lock['extension_files']
        assert self.extension_hashes == extension_files()
        assert lock['old_lock_sha256'] == sha(OLD_LOCK)
        assert stopped_prefix() == lock['preserved_prefix']
        old = load(OLD_LOCK)
        plan = json.loads(json.dumps(old['plan']))
        plan['cpu_seconds_limit'] = CAP
        assert plan == lock['plan']
        assert fingerprint() == old['fingerprint'] == lock['fingerprint']
        self.old_manifest = lock['preserved_prefix']['manifest']
        super().__init__(plan, lock_path, root, 'science', lock['preserved_prefix']['prior_cpu_seconds'])
        for rel, digest in self.extension_hashes.items():
            data = subprocess.check_output(['git', 'show', f'{self.source_commit}:{rel}'], cwd=ROOT)
            assert hashlib.sha256(data).hexdigest() == digest

    def verify_sources(self):
        super().verify_sources()
        assert self.extension_hashes == extension_files()


def complete_remaining(budget):
    root = budget.root
    replay = Path(str(root)+'_replay')
    replay.mkdir(exist_ok=False)
    imported = {}
    for name, digest in budget.old_manifest.items():
        assert not (root/name).exists()
        shutil.copyfile(OLD/name, root/name)
        assert sha(root/name) == sha(OLD/name) == digest
        # Integrity/finiteness only; never rank incomplete scientific outcomes.
        audit(load(root/name), budget.plan)
        imported[name] = digest
    dump(root/'imported_payload_hashes.json', imported)
    for repeat, directory in ((False, root), (True, replay)):
        hashes = {}
        for name, cell in mapping(budget.plan, repeat).items():
            if not repeat and name in imported:
                hashes[name] = imported[name]
                continue
            budget.active_cell = cell
            budget()
            row = run_cell(*cell, budget.plan, budget)
            dump(directory/name, row)
            hashes[name] = sha(directory/name)
            dump(directory/f'hashes_{len(hashes):04d}.json', hashes)
            audit(row, budget.plan)
            budget.verify_sources()
            budget.progress('replay' if repeat else 'science', len(hashes))
        dump(directory/'hashes.json', hashes)
    a, b = Path(str(root)+'_analysis_a'), Path(str(root)+'_analysis_b')
    budget()
    rows, go = analyze(root, replay, budget.plan, a)
    budget()
    analyze(root, replay, budget.plan, b)
    for name in ('metrics.csv', 'summary.json', 'integrity.json'):
        assert (a/name).read_bytes() == (b/name).read_bytes()
    budget()
    plots(rows, budget.plan, Path(str(root)+'_plots'))
    budget()
    return dict(decision='DEVELOPMENT_GO' if go else 'DEVELOPMENT_NO_GO',
                analysis_sha256={n: sha(a/n) for n in ('metrics.csv', 'summary.json', 'integrity.json')},
                byte_identical_analysis=True, imported_cells=len(imported),
                new_science_cells=117-len(imported), replay_cells=6,
                all_scientific_settings_unchanged=True, manuscript_admission=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    with SingleOwner():
        if args.freeze:
            prefix = stopped_prefix()
            old = load(OLD_LOCK)
            assert fingerprint() == old['fingerprint']
            plan = json.loads(json.dumps(old['plan']))
            plan['cpu_seconds_limit'] = CAP
            dump(args.freeze, dict(plan=plan, fingerprint=old['fingerprint'],
                 extension_files=extension_files(), old_lock_sha256=sha(OLD_LOCK),
                 preserved_prefix=prefix, resource_stop_only=True,
                 human_authorization='2026-10-18: additional100 CPU hours; preserve completed cells; finish all controls/replay/audit'))
            return
        assert args.lock and args.admission and args.root
        admission = load(args.admission)
        assert admission['lock_sha256'] == sha(args.lock) and admission['resource_continuation_authorized']
        committed = subprocess.check_output(['git', 'show', f'HEAD:{args.admission.resolve().relative_to(ROOT).as_posix()}'], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == sha(args.admission)
        budget = ContinuationBudget(args.lock.resolve(), args.root.resolve())
        root = budget.root
        root.mkdir(exist_ok=False)
        dump(root/'execution.json', dict(state='running', phase='continuation', pid=os.getpid(),
             source_commit=budget.source_commit, command=sys.argv, cpu_limit=CAP,
             earlier_cpu_seconds=budget.earlier_cpu, imported_cells=len(budget.old_manifest),
             lock_sha256=sha(args.lock), admission_sha256=sha(args.admission)))
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        try:
            result = complete_remaining(budget)
            budget.verify_sources()
            assert fingerprint() == budget.fingerprint
            budget()
            dump(root/'decision.json', dict(**result, total_cpu_seconds=budget.cpu,
                 prior_resource_stop_preserved=True, positive_results_promised=False))
            dump(root/'workflow_complete.json', dict(state='complete', decision=result['decision'],
                 cpu_seconds=budget.cpu, driver_pid=os.getpid()))
        except BaseException as error:
            dump(root/'workflow_failed.json', dict(state='stopped', exception=repr(error),
                 traceback=traceback.format_exc(), cpu_seconds=budget.cpu, retry_authorized=False))
            raise


if __name__ == '__main__':
    main()
