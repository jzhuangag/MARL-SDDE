"""Human-approved resource-only amendment; frozen scientific source stays intact.

The existing39-cell profile is NEVER rerun. One owner runs the original complete
117-cell science,6 fixed replays, double analysis and all-method plots unchanged.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from sealed_bridge_learning.guard import Budget, SingleOwner, ROOT, dump, fingerprint, sha
from sealed_bridge_learning.workflow import science

ORIGINAL_LOCK = ROOT/'ICML-2027/internal/sealed_bridge_learning_lock_20261016.json'
PROFILE = ROOT/'ICML-2027/results/sealed_bridge_learning_cost_20261016'
AMENDMENT = ROOT/'ICML-2027/internal/sealed_bridge_learning_40h_preregistration_20261016.md'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def amended_plan():
    plan = load(ORIGINAL_LOCK)['plan']
    plan['cpu_seconds_limit'] = 144000
    return plan


def amendment_files():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in (Path(__file__).resolve(), AMENDMENT)}


def check_profile():
    decision, final = load(PROFILE/'decision.json'), load(PROFILE/'workflow_complete.json')
    assert decision['decision'] == final['decision'] == 'RESOURCE_NO_GO'
    assert decision['science_started'] is False and decision['profile_cells'] == 39
    manifest = load(PROFILE/'hashes.json')
    assert len(manifest) == 39
    for name, digest in manifest.items():
        assert sha(PROFILE/name) == digest
    a = ROOT/'ICML-2027/results/sealed_bridge_learning_cost_20261016_validation_a'
    b = ROOT/'ICML-2027/results/sealed_bridge_learning_cost_20261016_validation_b'
    for name in ('integrity.json', 'decision.json'):
        assert (a/name).read_bytes() == (b/name).read_bytes()
    assert load(a/'decision.json')['evidence_checks_pass']
    assert decision['total_projected_cpu_seconds'] < 144000
    return decision, final


class ResourceBudget(Budget):
    def __init__(self, lock_path, root):
        lock = load(lock_path)
        self.amendment_hashes = lock['amendment_files']
        assert self.amendment_hashes == amendment_files()
        assert lock['original_lock_sha256'] == sha(ORIGINAL_LOCK)
        old = load(ORIGINAL_LOCK)
        assert fingerprint() == old['fingerprint'] == lock['fingerprint']
        actual = amended_plan()
        assert lock['plan'] == actual
        changes = [k for k in actual if actual[k] != old['plan'][k]]
        assert changes == ['cpu_seconds_limit']
        decision, final = check_profile()
        assert lock['profile_decision_sha256'] == sha(PROFILE/'decision.json')
        assert lock['amended_resource_forecast_pass'] is True
        super().__init__(actual, lock_path, root, 'science', final['cpu_seconds'])
        self.verify_sources()
        for rel, digest in self.amendment_hashes.items():
            data = subprocess.check_output(['git', 'show', f'{self.source_commit}:{rel}'], cwd=ROOT)
            assert hashlib.sha256(data).hexdigest() == digest

    def verify_sources(self):
        super().verify_sources()
        assert self.amendment_hashes == amendment_files()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    if args.freeze:
        decision, _ = check_profile()
        assert fingerprint() == load(ORIGINAL_LOCK)['fingerprint']
        dump(args.freeze, dict(plan=amended_plan(), fingerprint=fingerprint(),
            amendment_files=amendment_files(), original_lock_sha256=sha(ORIGINAL_LOCK),
            profile_decision_sha256=sha(PROFILE/'decision.json'),
            original_resource_decision='RESOURCE_NO_GO', amended_resource_forecast_pass=True,
            human_authorization='2026-10-16: approve40 CPU hours; complete strong-baseline comparison; retain V3'))
        return
    if not all((args.lock, args.admission, args.root)):
        parser.error('Execution requires committed lock/admission/new root')
    admission = load(args.admission)
    assert admission['science_admitted'] is True and admission['lock_sha256'] == sha(args.lock)
    assert admission['profile_decision_sha256'] == sha(PROFILE/'decision.json')
    content = subprocess.check_output(['git', 'show', f'HEAD:{args.admission.resolve().relative_to(ROOT).as_posix()}'], cwd=ROOT)
    assert hashlib.sha256(content).hexdigest() == sha(args.admission)
    with SingleOwner():
        budget = ResourceBudget(args.lock.resolve(), args.root.resolve())
        root = budget.root; root.mkdir(exist_ok=False)
        dump(root/'execution.json', dict(state='running', phase='science', pid=os.getpid(),
            source_commit=budget.source_commit, original_scientific_commit='27cd2874a86b285aeb3e7d08b013d83b29c937a3',
            lock_sha256=sha(args.lock), admission_sha256=sha(args.admission),
            earlier_profile_cpu_seconds=budget.earlier_cpu, cpu_limit=144000, command=sys.argv,
            fingerprint=budget.fingerprint, amendment_files=budget.amendment_hashes))
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        try:
            result = science(budget.plan, budget)
            budget.verify_sources()
            assert fingerprint() == budget.fingerprint
            budget()
            dump(root/'decision.json', dict(**result, total_cpu_seconds=budget.cpu,
                cost_profile_rerun=False, all_scientific_settings_unchanged=True,
                positive_results_promised=False, manuscript_admission=False))
            summary = load(Path(str(root)+'_analysis_a')/'summary.json')
            with (root/'decision.md').open('x', encoding='utf-8', newline='\n') as stream:
                stream.write('# Complete asynchronous MARL development decision\n\n')
                stream.write('## Material Passport\n\nARS complete registered development;3 seeds/task, not confirmation.\n\n')
                stream.write(f'Decision: {result["decision"]}. Total CPU including prior cost profile: {budget.cpu:.6f}s.\n\n')
                stream.write('All117 original cells,6 fixed semantic replays and byte-identical analyses completed.\n')
                stream.write('Full metrics/controls/ablations/resource costs are in sibling_analysis_a/metrics.csv; all-method native/message/byte plots are in sibling_plots/.\n\n')
                for task in summary['tasks']:
                    stream.write(json.dumps(task, sort_keys=True, ensure_ascii=False)+'\n\n')
                stream.write('No confirmation, ICML-readiness or V3 manuscript admission follows automatically. No task/method/seed/gate was removed or changed.\n')
            dump(root/'workflow_complete.json', dict(state='complete', decision=result['decision'],
                cpu_seconds=budget.cpu, driver_pid=os.getpid()))
            print(json.dumps(dict(state='complete', decision=result['decision'], cpu_seconds=budget.cpu)))
        except BaseException as error:
            dump(root/'workflow_failed.json', dict(state='stopped', exception=repr(error),
                traceback=traceback.format_exc(), cpu_seconds=budget.cpu, retry_authorized=False))
            raise


if __name__ == '__main__':
    main()
