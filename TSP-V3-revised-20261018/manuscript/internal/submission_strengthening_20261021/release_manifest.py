"""Read-only final artifact verification and exclusive release manifest creation."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
folder = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
qa = json.loads((folder / 'final_qa.json').read_text())
source = json.loads((folder / 'final_source_gate_after_visual.json').read_text())
for name, expected in qa['final_pdf_sha256'].items():
    assert sha(root / (name + '.pdf')) == expected, name
for name, expected in source['included_source_sha256'].items():
    assert sha(root / name) == expected, name
assert qa['all_final_pages_visually_reviewed'] and not qa['invalid_destinations']
assert source['status'] == 'PASS' and source['fresh_reference_count'] == 50
files = {str(p.relative_to(root)).replace('\\', '/'): sha(p)
         for p in [root / 'main.pdf', root / 'supplement.pdf',
                   folder / 'RELEASE_REVIEW.txt', folder / 'final_qa.json',
                   folder / 'final_source_gate_after_visual.json',
                   folder / 'CONTRIBUTION_REVIEW.txt', folder / 'ADVERSARIAL_REVIEW.txt']}
record = {
    'decision': 'SUBMISSION_REVIEW_DRAFT_COMPLETE_SCOPED_AUTHOR_RELEASE_PENDING',
    'assessment_kind': 'Internal technical/editorial judgment, not external peer review',
    'scope': 'Model-specific paid-identification theory and useful/useless boundary',
    'technical_package_complete': True,
    'demonstrated_practical_superiority_to_strongest_fixed': False,
    'acceptance_or_exhaustive_priority_certified': False,
    'author_released': False, 'submitted': False,
    'pending_authentic_human_facts': ['ORCIDs', 'related/prior/concurrent disclosures',
                                     'all-author consent and final release'],
    'pages': qa['pages'], 'all_final_pages_visually_reviewed': True,
    'references': 50, 'preserved_original_direct_formal_blocks': 13,
    'conditional2h': {'science': 294912, 'all_replay': 294912, 'analyses': 2,
                      'audited': True, 'charged_seconds': 5228.984375,
                      'remaining_seconds': 1971.015625,
                      'decision': 'MECHANISM_NOT_CONFIRMED_AT_REGISTERED_THRESHOLD'},
    'observable1h': {'science': 294912, 'all_replay': 294912, 'analyses': 2,
                    'audited': True, 'charged_seconds': 2506.828125,
                    'remaining_seconds': 1093.171875,
                    'decision': 'MECHANISM_NOT_CONFIRMED_AT_REGISTERED_THRESHOLD'},
    'outcome_known_population_normalized_gain_approx': 0.008753338776815613,
    'population_computation_measured_child_cpu_seconds': 57.6875,
    'new_population_diagnostic_samples': 0,
    'ordinary_metered_paper_phase_charge_seconds': 686.78125,
    'includes_allocated_not_measured_auxiliary_seconds': 120,
    'release_git_ui_administration_unmetered_and_not_native_science': True,
    'sha256': files,
}
with (folder / 'release_manifest.json').open('x', encoding='utf-8') as f:
    json.dump(record, f, indent=2)
print(json.dumps({'status': 'PASS', 'pages': qa['pages'], 'files_verified': len(files)}))
