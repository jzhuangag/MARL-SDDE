"""Seal the independent read-only audit; never builds or edits the paper."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'TSP-V3/manuscript'
OUT = BASE/'internal/final_audit_20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = json.loads((OUT/'readonly_checks_complete.json').read_text(encoding='utf-8'))
    for name, expected in checks['preserved_files'].items():
        assert sha(BASE/name) == expected, f'Preserved V3 file changed: {name}'
    assert len(checks['equation_reference_style_findings']) == 40
    assert abs(checks['mpe_mixture_estimands']['paper_mean_regime_seed_relative_gain']-.023194994596377645)<1e-14
    result = checks['exp016b_denominator_audit']['results']
    assert abs(result['primary_layer_A']['actual_relative_reduction_vs_information_only']-.35244339812568215)<1e-14
    assert abs(result['layer_B']['actual_relative_reduction_vs_information_only']-.09789896589137295)<1e-14
    citations = BASE/'internal/final_citation_audit_20261016_a1'
    records = json.loads((citations/'citation_verification.json').read_text(encoding='utf-8'))['records']
    assert len(records)==37 and all(r['resolvers']['authoritative']['status']=='AVAILABLE' for r in records)
    assert {r['key'] for r in records if r['definite_conflicts']} == {'paulin2015concentration'}
    pages = sorted(OUT.glob('page-*.png'))
    assert len(pages)==13
    selected = [
        BASE/'main.tex', BASE/'appendices.tex', BASE/'references.bib', BASE/'main.pdf',
        *sorted((BASE/'figures').glob('*.pdf')),
        OUT/'final_audit.md', OUT/'theory_repair_memo.md', OUT/'citation_resolutions.json',
        OUT/'pdf_preflight.json', OUT/'readonly_checks_complete.json',
        OUT/'tex_diagnostics/main.log', OUT/'tex_diagnostics/main.blg',
        citations/'citation_verification.json', citations/'bibliography_style.json',
        *sorted((citations/'records').glob('*.json')), *pages,
        ROOT/'TSP/internal/marl_mpe_qgrid_audit_gate.json',
        ROOT/'TSP/internal/convergence_curve_validation.json',
        ROOT/'TSP/data/convergence_curve_summary.csv',
        ROOT/'TSP-V3/results/marl_mamujoco_calibrated_confirmation_20260930/gate.json',
        ROOT/'TSP-V3/results/marl_mamujoco_calibrated_confirmation_20260930/summary.csv',
        ROOT/'TSP-V3/results/certificate_robust_shield_confirmation_20261003/summary.json',
        ROOT/'TSP-V3/results/certificate_robust_shield_confirmation_20261003/paired_comparisons.csv',
        ROOT/'experiments/dependence_delay_linear/results/affine_finite_time_certificate/summary.json',
        ROOT/'experiments/dependence_delay_linear/results/linear_td_correlation/summary.json',
        ROOT/'experiments/dependence_delay_linear/results/exp016b_formal_20260801/analysis/cell_summary.csv',
    ]
    decision = dict(
        date='2026-10-16', decision='BLOCKED_SUBMISSION_RELEASE',
        audit_work_complete=True, clean_final_integrity_gate_pass=False,
        manuscript_or_results_modified=False, new_native_experiments_in_audit=0,
        release_blockers=['IL-SERIOUS-1','IL-SERIOUS-2','IL-SERIOUS-3','IL-SERIOUS-4',
                          'IL-MEDIUM-1','IL-MEDIUM-2','IL-MEDIUM-3'],
        final_context_originality_release_gate_complete=False,
        positive_empirical_findings_retained=True,
        independent_40h_science='ACTIVE; no incomplete outcomes inspected; no manuscript admission',
        preserved_files=checks['preserved_files'],
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in selected},
        inspected_pdf_pages=13, figure_fonts='Embedded Times New Roman',
        citation_existence_positive=37, unresolved_existence_failures=0,
        secondary_index_outages_disclosed=True,
        next_step='Non-GPU mathematical/reporting revision after preserving original V3; author metadata and final release re-audit. New actor science need not finish first.',
    )
    with (OUT/'decision.json').open('x', encoding='utf-8') as stream:
        json.dump(decision,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in decision.items() if k!='artifact_sha256'},indent=2))


if __name__=='__main__':
    main()
