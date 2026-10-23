"""Read-only V3 audit arithmetic. No trajectories, figures or manuscript rewrites."""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MANUSCRIPT = ROOT / 'TSP-V3/manuscript'
OUT = MANUSCRIPT / 'internal/final_audit_20261016'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    tex = (MANUSCRIPT / 'main.tex').read_text(encoding='utf-8')
    appendix = (MANUSCRIPT / 'appendices.tex').read_text(encoding='utf-8')
    citation = load('TSP-V3/manuscript/internal/final_citation_audit_20261016_a1/citation_verification.json')
    checks = dict(preserved_files={
        name: digest(MANUSCRIPT / name)
        for name in ('main.tex', 'appendices.tex', 'references.bib', 'main.pdf')
    })
    checks['abstract_words_whitespace'] = len(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S).group(1).split())
    checks['keywords'] = re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}', tex, re.S).group(1).strip().split(',')
    checks['equation_reference_style_findings'] = [
        dict(file=name, line=i, text=line)
        for name, source in (('main.tex', tex), ('appendices.tex', appendix))
        for i, line in enumerate(source.splitlines(), 1)
        if re.search(r'(?<!Eq\.~)\\eqref\{', line)
    ]
    checks['citation_resolver_status_counts'] = {
        resolver: dict(collections.Counter(r['resolvers'][resolver].get('status', 'UNKNOWN') for r in citation['records']))
        for resolver in ('authoritative', 'SemanticScholar', 'OpenAlex', 'arXiv')
    }
    checks['primary_citation_conflicts'] = {
        r['key']: r['definite_conflicts'] for r in citation['records'] if r['definite_conflicts']
    }
    log = (OUT / 'tex_diagnostics/main.log').read_text(encoding='utf-8', errors='replace')
    checks['tex_draft_diagnostics'] = dict(
        undefined_reference_or_citation=bool(re.search(r'(Reference|Citation).*undefined|There were undefined', log)),
        overfull_count=log.count('Overfull'), underfull_count=log.count('Underfull'),
        no_pdf_written='pdfdraftmode enabled, not changing output pdf' in log,
    )
    mpe_path = 'TSP/internal/marl_mpe_qgrid_audit_gate.json'
    mpe = load(mpe_path)
    rows = {(r['training_seed'], r['coupling'], r['method']): r['return_auc'] for r in mpe['records']}
    paired = []
    for seed in sorted({key[0] for key in rows}):
        controller = sum(rows[(seed, c, 'lyapunov_probe_commit')] for c in ('independent', 'shared')) / 2
        fixed = sum(rows[(seed, c, 'fixed_q8')] for c in ('independent', 'shared')) / 2
        paired.append((controller-fixed)/abs(fixed))
    checks['mpe_mixture_estimands'] = dict(
        source=mpe_path, source_sha256=digest(ROOT/mpe_path),
        mean_paired_relative_gain=sum(paired)/len(paired),
        paper_mean_regime_seed_relative_gain=sum(
            (rows[(seed, c, 'lyapunov_probe_commit')]-rows[(seed, c, 'fixed_q8')])
            / abs(rows[(seed, c, 'fixed_q8')])
            for seed in sorted({key[0] for key in rows}) for c in ('independent', 'shared')
        ) / (2*len(paired)),
        ratio_of_aggregate_means_gain=mpe['controller_relative_gain_over_descriptive_strong_static'],
        paired_seeds=len(paired), scope='Unknown-regime equal mixture versus strongest single fixed q8, not each regime optimum',
    )

    # Reconstruct only already-published aggregate cells; never call a native runner.
    sys.path.insert(0, str(ROOT / 'experiments/dependence_delay_linear'))
    from analyze_exp016b_pilot import analytic_cell_tags
    manifest = load('docs/exp016b_scenario_manifest.json')
    tags = analytic_cell_tags(manifest)
    base = ROOT / 'experiments/dependence_delay_linear/results/exp016b_formal_20260801/analysis'
    cells = pd.read_csv(base / 'cell_summary.csv')
    core = json.loads((base / 'core_results.json').read_text(encoding='utf-8'))
    assert digest(base / 'cell_summary.csv') == core['cell_summary_sha256']
    keys = ['layer', 'scenario_id', 'budget_point', 'scale', 'regime']
    frame = cells.pivot(index=keys, columns='policy', values='mean_terminal_risk').reset_index()
    frame = frame.merge(tags, on=keys[1:], how='left', validate='many_to_one')
    primary = frame[frame.finite_threshold_gate_eligible & frame.practical_Z]
    audits = {}
    first = 'A_gaussian_mechanism'
    for name, selected in [
        ('primary_layer_A', primary[primary.layer.eq(first)]),
        ('layer_B', primary[~primary.layer.eq(first)]),
        ('delay_active', primary[primary.layer.eq(first) & primary.delay_active]),
        ('message_binding', primary[primary.layer.eq(first) & primary.binding_type.eq('message')]),
        ('environment_binding', primary[primary.layer.eq(first) & primary.binding_type.eq('environment')]),
    ]:
        means = {p: float(selected[p].mean()) for p in ('always_all', 'information_only', 'learning_aware')}
        improvement = means['information_only']-means['learning_aware']
        normalized = improvement/means['always_all']
        advertised = core[name]['relative_difference'] if name in core else core['subset_results'][name]['relative_difference']
        assert abs(normalized-advertised) < 1e-12
        audits[name] = dict(
            aggregate_cells=len(selected), means=means,
            saved_normalized_paired_improvement=normalized,
            actual_relative_reduction_vs_information_only=improvement/means['information_only'],
            denominator_mismatch_in_paper=True,
        )
    checks['exp016b_denominator_audit'] = dict(
        source=str((base/'cell_summary.csv').relative_to(ROOT)),
        source_sha256=digest(base/'cell_summary.csv'),
        analysis_source='experiments/dependence_delay_linear/analyze_exp016b_pilot.py:103',
        analysis_sha256=digest(ROOT/'experiments/dependence_delay_linear/analyze_exp016b_pilot.py'),
        results=audits,
    )
    checks['visual_review'] = dict(pages=list(range(1, 14)),
        clipped_or_overlapping_content=False, figures_times_new_roman_embedded=True,
        fraction_ticks=['0.00', '0.25', '0.50', '0.75', '1.00'],
        subfigure_labels_below=True, fig2_high_persistence_ticks_crowded=True)
    with (OUT/'readonly_checks_complete.json').open('x', encoding='utf-8') as stream:
        json.dump(checks, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({k: checks[k] for k in ('mpe_mixture_estimands', 'exp016b_denominator_audit', 'citation_resolver_status_counts', 'tex_draft_diagnostics', 'abstract_words_whitespace')}, indent=2))


if __name__ == '__main__':
    main()
