"""Read-only source/evidence checks for the isolated V3 repair, not a native run."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from pypdf import PdfReader


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    main = (HERE / 'main.tex').read_text(encoding='utf-8')
    appendix = (HERE / 'appendices.tex').read_text(encoding='utf-8')
    baseline = json.loads((ROOT / 'TSP-V3/manuscript/internal/final_audit_20261016/readonly_checks_complete.json').read_text())
    originals = {}
    for name, expected in baseline['preserved_files'].items():
        actual = sha(ROOT / 'TSP-V3/manuscript' / name)
        originals[name] = dict(sha256=actual, unchanged=actual == expected)
    source_checks = {}
    for name, source in [('main.tex', main), ('appendices.tex', appendix)]:
        source_checks[name] = dict(
            sha256=sha(HERE / name),
            leading_whitespace=[n for n, line in enumerate(source.splitlines(), 1) if re.match(r'^[ \t]+\S', line)],
            bare_equation_refs=[n for n, line in enumerate(source.splitlines(), 1) if re.search(r'(?<!Eq\.~)\\eqref', line)],
            qquad=source.count('\\qquad'),
        )
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', main, re.S).group(1)
    keywords = re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}', main, re.S).group(1)
    log = (HERE / 'main.log').read_text(encoding='utf-8', errors='replace')
    action_path = ROOT / 'experiments/dependence_delay_linear/results/affine_finite_time_certificate/selected_actions.csv'
    with action_path.open(newline='') as stream:
        actions = list(csv.DictReader(stream))
    margins = {key: min(float(row[column]) for row in actions) for key, column in
               [('eta_min', 'eta'), ('mu_delta_min', 'effective_monotonicity'), ('a_delta_min', 'a_delta')]}
    margins['one_minus_c_min'] = min(1 - float(row['contraction']) for row in actions)
    assert all(math.isfinite(x) and x > 0 for x in margins.values())
    citation_path = HERE / 'internal/fresh_citation_20261018/citation_verification.json'
    reader = PdfReader(HERE / 'main.pdf')
    return dict(
        checked_on='2026-10-18',
        kind='ISOLATED_REVISION_INTERMEDIATE_CHECK_NOT_RELEASE',
        originals=originals, sources=source_checks,
        abstract_words=len(abstract.split()), keywords=len(keywords.split(',')),
        pdf=dict(pages=len(reader.pages), sha256=sha(HERE / 'main.pdf'),
                 page_text_characters=[len(page.extract_text()) for page in reader.pages]),
        latex=dict(undefined=bool(re.search(r'undefined|There were undefined', log)),
                   overfull=log.count('Overfull'), underfull=log.count('Underfull')),
        saved_action_domain=dict(source=str(action_path.relative_to(ROOT)), sha256=sha(action_path),
                                 rows=len(actions), positive_margins=margins,
                                 interpretation='Retrospective containment witness for these saved actions only; no saved gate, action, source or result changed. Not evidence of global scalar minimization.'),
        citation_report_sha256=sha(citation_path),
        bibliography_unchanged=sha(HERE / 'references.bib') == baseline['preserved_files']['references.bib'],
        pending=['Full current-PDF visual review', 'Complete proof/claim and citation-context review',
                 'Cross-check every retained simulation protocol against repaired certificate scope',
                 'Real author and submission metadata'],
        submission_release=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
