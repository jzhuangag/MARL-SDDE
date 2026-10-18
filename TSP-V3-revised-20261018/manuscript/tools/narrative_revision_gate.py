"""Read-only revision-fidelity/artifact gate; no native worker or result mutation."""
import argparse
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

import submission_release_check

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PREFIX = 'TSP-V3-revised-20261018/manuscript/'
BASE = 'a667d32a1fc8e42c4cbcfc038765459e61da1a99'


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def old(name):
    return subprocess.check_output(['git', 'show', f'{BASE}:{PREFIX}{name}'], cwd=ROOT)


def check(render, viewed_pages):
    fidelity = {}
    patch_ledger = []
    current_sources = {}
    for name in ['main.tex', 'appendices.tex']:
        before = old(name).decode('utf-8').replace('\r\n', '\n')
        after = (HERE / name).read_text(encoding='utf-8')
        current_sources[name] = after
        inline = r'(?<!\\)\$(?!\$).*?(?<!\\)\$'
        displays = r'\\begin\{(equation\*?|align\*?|multline\*?|gather\*?)\}.*?\\end\{\1\}'
        formal = r'\\begin\{(assumption|theorem|proposition|corollary|lemma)\}.*?\\end\{\1\}'
        # Preserve every math token and entire formal statement, not just its number.
        assert re.findall(inline, before, re.S) == re.findall(inline, after, re.S)
        for pattern in [displays, formal]:
            assert [m.group() for m in re.finditer(pattern, before, re.S)] == [m.group() for m in re.finditer(pattern, after, re.S)]
        for command in ['cite', 'label', 'ref', 'eqref']:
            pattern = r'\\' + command + r'\{[^}]*\}'
            assert re.findall(pattern, before) == re.findall(pattern, after), (name, command)
        before_lines, after_lines = before.splitlines(keepends=True), after.splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
        for tag, i, j, k, l in matcher.get_opcodes():
            if tag == 'equal':
                continue
            patch_ledger.append(dict(file=name, operation=tag, old_lines=[i+1,j],
                                     new_lines=[k+1,l],
                                     old_sha256=sha_bytes(''.join(before_lines[i:j]).encode()),
                                     new_sha256=sha_bytes(''.join(after_lines[k:l]).encode()),
                                     review_items=['R-NARRATIVE', 'R-NONDEFENSIVE', 'R-KIMI'],
                                     before=''.join(before_lines[i:j]), after=''.join(after_lines[k:l])))
        fidelity[name] = dict(base_sha256=sha_bytes(old(name)), revised_sha256=sha(HERE/name),
                              all_inline_math_identical=True, all_displays_identical=True,
                              all_formal_statements_identical=True,
                              citation_and_reference_command_sequences_identical=True)
    author = r'\\author\{.*?(?=\\begin\{document\})'
    assert re.search(author, old('main.tex').decode(), re.S).group() == re.search(author, current_sources['main.tex'], re.S).group()
    assets = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE, '--',
                                      PREFIX+'figures'], cwd=ROOT, text=True).splitlines()
    assert assets
    asset_hashes = {}
    for name in ['references.bib'] + [p.removeprefix(PREFIX) for p in assets]:
        assert sha_bytes(old(name)) == sha(HERE/name)
        asset_hashes[name] = sha(HERE/name)
    # Critical metrics/interval endpoints, including unfavorable and null evidence.
    metrics = ['35.24', '9.79', '0.1612', '0.0577', '32.94', '43.72', '30.13',
               '0.9009', '0.8914', '0.9102', '0.8868', '0.8678', '0.9072',
               '26.48', '21.67', '2.319', '8.341', '-0.542', '-0.997',
               '1.107', '0.0352', '4.637', '12.20', '64.22', '0.305', '20.77', '228.98']
    for value in metrics:
        assert value in current_sources['main.tex'], value
    report = submission_release_check.check(render, viewed_pages)
    fresh_path = HERE / 'internal/tsp_narrative_citation_20261018/citation_verification.json'
    style_path = fresh_path.with_name('bibliography_style.json')
    resolution_path = HERE / 'internal/tsp_narrative_citation_resolution_20261018.json'
    fresh = json.loads(fresh_path.read_text())
    style = json.loads(style_path.read_text())
    resolution = json.loads(resolution_path.read_text())
    assert len(fresh['records']) == 37 and not fresh['preserved_records_used']
    assert sum(x['status'] == 'PASS' for x in fresh['records']) == 35
    assert set(fresh['unresolved_material_conflicts']) == {'paulin2015concentration','perazzone2025lyapunov'}
    assert resolution['authoritative_existence_matches'] == 37
    assert fresh['bibliography_sha256'] == sha(HERE/'references.bib')
    assert not style['definite_style_errors'] and not style['bib_edited']
    preflight_path = HERE/'internal/tsp_narrative_preflight_20261018.json'
    preflight = json.loads(preflight_path.read_text())
    assert preflight['verdict'] == 'PASS' and preflight['sha256'] == report['pdf']['sha256']
    report.update(
        kind='FULL_TEXT_TSP_POSITIONING_AND_PROSE_REVISION_FIDELITY_GATE',
        decision='TSP_CANDIDATE_EDITORIAL_REVIEW_NOT_ACCEPTANCE_CLEARANCE',
        base_commit=BASE, fidelity=fidelity, original_author_block_unchanged=True,
        unchanged_assets=asset_hashes, preserved_critical_metrics=metrics,
        patch_adapter='Native LaTeX apply_patch and SHA/diff sidecar; no Markdown HTML markers inserted into TeX',
        patch_ledger=patch_ledger, revision_round=1,
        citation_report_sha256=sha(fresh_path), citation_resolution_sha256=sha(resolution_path),
        bibliography_style_sha256=sha(style_path),
        citation_lookup_main_sha256=fresh['manuscript_sha256'],
        citation_final_main_sha256=sha(HERE/'main.tex'),
        post_lookup_changes='Local prose compression; all ordered citation groups and Bib bytes unchanged, final context reviewed',
        fresh_citations=dict(raw_pass=35, publisher_rate_limit_resolution=1,
                             article_number_resolution=1,total=37,rate_limits_preserved=True),
        pending=['Authentic author ORCIDs required by IEEE submission system',
                 'Human prior/concurrent/related-manuscript disclosures and all-author consent/final release'],
        scientific_depth_judgment='Substantive resource-control theory/experiments; not a new fully validated asynchronous MARL optimizer',
        pdf_preflight_sha256=sha(preflight_path),
        submission_release=False, submitted=False, new_scientific_data_used=False,
        independent_external_review=False, detector_evasion_claim=False,
    )
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', type=Path, required=True)
    parser.add_argument('--viewed-pages', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.render.resolve(), args.viewed_pages)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:report[k] for k in ['decision','pdf','latex','fresh_citations','pending']},indent=2))
