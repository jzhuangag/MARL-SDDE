"""Metadata-only artifact check; does not execute or inspect native learning."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from pypdf import PdfReader
import submission_release_check

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PREFIX = 'TSP-V3-revised-20261018/manuscript/'
BASE = '3407febe1b5018bb2c6e2cda590a1018b652c00b'
AUTHOR = r'''\author{Jialin~Zhuang,~\IEEEmembership{Member,~IEEE,}
and~Vincent~K.~N.~Lau,~\IEEEmembership{Fellow,~IEEE}%
\thanks{J.~Zhuang and V.~Lau are with the Department of Electronic and Computer Engineering, The Hong Kong University of Science and Technology, Clear Water Bay, Hong Kong SAR, China (e-mail: \mbox{jzhuangag@connect.ust.hk}; \mbox{eeknlau@ece.ust.hk}).
This work was supported by the Research Grants Council (RGC), Hong Kong, under the Areas of Excellence Scheme Grant AoE/E-101/23-N and AoE/E-601/22-R.
(\em{Corresponding author: Vincent K. N. LAU}.)}
}'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_bytes(name):
    return subprocess.check_output(['git', 'show', f'{BASE}:{PREFIX}{name}'], cwd=ROOT)


def check(render, viewed_pages):
    main = (HERE / 'main.tex').read_text(encoding='utf-8')
    baseline_main = old_bytes('main.tex').decode('utf-8').replace('\r\n', '\n')
    assert baseline_main.count(r'\author{Anonymous Authors}') == 1
    assert main == baseline_main.replace(r'\author{Anonymous Authors}', AUTHOR)
    preserved = {}
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE, '--',
                                     PREFIX + 'figures'], cwd=ROOT, text=True).splitlines()
    names = ['appendices.tex', 'references.bib'] + [x.removeprefix(PREFIX) for x in files]
    for name in names:
        expected = hashlib.sha256(old_bytes(name)).hexdigest()
        preserved[name] = dict(sha256=sha(HERE / name), unchanged=sha(HERE / name) == expected)
        assert preserved[name]['unchanged']
    assert files, 'Require the tracked figure set, not an empty preservation check.'
    report = submission_release_check.check(render, viewed_pages)
    folder = HERE / 'internal/author_metadata_citation_20261018'
    fresh_path = folder / 'citation_verification.json'
    fresh = json.loads(fresh_path.read_text(encoding='utf-8'))
    style_path = folder / 'bibliography_style.json'
    style = json.loads(style_path.read_text(encoding='utf-8'))
    resolution_path = HERE / 'internal/author_metadata_citation_resolution_20261018.json'
    resolution = json.loads(resolution_path.read_text(encoding='utf-8'))
    assert len(fresh['records']) == 37 and not fresh['preserved_records_used']
    citation_checked_main = main
    for email in ['jzhuangag@connect.ust.hk', 'eeknlau@ece.ust.hk']:
        citation_checked_main = citation_checked_main.replace(r'\mbox{' + email + '}', email)
    assert fresh['manuscript_sha256'] == hashlib.sha256(citation_checked_main.encode()).hexdigest()
    assert fresh['bibliography_sha256'] == sha(HERE / 'references.bib')
    assert sum(x['status'] == 'PASS' for x in fresh['records']) == 36
    assert fresh['unresolved_material_conflicts'] == ['paulin2015concentration']
    assert resolution['resolution']['identity_verified']
    assert resolution['resolution']['fresh_lookup']
    assert resolution['fresh_authoritative_existence_matches'] == 37
    assert not style['definite_style_errors'] and not style['bib_edited']
    self_citations = [x['key'] for x in fresh['records']
                      if re.search(r'\b(?:Zhuang|Lau)\b', x['bib_fields'].get('author', ''))]
    reader = PdfReader(HERE / 'main.pdf')
    first_page = re.sub(r'\s+', '', reader.pages[0].extract_text())
    for token in ['Jialin Zhuang', 'Vincent K. N. Lau', 'jzhuangag@connect.ust.hk',
                  'eeknlau@ece.ust.hk', 'AoE/E-101/23-N', 'AoE/E-601/22-R',
                  'Corresponding author: Vincent K. N. LAU']:
        assert re.sub(r'\s+', '', token) in first_page, token
    assert 'AnonymousAuthors' not in first_page
    preflight_path = HERE / 'internal/author_metadata_unbroken_preflight_20261018.json'
    preflight = json.loads(preflight_path.read_text())
    assert preflight['verdict'] == 'PASS' and preflight['sha256'] == report['pdf']['sha256']
    report.update(
        kind='AUTHOR_METADATA_ONLY_POST_BUILD_AUDIT',
        decision='AUTHOR_METADATA_VERIFIED_SUBMISSION_DECLARATIONS_PENDING',
        base_commit=BASE, metadata_source='Direct human-provided LaTeX author block',
        metadata=dict(ordered_authors=['Jialin Zhuang', 'Vincent K. N. Lau'],
                      affiliation='ECE, HKUST, Clear Water Bay, Hong Kong SAR, China',
                      corresponding_author='Vincent K. N. Lau',
                      grants=['AoE/E-101/23-N', 'AoE/E-601/22-R'],
                      orcids=None, prior_or_concurrent_submission=None, all_author_consent=None),
        author_block_sha256=hashlib.sha256(AUTHOR.encode()).hexdigest(),
        only_author_block_changed=True, scientific_body_unchanged_from_base=True,
        prior_revised_assets_preserved=preserved,
        citation_report_sha256=sha(fresh_path),
        citation_checked_main_sha256=fresh['manuscript_sha256'],
        citation_postcheck_delta='Only two email nonbreaking mbox wrappers; all citation commands and bibliography bytes unchanged',
        citation_resolution_sha256=sha(resolution_path),
        bibliography_style_sha256=sha(style_path),
        fresh_citations=dict(raw_pass=36, article_number_resolution=1, total=37,
                             index_outages_not_bypassed=True,
                             self_citation_keys=self_citations,
                             self_citation_count=len(self_citations)),
        pdf_preflight_sha256=sha(preflight_path),
        pending=['Authentic ORCIDs if required by the submission system',
                 'Human confirmation of prior/concurrent submission or related-manuscript disclosures',
                 'Human confirmation of all-author consent and final release'],
        submission_release=False, submitted=False,
    )
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', type=Path, required=True)
    parser.add_argument('--viewed-pages', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.render.resolve(), args.viewed_pages)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ['decision', 'pdf', 'latex', 'fresh_citations', 'pending']}, indent=2))
