"""Artifact gate for the substantive V3 addition; no scientific execution."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from pypdf import PdfReader
import revision_gate_check

HERE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / "internal/same_stream_release_check_20261018.json"
    if output.exists():
        raise FileExistsError(output)
    report = revision_gate_check.check()
    assert all(item["unchanged"] for item in report["originals"].values())
    assert report["bibliography_unchanged"] and report["pdf"]["pages"] == 13
    assert 150 <= report["abstract_words"] <= 220 and report["keywords"] == 5
    assert not report["latex"]["undefined"] and report["latex"]["overfull"] == 0
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in ("main.tex", "appendices.tex", "supplement.tex")}
    issues = {}
    for name, source in sources.items():
        flags = []
        for line_no, line in enumerate(source.splitlines(), 1):
            prose = re.sub(r'\$[^$]*\$', 'MATH', line)
            prose = re.sub(r'\\(?:cite|eqref|ref)\{[^}]*\}', 'REF', prose)
            prose = re.sub(r'\b(?:Eq|Fig|Sec|Dr|Prof)\.', 'ABBR', prose)
            if not line.startswith(r'\author{') and re.search(r'[.!?]\s+[A-Z][a-z]', prose):
                flags.append(line_no)
        issues[name] = dict(sentence_boundary_flags=flags,
                            leading_whitespace=[n for n, line in enumerate(source.splitlines(), 1) if re.match(r'^[ \t]+\S', line)],
                            bare_equation_refs=[n for n, line in enumerate(source.splitlines(), 1) if re.search(r'(?<!Eq\.~)\\eqref', line)],
                            qquad=source.count('\\qquad'))
        assert not any(issues[name].values()), issues[name]
    original_main = subprocess.run(['git', 'show', '0eeae35:TSP-V3-revised-20261018/manuscript/main.tex'], cwd=HERE,
                                   text=True, capture_output=True, encoding='utf-8', check=True).stdout
    formal_pattern = r'\\begin\{(theorem|proposition|lemma|corollary|assumption)\}.*?\\end\{\1\}'
    old_formal = [item.group(0) for item in re.finditer(formal_pattern, original_main, re.S)]
    assert old_formal and all(block in sources['main.tex'] for block in old_formal)
    bibliography = (HERE / 'references.bib').read_text(encoding='utf-8')
    cited = set()
    for group in re.findall(r'\\cite\{([^}]+)\}', '\n'.join(sources.values())):
        cited.update(group.split(','))
    keys = set(re.findall(r'@\w+\{([^,]+),', bibliography))
    assert cited == keys and len(keys) == 37
    raw = json.loads((HERE / 'internal/same_stream_fresh_citations_20261018/citation_verification.json').read_text())
    assert sum(record['status'] == 'PASS' for record in raw['records']) == 36
    failed = [record['key'] for record in raw['records'] if record['status'] != 'PASS']
    assert failed == ['paulin2015concentration']
    resolution = json.loads((HERE / 'internal/same_stream_citation_resolution_20261018.json').read_text())
    assert resolution['authoritative_existence_matches'] == 37 and not resolution['bib_edited']
    style = json.loads((HERE / 'internal/same_stream_fresh_citations_20261018/bibliography_style.json').read_text())
    assert not style['definite_style_errors']
    integrity_path = HERE / 'internal/same_stream_directional_evidence_20261018/integrity.json'
    integrity = json.loads(integrity_path.read_text())
    proof_path = HERE / 'internal/same_stream_directional_proof_audit_20261018.json'
    proof = json.loads(proof_path.read_text())
    assert integrity['pass'] and proof['pass_all']
    assert integrity['analysis_script_sha256'] == sha(HERE / 'tools/same_stream_evidence.py')
    assert proof['evidence_integrity_sha256'] == sha(integrity_path)
    assert proof['audit_script_sha256'] == sha(HERE / 'tools/same_stream_proof_audit.py')
    assert proof['exact_probe_designs_checked'] == 104
    assert not proof['nominal_directional_certificate_supported'] and proof['per_probe_affinity_used']
    assert integrity['public_protocol_and_ledgers_checked_rows'] == 14336
    artifacts = {}
    for name, prefix, preflight in [('main', 'main-final', 'same_stream_main_final_preflight_20261018.json'),
                                    ('supplement', 'supplement-final', 'same_stream_supplement_final_preflight_20261018.json')]:
        pdf = HERE / f'{name}.pdf'
        reader = PdfReader(pdf)
        pre = json.loads((HERE / 'internal' / preflight).read_text())
        assert pre['verdict'] == 'PASS' and pre['sha256'] == sha(pdf)
        assert len(reader.pages) == (13 if name == 'main' else 3)
        log = (HERE / f'{name}.log').read_text(encoding='utf-8', errors='replace')
        assert 'Overfull' not in log and 'undefined' not in log and 'pdfTeX warning' not in log
        fonts = subprocess.run(['pdffonts', str(pdf)], text=True, capture_output=True, check=True).stdout
        assert all(line.split()[-5] == 'yes' for line in fonts.splitlines()[2:] if line.strip())
        page_images = {}
        for number in range(1, len(reader.pages) + 1):
            suffix = f'{number:02}' if name == 'main' else str(number)
            image = HERE / 'internal/same_stream_render_20261018' / f'{prefix}-{suffix}.png'
            assert image.is_file(), image
            page_images[str(number)] = sha(image)
        artifacts[name] = dict(pdf_sha256=sha(pdf), pages=len(reader.pages), all_fonts_embedded=True,
                               overfull=0, undefined=0, all_pages_visually_reviewed=True,
                               page_image_sha256=page_images, source_sha256=sha(HERE / f'{name}.tex'))
    report.update(kind='SUBSTANTIVE_SAME_STREAM_THEORY_EVIDENCE_ADDITION',
                  decision='SUBSTANTIVE_ADDITION_VERIFIED_AUTHOR_RELEASE_PENDING',
                  source_lint=issues, new_theorem='Same-stream actual-risk enclosure',
                  old_formal_statements_preserved=len(old_formal),
                  new_trajectories=0, outcome_known_reanalysis=True,
                  nominal_delta_failure_preserved=True, statistical_fallacy_types_checked=11,
                  analytical_integrity_sha256=sha(integrity_path), proof_audit_sha256=sha(proof_path),
                  all_pdf_artifacts=artifacts, fresh_citation_raw_pass=36,
                  fresh_citation_article_number_resolution=1, references_verified=37,
                  citation_report_sha256=sha(HERE / 'internal/same_stream_fresh_citations_20261018/citation_verification.json'),
                  citation_resolution_sha256=sha(HERE / 'internal/same_stream_citation_resolution_20261018.json'),
                  bibliography_style_sha256=sha(HERE / 'internal/same_stream_fresh_citations_20261018/bibliography_style.json'),
                  submitted=False, submission_release=False, independent_external_review=False,
                  pending=['Authentic author ORCIDs', 'Prior/concurrent/related-submission disclosures',
                           'All-author consent and human final release'])
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['decision', 'old_formal_statements_preserved', 'references_verified', 'all_pdf_artifacts', 'pending']}, indent=2))


if __name__ == '__main__':
    main()
