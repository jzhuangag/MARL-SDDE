"""Final isolated-revision artifact gate. Never executes a scientific payload."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import revision_gate_check

HERE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(render, viewed_pages):
    report = revision_gate_check.check()
    main = (HERE / 'main.tex').read_text(encoding='utf-8')
    appendix = (HERE / 'appendices.tex').read_text(encoding='utf-8')
    bibliography = (HERE / 'references.bib').read_text(encoding='utf-8')
    context_path = HERE / 'internal/citation_context_20261018.json'
    context = json.loads(context_path.read_text(encoding='utf-8'))
    cited = set()
    for group in re.findall(r'\\cite\{([^}]+)\}', main + appendix):
        cited.update(group.split(','))
    keys = set(re.findall(r'@\w+\{([^,]+),', bibliography))
    assert cited == keys == {entry['key'] for entry in context['entries']}
    assert len(keys) == 37
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', main, re.S).group(1)
    assert 150 <= report['abstract_words'] <= 220
    assert not re.search(r'\b[A-Z]{2,}\b', abstract)
    assert report['keywords'] == 5 and report['pdf']['pages'] == 13
    assert all(item['unchanged'] for item in report['originals'].values())
    assert not report['latex']['undefined'] and report['latex']['overfull'] == 0
    assert r'\usepackage[margin=1in]{geometry}' in main
    assert r'\documentclass[journal]{IEEEtran}' in main
    assert r'\IEEEtriggeratref' not in main
    sentence_flags = {}
    for name, source in [('main.tex', main), ('appendices.tex', appendix)]:
        flags = []
        for number, line in enumerate(source.splitlines(), 1):
            # Mask math, citations and IEEE cross-reference abbreviations.
            prose = re.sub(r'\$[^$]*\$', 'MATH', line)
            prose = re.sub(r'\\(?:cite|eqref|ref)\{[^}]*\}', 'REF', prose)
            prose = re.sub(r'\b(?:Eq|Fig|Sec|Dr|Prof)\.', 'ABBR', prose)
            if re.search(r'[.!?]\s+[A-Z][a-z]', prose):
                flags.append(number)
        sentence_flags[name] = flags
        assert not flags, (name, flags)
        assert not report['sources'][name]['leading_whitespace']
        assert not report['sources'][name]['bare_equation_refs']
        assert report['sources'][name]['qquad'] == 0
    assert viewed_pages == list(range(1, 14)), 'Human-visible review must precede this gate.'
    images = {}
    for number in viewed_pages:
        path = render / f'page-{number:02}.png'
        assert path.is_file()
        images[str(number)] = sha(path)
    font_report = subprocess.run(['pdffonts', str(HERE / 'main.pdf')],
                                 capture_output=True, text=True, check=True).stdout
    # Poppler's columns end with emb/sub/uni/object ID; all fonts must embed.
    font_lines = font_report.splitlines()[2:]
    assert font_lines
    for line in font_lines:
        fields = line.split()
        assert fields[-5] == 'yes', line
    clock = HERE / 'internal/clock_protocol_audit_20261018.json'
    clock_data = json.loads(clock.read_text(encoding='utf-8'))
    assert clock_data['all_integer_clock_cases_match']
    assert not clock_data['native_execution'] and not clock_data['old_files_modified']
    report.update(
        kind='FINAL_TECHNICAL_REPAIR_AUDIT_NOT_JOURNAL_ACCEPTANCE',
        decision='TECHNICAL_REPAIRS_VERIFIED_AUTHOR_RELEASE_PENDING',
        citation_context=dict(entries=len(keys), sha256=sha(context_path),
                              full_text_plagiarism_or_priority_clearance=False),
        clock_audit_sha256=sha(clock),
        format=dict(class_name='IEEEtran journal, default10pt', body_margins_inches=1,
                    abstract_abbreviations=False, forced_reference_break=False,
                    sentence_boundary_flags=sentence_flags),
        visual=dict(pdf_sha256=report['pdf']['sha256'], viewed_pages=viewed_pages,
                    render_directory=str(render.relative_to(HERE)), page_image_sha256=images,
                    all_fonts_embedded=True, poppler_fonts=font_report),
        pending=['Human-provided author order, affiliations, contact author and ORCIDs',
                 'Funding/related-work or prior-submission disclosures and all-author consent',
                 'Rebuild/recheck pagination after authentic author metadata is inserted'],
        submission_release=False, submitted=False, new_scientific_data_used=False,
        independent_external_review=False,
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
    print(json.dumps({key: report[key] for key in ['decision', 'pdf', 'latex', 'pending']}, indent=2))
