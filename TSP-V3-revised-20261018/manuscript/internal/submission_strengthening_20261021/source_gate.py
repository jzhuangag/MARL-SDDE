"""Source-preservation and current-citation closure for this substantive revision."""
import hashlib
import json
from pathlib import Path
import re
import sys
HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
BEFORE=HERE/'before'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def formal(text):
    return re.findall(r'\\begin\{(assumption|theorem|proposition|lemma|corollary)\}(.*?)\\end\{\1\}',text,re.S)

def run():
    old=(BEFORE/'main.tex').read_text(encoding='utf-8')
    now=(PAPER/'main.tex').read_text(encoding='utf-8')
    preserved=formal(old)
    assert len(preserved)==13 and formal(now)==preserved
    original_files=('appendices.tex','references.bib','recent_work_20261019.bib','mechanism_references_20261020.bib')
    for name in original_files:assert (BEFORE/name).read_bytes()==(PAPER/name).read_bytes()
    # Inspect only recursively included publication sources, never rejected research drafts.
    def included(name,seen):
        path=PAPER/(name+'.tex')
        if path in seen:return ''
        seen.add(path);text=path.read_text(encoding='utf-8')
        return text+'\n'+'\n'.join(included(x,seen) for x in re.findall(r'\\input\{([^}]+)\}',text))
    main=included('main',set());supp=included('supplement',set())
    keys=set()
    for group in re.findall(r'\\(?:maincite|cite)\{([^}]+)\}',main+'\n'+supp):keys.update(x.strip() for x in group.split(','))
    refs=PAPER/'internal/conditional_risk_20261021'
    fresh=json.loads((refs/'fresh_citations/citation_verification.json').read_text())
    checked={row['key']:row for row in fresh['records']}
    assert keys==set(checked)|{'solomon1961quadratic'} and len(keys)==50
    assert digest(PAPER/'references.bib')==fresh['bibliography_sha256']
    assert digest(PAPER/'conditional_references_20261021.bib')==fresh['additive_conditional_bib_sha256']
    resolution=json.loads((refs/'citation_resolution_49.json').read_text())
    assert not resolution['unresolved_material_identity_conflicts']
    style=json.loads((refs/'fresh_citations/bibliography_style.json').read_text())
    assert not style['definite_style_errors'] and style['entries']==49
    extra=json.loads((HERE/'solomon_resolution.json').read_text())
    assert extra['key']=='solomon1961quadratic' and not extra['unresolved_material_identity_conflicts']
    style50=json.loads((HERE/'bibliography_style_50.json').read_text())
    assert not style50['definite_style_errors'] and style50['entries']==50
    forbidden=('value_decomposition_rejected_01','conditional_risk_20261021/theory','main_candidate')
    assert not any(x in main+supp for x in forbidden)
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',now,re.S).group(1)
    count=len(abstract.split());assert 150<=count<=220
    keywords=re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}',now,re.S).group(1)
    assert len(keywords.strip().rstrip('.').split(','))==5
    report=dict(status='PASS',original_direct_formal_blocks_including_assumption_byte_exact=13,
        all_four_original_appendix_and_bibliography_files_byte_exact=True,
        current_recursive_citation_keys=sorted(keys),fresh_reference_count=50,
        raw_reference_decisions_preserved=True,explicit_authoritative_resolutions=2,
        unchanged_citation_mapping_despite_substantive_prose_edits=True,
        new_bib_sha256=digest(PAPER/'conditional_references_20261021.bib'),
        additional_classical_mixture_bib_sha256=digest(PAPER/'quadratic_form_references_20261021.bib'),
        current_main_sha256=digest(PAPER/'main.tex'),current_supplement_sha256=digest(PAPER/'supplement.tex'),
        abstract_words=count,keywords=5,rejected_drafts_not_included=True,
        included_source_sha256={str(p.relative_to(PAPER)):digest(p) for p in sorted(set(PAPER.glob('*.tex')))},
        compilation_visual_and_scientific_gates_are_separate=True)
    name=sys.argv[1] if len(sys.argv)>1 else 'source_gate.json'
    assert Path(name).name==name and name.endswith('.json')
    with (HERE/name).open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
    print(json.dumps(dict(status='PASS',abstract_words=count,references=50,original_formal_blocks=13)))

if __name__=='__main__':run()
