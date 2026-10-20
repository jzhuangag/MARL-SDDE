"""Source preservation and claim-scope gate; no simulation or random sampling."""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

START=time.process_time();HERE=Path(__file__).resolve().parent;PAPER=HERE.parents[1]
ROOT=PAPER.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=(HERE/'before/main.tex').read_text(encoding='utf-8')
new=(PAPER/'main.tex').read_text(encoding='utf-8')
pattern=r'\\begin\{(theorem|lemma|proposition|assumption|corollary)\}.*?\\end\{\1\}'
blocks=lambda s:[m.group(0) for m in re.finditer(pattern,s,re.S)]
assert blocks(old)==blocks(new) and len(blocks(new))==13
fixed_names=['appendices.tex','references.bib','recent_work_20261019.bib','transfer_details.tex']
assert all(sha(PAPER/n)==sha(HERE/'before'/n) for n in fixed_names)
supp=(PAPER/'supplement.tex').read_text(encoding='utf-8')
old_supp=(HERE/'before/supplement.tex').read_text(encoding='utf-8')
assert supp.replace('\\input{mechanism_details}\n','')==old_supp
fragment=(PAPER/'mechanism_details.tex').read_text(encoding='utf-8')
for label in ('joint-actions','risk-reduction'):
    find=rf'\\begin\{{figure\}}.*?\\label\{{fig:{label}\}}\s*\\end\{{figure\}}'
    # Select the smallest containing figure rather than spanning prior floats.
    figures=re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',old,re.S)
    prior=next(x for x in figures if '\\label{fig:'+label+'}' in x)
    assert prior in fragment
for text in (
    'There are 23/256 adverse cell means; the low-instance allowance is $0.1$, not zero harm.',
    'Corrected and no-carryover decisions and risks coincide on all 32,768 paired rows, leaving the correction\'s incremental empirical benefit unresolved.',
    'Fixed $(2,1)$ has lower pooled risk and environment cost but uses more messages: the gate does not dominate every control.',
    'Known-model population enclosures are neither confidence limits for sample means nor unknown-kernel calibration guarantees.',
):assert text in old and text in new
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',new,re.S).group(1)
words=len(abstract.split());assert 150<=words<=220
keywords=re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}',new,re.S).group(1)
assert len(keywords.split(','))==5
assert new.count('\\qquad')==old.count('\\qquad')
assert not re.search(r'^[ \t]+\S',fragment,re.M)
assert not re.search(r'(?<!\\)\\$',fragment,re.M), 'Single trailing slash in new source'
spec=importlib.util.spec_from_file_location('citations',HERE/'audit_references.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
entries=module.adapter.bib_entries((PAPER/'references.bib').read_text(encoding='utf-8'))
cited={k.strip() for b in re.findall(r'\\cite\w*\{([^}]+)\}',new+(PAPER/'appendices.tex').read_text(encoding='utf-8')) for k in b.split(',')}
supp_cited=set(re.findall(r'\\maincite\{([^}]+)\}',supp+fragment))
assert set(entries)==cited and supp_cited<=cited and len(cited)==43
summary=json.loads((PAPER/'internal/mechanism_analysis_20261020/complete/analysis_summary.json').read_text())
assert summary['balanced_frozen_certified_advantage_over_all_fixed']==29
assert summary['corrected_admissions']==summary['signed_bound_admissions_unexecuted']==113
for path,expected in summary['inputs'].items():assert sha(ROOT/Path(path))==expected
raw=json.loads((HERE/'fresh_citations/citation_verification.json').read_text())
assert len(raw['records'])==43
assert {x['key'] for x in raw['records'] if x['status']!='PASS'}=={'paulin2015concentration','manjunath2021moments'}
style=json.loads((HERE/'fresh_citations/bibliography_style.json').read_text())
assert not style['definite_style_errors']
for stem in ('esary1967','russo2016','russo2016_supp','manjunath2021'):
    pre=json.loads((HERE/'literature'/(stem+'_preflight.json')).read_text())
    assert pre['verdict']=='PASS'
pre=json.loads((HERE/'literature/chaloner1995_preflight.json').read_text())
assert pre['verdict']=='UNAVAILABLE' and pre['declared_page_count']==33
import pypdf
counts={name:len(pypdf.PdfReader(PAPER/name).pages) for name in ('main.pdf','supplement.pdf')}
changes={}
for name in ('main.tex','supplement.tex'):
    a=(HERE/'before'/name).read_bytes();b=(PAPER/name).read_bytes()
    matcher=difflib.SequenceMatcher(a=a.splitlines(keepends=True),b=b.splitlines(keepends=True),autojunk=False)
    ops=[]
    for tag,i,j,k,l in matcher.get_opcodes():
        if tag!='equal':
            chunk=b''.join(a.splitlines(keepends=True)[i:j])
            ops.append(dict(op=tag,old_first_line=i+1,old_end_exclusive=j+1,
                old_hash=hashlib.sha256(chunk).hexdigest(),new_first_line=k+1,new_end_exclusive=l+1))
    changes[name]=dict(base_sha256=hashlib.sha256(a).hexdigest(),new_sha256=hashlib.sha256(b).hexdigest(),ops=ops,
        untouched_source_lines_preserved=sum(j-i for tag,i,j,k,l in matcher.get_opcodes() if tag=='equal'),
        adapter='LaTeX exact apply_patch edits plus before-byte/diff verification; NOT upstream Markdown anchorizer execution')
report=dict(main_formal_blocks_unchanged=True,formal_blocks=13,original_supplement_preserved=True,
    original_bibs_and_appendices_preserved=True,old_figures_moved_intact=True,
    direct_negative_evidence_preserved=True,abstract_words=words,keywords=5,cited_references=43,
    new_references=4,raw_reference_pass=41,explicit_authoritative_resolutions=2,
    new_study_trajectories=0,new_seeds=0,new_bootstraps=0,original_analysis_inputs_unchanged=True,
    pages=counts,hashes={n:sha(PAPER/n) for n in ('main.tex','supplement.tex','main.pdf','supplement.pdf','mechanism_details.tex','mechanism_references_20261020.bib')},
    patch_manifest=changes,observed_checker_cpu_seconds=time.process_time()-START)
with (HERE/'source_checks.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k not in ('patch_manifest','hashes')}))
