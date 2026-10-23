import hashlib,json,re,subprocess
from pathlib import Path
from pypdf import PdfReader
here=Path(__file__).resolve().parent;paper=here.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pages={}
for name,expected in [('main',13),('supplement',32)]:
    reader=PdfReader(paper/(name+'.pdf'));assert len(reader.pages)==expected
    pages[name]=expected
    log=(paper/(name+'.log')).read_text(errors='replace')
    assert not re.search(r'Overfull|undefined|has been referenced but does not exist',log,re.I)
    fonts=subprocess.check_output(['pdffonts',str(paper/(name+'.pdf'))],text=True)
    assert all(re.search(r'\s+yes\s+yes\s+(yes|no)\s+\d+\s+\d+\s*$',x) for x in fonts.splitlines()[2:] if x.strip())
main=(paper/'main.tex').read_text(encoding='utf-8')
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',main,re.S).group(1)
assert 150<=len(abstract.split())<=220
formal=lambda s:re.findall(r'\\begin\{(assumption|theorem|proposition|corollary|lemma)\}(.*?)\\end\{\1\}',s,re.S)
assert formal(main)==formal((here/'before/main.tex').read_text(encoding='utf-8'))
fresh=json.loads((here/'fresh_citations/citation_verification.json').read_text(encoding='utf-8'))
resolution=json.loads((here/'citation_resolution.json').read_text())
assert len(fresh['records'])==50 and not resolution['unresolved_material_metadata_conflicts']
unresolved={r['key'] for r in fresh['records'] if r['status']!='PASS'}
assert unresolved<={r['key'] for r in resolution['authoritative_resolution']}
assert not json.loads((here/'fresh_citations/bibliography_style.json').read_text())['definite_style_errors']
for p in paper.glob('*.bib'):
    git_blob=subprocess.check_output(['git','show','80db558:'+str(p.relative_to(paper.parents[1])).replace('\\','/')],cwd=paper.parents[1])
    assert p.read_bytes().replace(b'\r\n',b'\n')==git_blob.replace(b'\r\n',b'\n')
rows=json.loads((here/'refined.json').read_text())['rows']
assert next(r['gain'] for r in rows if not r['td'] and r['B']==8192)>.0254
rows=json.loads((here/'transfer.json').read_text())['rows']
assert next(r['gain'] for r in rows if r['B']==8192)>.0186
assert json.loads((here/'checks.json').read_text())['status']=='PASS'
for stem,count in [('main',13),('supp',32)]:assert len(list(here.glob(stem+'-*.png')))==count
for name in ('main.tex','finite_robust_separation.tex','finite_robust_proof.tex','arbitrary_schedule_proof.tex','td_extension_proof.tex'):
    assert not any(line and line[0].isspace() for line in (paper/name).read_text(encoding='utf-8').splitlines())
report=dict(status='PASS_ESSENTIAL_TECHNICAL_CHECKS',pages=pages,abstract_words=len(abstract.split()),
            all_final_pages_visually_reviewed=True,new_native_trajectories=0,
            original_main_formal_blocks_preserved=len(formal(main)),fresh_references=50,
            unresolved_raw_with_explicit_resolutions=sorted(unresolved),
            final_pdf_sha256={n:sha(paper/(n+'.pdf')) for n in pages},
            source_sha256={n:sha(paper/n) for n in ('main.tex','finite_robust_separation.tex','finite_robust_proof.tex','arbitrary_schedule_proof.tex','td_extension_proof.tex')})
with (here/'final_check.json').open('x') as f:json.dump(report,f,indent=2)
print(report)
