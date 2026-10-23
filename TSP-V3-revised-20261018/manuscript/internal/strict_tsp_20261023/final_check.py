"""Minimal release checks; scientific significance is judged separately."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import pypdf

folder=Path(__file__).resolve().parent;paper=folder.parents[1]
assert sys.argv[1:]==['--visually-reviewed']
target=folder/'final_check.json';assert not target.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
formal=lambda s:re.findall(r'\\begin\{(assumption|theorem|proposition|corollary|lemma)\}(.*?)\\end\{\1\}',s,re.S)
main=(paper/'main.tex').read_text(encoding='utf-8')
assert formal(main)==formal((folder/'before/main.tex').read_text(encoding='utf-8'))
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',main,re.S).group(1)
assert 150<=len(abstract.split())<=220
seen=set()
def read(stem):
    path=paper/(stem+'.tex')
    if path in seen:return ''
    seen.add(path);text=path.read_text(encoding='utf-8')
    return text+'\n'+'\n'.join(read(s) for s in re.findall(r'\\input\{([^}]+)\}',text))
text=read('main')+'\n'+read('supplement')
keys={k for chunk in re.findall(r'\\(?:cite|maincite)\{([^}]+)\}',text) for k in chunk.split(',')}
fresh=json.loads((folder/'fresh_citations/citation_verification.json').read_text(encoding='utf-8'))
assert keys=={r['key'] for r in fresh['records']} and len(keys)==50
assert sum(r['status']=='PASS' for r in fresh['records'])==47
resolved=json.loads((folder/'citation_resolution.json').read_text())
assert not resolved['unresolved_material_metadata_conflicts']
style=json.loads((folder/'fresh_citations/bibliography_style.json').read_text())
assert not style['definite_style_errors']
readers={n:pypdf.PdfReader(paper/(n+'.pdf')) for n in ('main','supplement')}
assert len(readers['main'].pages)==13
links=0
for name,reader in readers.items():
    log=(paper/(name+'.log')).read_text(errors='replace')
    assert not re.search(r'Overfull|undefined|has been referenced but does not exist',log,re.I)
    fonts=subprocess.check_output(['pdffonts',str(paper/(name+'.pdf'))],text=True)
    assert all(re.search(r'\s+yes\s+yes\s+(yes|no)\s+\d+\s+\d+\s*$',x) for x in fonts.splitlines()[2:] if x.strip())
    prefix='main' if name=='main' else 'supp'
    assert len(list((folder/'render').glob(prefix+'-*.png')))==len(reader.pages)
    for page in reader.pages:
        for ref in page.get('/Annots',[]):
            action=ref.get_object().get('/A')
            if not action:continue
            action=action.get_object();kind=action.get('/S');dest=action.get('/D')
            if kind=='/GoTo' and isinstance(dest,str):assert dest in reader.named_destinations
            if kind=='/GoToR' and str(action.get('/F','')) in ('main.pdf','supplement.pdf'):
                remote=str(action['/F'])[:-4]
                if isinstance(dest,str):assert dest in readers[remote].named_destinations
            links+=1
math=json.loads((folder/'periodic_finite.json').read_text())
row=next(r for r in math['rows'] if r['context']=='band' and r['B']==8192 and r['prior']=='11/20')
assert row['relative_gain_lower']>.0115 and math['word_count']==4368
report=dict(status='PASS_ESSENTIAL_TECHNICAL_CHECKS_NOT_A_VENUE_CERTIFICATE',
    pages={n:len(r.pages) for n,r in readers.items()},all_final_pages_visually_reviewed=True,
    abstract_words=len(abstract.split()),original_main_formal_blocks_preserved=len(formal(main)),
    references_fresh=50,raw_pass=47,explicit_resolutions=3,ieee_style_definite_errors=0,
    checked_links=links,new_native_trajectories=0,
    final_pdf_sha256={n:sha(paper/(n+'.pdf')) for n in readers},
    source_sha256={str(p.relative_to(paper)):sha(p) for p in sorted(seen)})
with target.open('x',encoding='utf-8') as out:json.dump(report,out,indent=2)
print(json.dumps(report['pages']))
