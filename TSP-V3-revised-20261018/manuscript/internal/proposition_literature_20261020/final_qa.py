"""Final PDF links/fonts/renders and additive ordinary-work resource record."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import pypdf

HERE=Path(__file__).resolve().parent;PAPER=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
readers={n:pypdf.PdfReader(PAPER/(n+'.pdf')) for n in ('main','supplement')}
assert len(readers['main'].pages)==13 and len(readers['supplement'].pages)==12
links=[];bad=[];fonts={}
for name,reader in readers.items():
    dests=reader.named_destinations
    for number,page in enumerate(reader.pages,1):
        for ref in page.get('/Annots',[]):
            ann=ref.get_object();act=ann.get('/A')
            if act:
                act=act.get_object();kind=str(act.get('/S'))
                target=act.get('/D')
                if kind=='/GoTo' and isinstance(target,str) and target not in dests:
                    bad.append((name,number,kind,target))
                if kind=='/GoToR' and str(act.get('/F',''))=='main.pdf' and isinstance(target,str) and target not in readers['main'].named_destinations:
                    bad.append((name,number,kind,target))
                if kind=='/URI':
                    uri=unquote(str(act.get('/URI','')))
                    if uri.startswith('main.pdf#'):
                        target=uri.split('#',1)[1]
                        if target not in readers['main'].named_destinations:bad.append((name,number,kind,target))
                links.append((name,number,kind))
    output=subprocess.check_output(['pdffonts',str(PAPER/(name+'.pdf'))],text=True)
    lines=[x for x in output.splitlines()[2:] if x.strip()]
    assert all(re.search(r'\s+yes\s+yes\s+(yes|no)\s+\d+\s+\d+\s*$',x) for x in lines)
    fonts[name]=dict(count=len(lines),all_embedded=True,raw=output)
assert not bad,bad
for stem,count in (('main',13),('supp',12)):
    files=sorted((HERE/'render').glob(stem+'-*.png'));assert len(files)==count
    assert all(p.stat().st_mtime>(PAPER/('main.pdf' if stem=='main' else 'supplement.pdf')).stat().st_mtime for p in files)
for name in ('main','supplement'):
    log=(PAPER/(name+'.log')).read_text(errors='replace')
    assert not re.search(r'Overfull|undefined|destination with the same identifier|has been referenced but does not exist',log,re.I)
costs=[json.loads(p.read_text()) for p in HERE.glob('*.cpu.json')]
measured=sum(c['child_cpu_seconds']+c['helper_cpu_seconds'] for c in costs)
assert all(c['exit_code']==0 for c in costs)
report=dict(pages=dict(main=13,supplement=12),all_25_pages_visually_reviewed=True,
    visual_method='Each of13paired contact sheets inspected; critical new formula/proof/table pages included; no clipping/overlap/unreadable figures found.',
    fonts=fonts,checked_links=len(links),invalid_destinations=bad,
    render_sha256={p.name:sha(p) for p in sorted((HERE/'render').glob('*.png'))},
    final_pdf_sha256={n:sha(PAPER/(n+'.pdf')) for n in readers},
    source_gate_sha256=sha(HERE/'source_checks.json'),
    resource=dict(measured_paper_child_and_meter_cpu_seconds=measured,metered_attempts=len(costs),
        all_metered_exit_codes_zero=True,pre_meter_retrieval_shell_and_contact_helpers_not_measured=True,
        conservative_unmetered_auxiliary_charge_seconds=120,
        ordinary_paper_phase_total_charge_seconds=measured+120,
        NOT_native_CPU_measurement=True,
        scope='Ordinary manuscript/literature operation; does not rewrite, reallocate or extend either immutable native scientific budget.',
        science_140h_terminal_cpu_hours=84.8294140625,
        v3_native_plus_independent_audit_cpu_seconds=2721.78125,
        earlier_v3_paper_integration_combined_charge_seconds=3412.265625,
        new_trajectories=0,new_seeds=0,new_bootstraps=0),
    reading_cache_hashes={p.name:sha(p) for p in sorted((HERE/'literature').glob('*.pdf'))},
    third_party_pdfs_local_only_not_redistributed=True)
with (HERE/'final_qa.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k not in ('fonts','render_sha256','reading_cache_hashes')}))
