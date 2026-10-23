"""Final PDF links/fonts/renders and additive ordinary-work resource record."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote
import pypdf

HERE=Path(__file__).resolve().parent;PAPER=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
readers={n:pypdf.PdfReader(PAPER/(n+'.pdf')) for n in ('main','supplement')}
assert sys.argv[1:]==['--all-pages-visually-reviewed'],'Manual visual review required'
counts={name:len(reader.pages) for name,reader in readers.items()}
assert counts['main']==13
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
                if kind=='/GoToR' and str(act.get('/F','')) in ('main.pdf','supplement.pdf') and isinstance(target,str):
                    remote=str(act.get('/F'))[:-4]
                    if target not in readers[remote].named_destinations:bad.append((name,number,kind,target))
                if kind=='/URI':
                    uri=unquote(str(act.get('/URI','')))
                    for remote in readers:
                        if uri.startswith(remote+'.pdf#'):
                            target=uri.split('#',1)[1]
                            if target not in readers[remote].named_destinations:bad.append((name,number,kind,target))
                links.append((name,number,kind))
    output=subprocess.check_output(['pdffonts',str(PAPER/(name+'.pdf'))],text=True)
    lines=[x for x in output.splitlines()[2:] if x.strip()]
    assert all(re.search(r'\s+yes\s+yes\s+(yes|no)\s+\d+\s+\d+\s*$',x) for x in lines)
    fonts[name]=dict(count=len(lines),all_embedded=True,raw=output)
assert not bad,bad
for stem,count in (('main',counts['main']),('supp',counts['supplement'])):
    files=sorted((HERE/'render_final3').glob(stem+'-*.png'));assert len(files)==count
    assert all(p.stat().st_mtime>(PAPER/('main.pdf' if stem=='main' else 'supplement.pdf')).stat().st_mtime for p in files)
for name in ('main','supplement'):
    log=(PAPER/(name+'.log')).read_text(errors='replace')
    assert not re.search(r'Overfull|undefined|destination with the same identifier|has been referenced but does not exist',log,re.I)
costs=[json.loads(p.read_text()) for p in HERE.glob('*.cpu.json')]
measured=sum(c['child_cpu_seconds']+c['helper_cpu_seconds'] for c in costs)
report=dict(pages=counts,all_final_pages_visually_reviewed=True,
    visual_method='Every final page visually inspected, with individual-page review of dense new mathematics and tables; no clipping/overlap/unreadable figures accepted.',
    fonts=fonts,checked_links=len(links),invalid_destinations=bad,
    render_sha256={p.name:sha(p) for p in sorted((HERE/'render_final3').glob('*.png'))},
    final_pdf_sha256={n:sha(PAPER/(n+'.pdf')) for n in readers},
    source_gate_sha256=sha(HERE/'final_source_gate_after_visual.json'),
    resource=dict(measured_paper_child_and_meter_cpu_seconds=measured,metered_attempts=len(costs),
        preserved_failed_attempts=[c['label'] for c in costs if c['exit_code']!=0],pre_meter_retrieval_shell_and_contact_helpers_not_measured=True,
        conservative_unmetered_auxiliary_charge_seconds=120,
        ordinary_paper_phase_total_charge_seconds=measured+120,
        NOT_native_CPU_measurement=True,
        scope='Ordinary manuscript/literature operation; does not rewrite, reallocate or extend either immutable native scientific budget.',
        science_140h_terminal_cpu_hours=84.8294140625,
        v3_native_plus_independent_audit_cpu_seconds=2721.78125,
        earlier_v3_paper_integration_combined_charge_seconds=3412.265625,
        new_trajectories_generated_by_paper_tools=0,new_seeds_generated_by_paper_tools=0,new_bootstraps_by_paper_tools=0),
    third_party_pdfs_local_only_not_redistributed=True)
with (HERE/'final_qa.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k not in ('fonts','render_sha256','reading_cache_hashes')}))
