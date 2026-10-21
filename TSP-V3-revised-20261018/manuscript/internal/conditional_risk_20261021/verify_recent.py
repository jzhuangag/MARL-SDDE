"""Preserve the late 2026 primary-source check; no scientific execution."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request
import urllib.error
START=time.process_time()
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'recent_source_records.json'
if OUTPUT.exists():raise RuntimeError('Preserve existing lookup attempt')
urls={
 'arxiv_primary':'https://arxiv.org/abs/2604.22740',
 'arxiv_full_text':'https://arxiv.org/html/2604.22740',
 'openalex':'https://api.openalex.org/works/https://doi.org/10.48550/arXiv.2604.22740',
 'semantic_scholar':'https://api.semanticscholar.org/graph/v1/paper/ARXIV:2604.22740?fields=title,authors,year,venue,externalIds',
 'crossref_2024_search_lead':'https://api.crossref.org/works/10.1016/j.sigpro.2024.109410'
}
records={}
for key,url in urls.items():
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'V3-reference-check/1.0'})
        with urllib.request.urlopen(req,timeout=12) as response:
            raw=response.read();entry=dict(status=response.status,url=url,
                    bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            if key in ('openalex','semantic_scholar','crossref_2024_search_lead'):
                entry['metadata']=json.loads(raw)
            else:
                with (HERE/(key+'.source.html')).open('xb') as stream:stream.write(raw)
            records[key]=entry
    except Exception as exc:records[key]=dict(url=url,status='UNAVAILABLE_NO_RETRY',error=str(exc))
result=dict(records=records,cpu_seconds=time.process_time()-START)
with OUTPUT.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(dict(records=len(records),cpu_seconds=result['cpu_seconds'])))
