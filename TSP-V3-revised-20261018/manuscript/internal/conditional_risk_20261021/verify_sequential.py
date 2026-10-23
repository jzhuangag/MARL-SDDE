"""Fresh indexed metadata and primary snapshots for the sequential-JDE comparison."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'sequential_source_records.json'
rows=[dict(key='reinhard2024asymptotic',doi='10.1016/j.sigpro.2024.109410',arxiv='2105.04828v3',html='https://arxiv.org/html/2105.04828v3',locators=['Section2.1 A1--A6','Sections2.2--2.4 Eq.(2)--(6)','Section3 Theorems1,2 and their displayed proof']),dict(key='reinhard2026overview',arxiv='2608.26278v1',html='https://arxiv.org/html/2608.26278',locators=['Sections2.2--2.3 Eq.(6)--(8),P1','Section3.3 Eq.(28)--(29)','Sections4.1--4.3'])]

def run():
    if OUTPUT.exists():raise RuntimeError('Preserve prior lookup')
    start=time.process_time();stopped=set()
    for row in rows:
        row['checked_utc']=datetime.now(timezone.utc).isoformat();row['resolvers']={}
        identifier=row.get('doi','10.48550/arXiv.'+row['arxiv'].split('v')[0])
        urls={'primary_arxiv':row['html'],'openalex':'https://api.openalex.org/works/https://doi.org/'+identifier,'semantic_scholar':'https://api.semanticscholar.org/graph/v1/paper/'+('DOI:'+row['doi'] if row.get('doi') else 'ARXIV:'+row['arxiv'].split('v')[0])+'?fields=title,authors,year,venue,externalIds'}
        if row.get('doi'):urls['crossref']='https://api.crossref.org/works/'+row['doi']
        for name,url in urls.items():
            if name in stopped:row['resolvers'][name]=dict(status='NOT_RETRIED_AFTER_UNAVAILABILITY');continue
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'Academic-integrity-source-check/1.0'})
                with urllib.request.urlopen(req,timeout=15) as response:
                    raw=response.read();entry=dict(status='RETRIEVED_REQUIRES_COMPARISON',http_status=response.status,url=response.geturl(),sha256=hashlib.sha256(raw).hexdigest())
                    if name!='primary_arxiv':
                        obj=json.loads(raw);entry['metadata']=obj.get('message',obj)
                    row['resolvers'][name]=entry
            except Exception as exc:
                row['resolvers'][name]=dict(status='UNAVAILABLE_NO_RETRY',url=url,error=repr(exc),http_status=getattr(exc,'code',None))
                if name in ('openalex','semantic_scholar'):stopped.add(name)
    with OUTPUT.open('x',encoding='utf-8') as out:json.dump(dict(records=rows,cpu_seconds=time.process_time()-start,raw_failures_preserved=True,private_material_uploaded=False),out,indent=2)
    print(json.dumps(dict(records=len(rows),cpu_seconds=time.process_time()-start)))

if __name__=='__main__':run()
