"""Public metadata checks for the two newly used classical tools, no Bib rewrite."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'classical_source_records.json'
records=[dict(key='sion1958minimax',doi='10.2140/pjm.1958.8.171',primary='https://msp.org/pjm/1958/8-1/p14.xhtml',title='On general minimax theorems',authors=['Maurice Sion'],year=1958,volume='8',issue='1',pages='171-176'),dict(key='harville1974contrasts',doi='10.1093/biomet/61.2.383',primary='https://academic.oup.com/biomet/article-abstract/61/2/383/268084',title='Bayesian inference for variance components using only error contrasts',authors=['David A. Harville'],year=1974,volume='61',issue='2',pages='383-385')]

def run():
    if OUTPUT.exists():raise RuntimeError('Preserve prior source check')
    start=time.process_time();stopped=set()
    for row in records:
        row['checked_utc']=datetime.now(timezone.utc).isoformat();row['resolvers']={}
        urls={'crossref':'https://api.crossref.org/works/'+row['doi'],'openalex':'https://api.openalex.org/works/https://doi.org/'+row['doi'],'semantic_scholar':'https://api.semanticscholar.org/graph/v1/paper/DOI:'+row['doi']+'?fields=title,authors,year,venue,externalIds','primary':row['primary']}
        for name,url in urls.items():
            if name in stopped:row['resolvers'][name]=dict(status='NOT_RETRIED_AFTER_UNAVAILABILITY');continue
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'Academic-source-integrity-check/1.0'})
                with urllib.request.urlopen(request,timeout=15) as response:
                    raw=response.read();entry=dict(status='RETRIEVED_REQUIRES_COMPARISON',http_status=response.status,url=response.geturl(),sha256=hashlib.sha256(raw).hexdigest())
                    if name!='primary':
                        data=json.loads(raw);entry['metadata']=data.get('message',data)
                    row['resolvers'][name]=entry
            except Exception as exc:
                row['resolvers'][name]=dict(status='UNAVAILABLE_NO_RETRY',url=url,error=repr(exc),http_status=getattr(exc,'code',None))
                if name in ('semantic_scholar','openalex'):stopped.add(name)
        row['arxiv']='NOT_APPLICABLE_TO_PUBLISHED_CLASSICAL_ARTICLE'
    report=dict(records=records,cpu_seconds=time.process_time()-start,private_material_uploaded=False,raw_failures_preserved=True)
    with OUTPUT.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(dict(records=len(records),cpu_seconds=report['cpu_seconds'])))

if __name__=='__main__':run()
