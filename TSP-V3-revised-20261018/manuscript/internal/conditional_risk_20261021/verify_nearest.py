"""Fresh public metadata/full-text snapshot, no private manuscript upload."""
import hashlib
import json
from pathlib import Path
import time
START=time.process_time()
from datetime import datetime,timezone
import urllib.request
import urllib.error
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'nearest_source_records.json'
if OUTPUT.exists():raise RuntimeError('Preserve existing lookup attempt')
def retrieve(url,timeout=12):
    request=urllib.request.Request(url,headers={'User-Agent':'V3-research-reference-verification/1.0'})
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            return response.status,response.read()
    except urllib.error.HTTPError as exc:
        return exc.code,exc.read()
records=[dict(key='moustakides2012joint',doi='10.1109/TIT.2012.2184260',
              html='https://arxiv.org/html/1101.5084',arxiv='1101.5084',
              locators=['Section II, Eq.(3)-(9), Lemma 1 proof','Section II-A, Theorem 1'],
              version_scope='2011 preprint v1; published 2012 metadata verified separately'),
         dict(key='vasconcelos2020observation',doi='10.1109/TCNS.2019.2900864',
              html='https://arxiv.org/html/1804.01709',arxiv='1804.01709',
              locators=['Section II, Problem 1','Section IV, Lemma 3 and proof, Eq.(19)-(23)',
                        'Section IV-A, Eq.(35)-(37)'],
              version_scope='2018 preprint v1, sources title; 2020 journal metadata has random variables title')]
rate_limited=set()
for row in records:
    row['checked_utc']=datetime.now(timezone.utc).isoformat()
    row['resolvers']={}
    endpoints={'crossref':'https://api.crossref.org/works/'+row['doi'],
               'openalex':'https://api.openalex.org/works/https://doi.org/'+row['doi'],
               'semantic_scholar':'https://api.semanticscholar.org/graph/v1/paper/DOI:'+row['doi']+'?fields=title,authors,year,venue,externalIds'}
    for name,url in endpoints.items():
        if name in rate_limited:
            row['resolvers'][name]=dict(status='NOT_RETRIED_AFTER_RATE_LIMIT');continue
        try:
            status,content=retrieve(url,timeout=12)
            entry=dict(http_status=status)
            if status==429:
                rate_limited.add(name);entry['status']='RATE_LIMIT_PRESERVED_NO_RETRY'
            elif status==200:
                entry['status']='RETRIEVED_METADATA_REQUIRES_COMPARISON'
                data=json.loads(content);entry['data']=data.get('message',data)
            else:entry['status']='UNAVAILABLE'
        except Exception as exc:entry=dict(status='UNAVAILABLE',error=str(exc))
        row['resolvers'][name]=entry
    try:
        status,content=retrieve(row['html'],timeout=15)
        row['full_text_snapshot']=dict(http_status=status,
              sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),
              locators=row['locators'],read_scope='Listed technical passages read through web tool; not every unrelated section')
        if status==200:
            # A public source reading cache is generated data, not manuscript prose.
            with (HERE/(row['key']+'.source.html')).open('xb') as stream:stream.write(content)
    except Exception as exc:row['full_text_snapshot']=dict(status='UNAVAILABLE',error=str(exc))
result=dict(records=records,raw_resolver_outcomes_preserved=True,
            private_material_uploaded=False,cpu_seconds=time.process_time()-START)
with OUTPUT.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(dict(records=len(records),cpu_seconds=result['cpu_seconds'])))
