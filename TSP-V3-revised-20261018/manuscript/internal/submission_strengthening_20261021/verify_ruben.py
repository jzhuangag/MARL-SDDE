"""Fresh public identifier checks; source PDF, if public, stays in local cache."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
HERE=Path(__file__).resolve().parent

def run():
    output=HERE/'ruben_source_records.json';assert not output.exists()
    doi='10.1214/aoms/1177704580'
    urls=dict(crossref='https://api.crossref.org/works/'+doi,
        openalex='https://api.openalex.org/works/https://doi.org/'+doi,
        semantic_scholar='https://api.semanticscholar.org/graph/v1/paper/DOI:'+doi+'?fields=title,authors,year,venue,externalIds',
        primary='https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2Faoms%2F1177704580')
    records={}
    for key,url in urls.items():
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'Academic-source-integrity-check/1.0'})
            with urllib.request.urlopen(request,timeout=15) as response:
                raw=response.read();row=dict(status='RETRIEVED_REQUIRES_COMPARISON',url=response.geturl(),
                    sha256=hashlib.sha256(raw).hexdigest(),http_status=response.status)
                if key!='primary':
                    value=json.loads(raw);row['metadata']=value.get('message',value)
                elif raw.startswith(b'%PDF'):
                    cache=Path('C:/Users/Administrator/.cache/v3_source_review_20261021/ruben1962.pdf')
                    assert not cache.exists()
                    with cache.open('xb') as stream:stream.write(raw)
                    row['local_primary_pdf']=str(cache);row['redistribution']=False
                else:row['primary_text_excerpt']=raw[:1000].decode('utf-8',errors='replace')
                records[key]=row
        except Exception as exc:records[key]=dict(status='UNAVAILABLE_NO_RETRY',url=url,error=repr(exc),http_status=getattr(exc,'code',None))
    report=dict(doi=doi,checked_utc=datetime.now(timezone.utc).isoformat(),records=records,
        arxiv='NOT_APPLICABLE',raw_web_tool_errors_preserved_in_conversation=True,
        positive_metadata_match_requires_manual_comparison=True,cpu_snapshot=time.process_time())
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({name:row['status'] for name,row in records.items()}))

if __name__=='__main__':run()
