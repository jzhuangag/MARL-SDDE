"""Verify the actually read classical mixture source, not the blocked Ruben PDF."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
HERE=Path(__file__).resolve().parent

def run():
    output=HERE/'solomon_source_records.json';assert not output.exists()
    title='On the Distribution of Quadratic Forms in Normal Variates'
    query=urllib.parse.quote(title)
    urls=dict(crossref='https://api.crossref.org/works?query.title='+query+'&query.author=Solomon&rows=3',
        openalex='https://api.openalex.org/works?search='+query+'&per-page=5',
        semantic_scholar='https://api.semanticscholar.org/graph/v1/paper/search?query='+query+'&limit=3&fields=title,authors,year,venue,externalIds',
        primary_library='https://digicoll.lib.berkeley.edu/record/112911',
        primary_pdf='https://digicoll.lib.berkeley.edu/record/112911/files/math_s4_v1_article-32.pdf')
    records={}
    for key,url in urls.items():
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'Academic-source-integrity-check/1.0'})
            with urllib.request.urlopen(request,timeout=15) as response:
                raw=response.read();row=dict(status='RETRIEVED_REQUIRES_COMPARISON',url=response.geturl(),
                    sha256=hashlib.sha256(raw).hexdigest(),http_status=response.status)
                if key in ('crossref','openalex','semantic_scholar'):row['metadata']=json.loads(raw)
                elif key=='primary_pdf':
                    assert raw.startswith(b'%PDF')
                    cache=Path('C:/Users/Administrator/.cache/v3_source_review_20261021/solomon1961.pdf')
                    with cache.open('xb') as stream:stream.write(raw)
                    row['local_primary_pdf']=str(cache);row['redistribution']=False
                else:row['html_excerpt']=raw[:2500].decode('utf-8',errors='replace')
                records[key]=row
        except Exception as exc:records[key]=dict(status='UNAVAILABLE_NO_RETRY',url=url,error=repr(exc),http_status=getattr(exc,'code',None))
    report=dict(title=title,checked_utc=datetime.now(timezone.utc).isoformat(),records=records,
        arxiv='NOT_APPLICABLE',no_doi_invented=True,publication_year_needs_conference_date_distinction=True,
        complete_web_pdf_read_pages='645-653; mixture identity at646 Eq2, Laplace product at647 Eq3',
        source_formula_is_classical_not_a_new_V3_distribution_theorem=True,cpu_snapshot=time.process_time())
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({name:row['status'] for name,row in records.items()}))

if __name__=='__main__':run()
