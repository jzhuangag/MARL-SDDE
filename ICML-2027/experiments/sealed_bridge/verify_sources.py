"""Minimal bibliographic metadata only; no unpublished content is uploaded."""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

SOURCES=[
 ('1806.05618','Stochastic Variance-Reduced Policy Gradient','https://proceedings.mlr.press/v80/papini18a.html','Snapshot importance-weighted gradient transport is established.'),
 ('1910.09066','From Importance Sampling to Doubly Robust Policy Gradient','https://proceedings.mlr.press/v119/huang20b.html','Differentiated off-policy value estimators and trajectory variance reduction are established.'),
 ('2111.11229','Off-Policy Correction For Multi-Agent Reinforcement Learning',None,'Joint-policy off-policy MARL correction is established; MA-Trace is required as a competitor.'),
 ('1912.00167','IMPACT: Importance Weighted Asynchronous Architectures with Clipped Target Networks',None,'Asynchronous clipped-target distributed policy optimization is established.'),
 ('2405.05630','Policy Gradient with Active Importance Sampling','https://rlj.org/papers/papini2024policy','Active/defensive sampling for lower policy-gradient variance is established.'),
 ('2602.17616','Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs',None,'ESS-dependent learning-rate and variance-controlled asynchronous policy updates are competitors.'),
 ('2603.01501','GAC: Stabilizing Asynchronous RL Training for LLMs via Gradient Alignment Control',None,'Stale-gradient projection is not a new direction by itself.'),
 ('2610.13692','Variance-Reduced Policy Gradient is Nearly Minimax-Optimal',None,'Recent defensive PAGE/STORM first-order theory is a close competitor, not our novelty.')]


class Metadata(HTMLParser):
    def __init__(self):super().__init__();self.meta={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='meta' and a.get('name','').startswith('citation_'):
            self.meta.setdefault(a['name'],[]).append(a.get('content',''))


def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'MARL-SDDE bibliography verification (metadata only)'})
    with urllib.request.urlopen(req,timeout=25) as r:return r.read().decode('utf-8')


def norm(x):return re.sub('[^a-z0-9]','',x.lower())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=[];s2_available=True
    for identifier,title,publisher,claim in SOURCES:
        url='https://arxiv.org/abs/'+identifier
        html=get(url);parser=Metadata();parser.feed(html)
        actual=parser.meta.get('citation_title',[''])[0]
        if norm(actual)!=norm(title):raise AssertionError((identifier,actual,title))
        authors=parser.meta.get('citation_author',[])
        if not authors:raise AssertionError('No authoritative author metadata')
        row=dict(arxiv_id=identifier,title=actual,authors=authors,
                 arxiv_date=parser.meta.get('citation_date',parser.meta.get('citation_online_date',[])),
                 authoritative_url=url,authoritative_existence='PASS',
                 authoritative_record_sha256=hashlib.sha256(html.encode()).hexdigest(),
                 publisher_url=publisher,claim_boundary=claim,
                 resolvers=dict(arxiv='PASS',crossref='NOT_APPLICABLE_ARXIV_DATACITE_IDENTIFIER',openalex='NOT_QUERIED'))
        if publisher:
            text=get(publisher)
            row['publisher_title_match']=norm(title) in norm(text)
            if not row['publisher_title_match']:raise AssertionError('Publisher title conflict')
        if s2_available:
            try:
                response=json.loads(get('https://api.semanticscholar.org/graph/v1/paper/ARXIV:'+identifier+'?fields=title,authors,year,venue,externalIds'))
                if norm(response['title'])!=norm(title):raise AssertionError('Semantic Scholar title conflict')
                row['semantic_scholar']=response;row['resolvers']['semantic_scholar']='PASS'
            except (urllib.error.URLError,TimeoutError) as e:
                # Do not route around rate limits using a browser or pretend a match.
                row['resolvers']['semantic_scholar']='UNAVAILABLE_'+str(getattr(e,'code',type(e).__name__))
                s2_available=False
            time.sleep(1.1)
        else:row['resolvers']['semantic_scholar']='NOT_QUERIED_AFTER_SERVICE_UNAVAILABLE'
        rows.append(row)
    report=dict(context_date='2026-10-16',purpose='successor-design nearest-work/citation integrity',
                citation_existence_pass=True,material_conflicts=[],references=rows,
                novelty_verdict='NOT_ESTABLISHED; defensive IS and variance-reduced PG are existing bases',
                unpublished_content_uploaded=False)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(references=len(rows),authoritative_matches=len(rows),s2_available=s2_available)))


if __name__=='__main__':main()
