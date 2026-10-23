from fractions import Fraction as F
from itertools import product
from pathlib import Path
import importlib.util,json,time
here=Path(__file__).resolve().parent
# Load function definitions without rerunning or overwriting the bound report.
source=(here/'bound.py').read_text().split('rows=[]')[0]
ns={'__file__':str(here/'bound.py')};exec(source,ns)
envelope=ns['envelope'];lower=ns['lower'];grid=(1,2,4,8)
tests=0
for lam in (F(0),F(39,200),F(1,5),F(41,200)):
    for N in range(1,7):
        for word in product(grid,repeat=N):
            avg=F(sum(word),N)
            assert sum(F(1,q) for q in word)>=N*envelope(avg,[F(1,q) for q in grid])
            assert sum(lam**b for b in word)>=N*envelope(avg,[lam**b for b in grid])
            times=[0]
            for b in word[1:]:times.append(times[-1]+b)
            dense=sum(lam**(times[t]-times[s]) for t in range(N) for s in range(t))
            assert dense>=sum(lam**b for b in word)-lam
            tests+=1
# Check minimization reduction by every feasible integer prefix at all five
# scales, all three model/prior configurations; this is not a scientific study.
rows=json.loads((here/'bound.json').read_text())['rows']
for row in rows:
    B=row['B'];pi=F(row['prior']);band=row['context']=='band'
    theta=(1-pi)*(F(99,2000) if band else F(1,20))+pi*(F(99,50) if band else F(2))
    lam=F(39,200) if band else F(1,5);L=F(row['lower'])
    for N in range(1,16*B//9+1):
        H=(1+N*(envelope(F(16*B,N)-8,[F(1,q) for q in grid])+theta*(1+2*envelope(F(4*B,N),[lam**b for b in grid])))-2*theta*lam)/(N+1)**2
        assert H>=L
        tests+=1
assert time.process_time()<120
out=dict(status='PASS',deterministic_checks=tests,all_scales_preserved=True,
         no_trajectories_or_seeds=True,cpu_seconds=time.process_time())
with (here/'checks.json').open('x') as f:json.dump(out,f,indent=2)
print(out)
