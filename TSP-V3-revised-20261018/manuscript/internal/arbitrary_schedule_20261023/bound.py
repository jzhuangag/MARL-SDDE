from fractions import Fraction as F
import json
from pathlib import Path
import time
here=Path(__file__).resolve().parent
old=here.parent/'strict_tsp_20261023'
data=json.loads((old/'periodic_finite.json').read_text())
grid=(1,2,4,8)
def envelope(x,values):
    if x>=8:return values[-1]
    assert x>=1
    for i in range(3):
        if x<=grid[i+1]:
            a,b=grid[i:i+2]
            return ((b-x)*values[i]+(x-a)*values[i+1])/(b-a)
def lower(B,theta,lam):
    # Relax all withheld updates: even an applied-only resource allowance
    # is enough for this stronger-than-required lower bound.
    maximum=F(16*B,9)
    points={F(1),maximum}
    points.update(F(16*B,8+q) for q in grid)
    points.update(F(4*B,b) for b in grid)
    points=sorted(x for x in points if 1<=x<=maximum)
    def value(N):
        private=envelope(F(16*B)/N-8,[F(1,q) for q in grid])
        adjacent=envelope(F(4*B)/N,[lam**b for b in grid])
        return (1+N*(private+theta*(1+2*adjacent))-2*theta*lam)/(N+1)**2
    choices=[(value(n),n) for n in points]
    L,N=min(choices)
    return min(F(1),L),N,[(str(n),str(v)) for v,n in choices]
rows=[]
for r in data['rows']:
    B=r['B'];pi=F(r['prior']);band=r['context']=='band'
    theta=(1-pi)*(F(99,2000) if band else F(1,20))+pi*(F(99,50) if band else F(2))
    lam=F(39,200) if band else F(1,5)
    L,N,points=lower(B,theta,lam);U=F(r['upper_fraction'])
    rows.append(dict(context=r['context'],B=B,prior=str(pi),lower=str(L),upper=str(U),
                     gain=str(1-U/L),gain_float=float(1-U/L),minimizer=str(N),points=points))
assert time.process_time()<120
result=dict(rows=rows,cpu_seconds=time.process_time(),new_trajectories=0)
with (here/'bound.json').open('x') as f:json.dump(result,f,indent=2)
print([(r['context'],r['B'],r['prior'],r['gain_float']) for r in rows])
