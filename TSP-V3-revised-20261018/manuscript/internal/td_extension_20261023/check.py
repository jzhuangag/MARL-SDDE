from pathlib import Path
from fractions import Fraction as F
import numpy as np,json,time
here=Path(__file__).resolve().parent;tests=0
for N in (1,2,8,32,128,513):
    for lam in (.15,.2,.25):
        for pattern in range(3):
            q=np.array([1,2,4,8]*((N+3)//4))[:N]
            b=np.roll(q,pattern);times=np.cumsum(b)
            theta=2.2;mu=.5;beta=.95
            C=theta*lam**np.abs(times[:,None]-times[None,:])+np.diag(1/q)
            weights=(1-beta**np.arange(N,0,-1))/(mu*N)
            exact=weights@C@weights
            row=1+theta*(1+lam)/(1-lam)
            floor=(C.sum()-38*row)/(mu*mu*N*N)
            assert exact>=floor-1e-12
            # Direct finite update sensitivity, not the closed-form formula.
            impulse=np.zeros(N);acc=np.zeros(N)
            for t in range(N):
                impulse*=beta;impulse[t]+=.1;acc+=impulse/N
            assert np.max(abs(weights-acc))<1e-12
            tests+=2
for name,floor in [('refined.json',.0254),('transfer.json',.0186)]:
    records=json.loads((here/name).read_text())['rows']
    row=next(r for r in records if r['B']==8192 and (name=='transfer.json' or not r['td']))
    assert F(row['exact_gain'])>F(str(floor));tests+=1
report=dict(status='PASS',checks=tests,cpu=time.process_time(),no_samples_or_seeds=True)
with (here/'checks.json').open('x') as f:json.dump(report,f,indent=2)
print(report)
