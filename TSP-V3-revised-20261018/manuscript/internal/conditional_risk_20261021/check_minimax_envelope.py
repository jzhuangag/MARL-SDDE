"""Exhaustive finite symmetric mathematical fixture; no stochastic samples."""
import itertools
import json
from pathlib import Path
import time
START=time.process_time()
import numpy as np
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'minimax_envelope_checks.json'
if OUTPUT.exists():raise RuntimeError('Preserve existing fixture attempt')
y=np.array([-.3,-1.,-2.,.3,1.,2.])
d=np.array([1.,.5,.2]);k=np.array([0.,.6,.3]);v=np.array([0.,.2,.05])
rows=[]
for radius in (.2,1.,2.):
    even_cost=radius*radius*d[None,:]**2+y[:,None]**2*k[None,:]**2+v[None,:]
    candidate=np.argmin(even_cost,axis=1)
    bound=float(np.min(even_cost,axis=1).mean())
    best=float('inf')
    for selector in itertools.product(range(3),repeat=6):
        a=np.array(selector)
        endpoint_risks=[float(np.mean((d[a]*e+k[a]*y)**2+v[a])) for e in (-radius,radius)]
        worst=max(endpoint_risks)
        # Lower bound applies even to sign-asymmetric selectors.
        assert worst>=bound-1e-13
        best=min(best,worst)
    actual=max(float(np.mean((d[candidate]*e+k[candidate]*y)**2+v[candidate]))
               for e in (-radius,radius))
    assert abs(best-bound)<1e-13 and abs(actual-bound)<1e-13
    assert np.array_equal(candidate[:3],candidate[3:])
    rows.append(dict(radius=radius,lower_envelope=bound,best_of_729_selectors=best,
                     robust_candidate_risk=actual))
result=dict(status='PASS',cases=len(rows),selectors_per_case=729,
            new_scientific_samples=0,random_number_calls=0,rows=rows,
            cpu_seconds=time.process_time()-START)
with OUTPUT.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps({k:result[k] for k in ('status','cases','cpu_seconds')}))
