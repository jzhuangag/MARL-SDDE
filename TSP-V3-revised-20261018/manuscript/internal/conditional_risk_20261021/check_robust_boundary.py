"""New proof-scope check; earlier fixture attempt remains immutable."""
import json
import time
from pathlib import Path
START=time.process_time()
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'robust_boundary_checks.json'
if OUTPUT.exists():raise RuntimeError('Preserve existing fixture attempt')
rows=[]
for c in (0.,3.,4.5,4.8,5.,5.4,6.):
    V=7.;v=V-c*c/V;sigma=c/np.sqrt(V);r=1.
    formula=direct=0.
    if v<3 and c>0:
        t=np.sqrt(3-v)/sigma
        P=2*norm.cdf(t)-1
        formula=((3-v)*P-sigma*sigma*(P-2*t*norm.pdf(t)))/4
        direct=quad(lambda z:(3-v-sigma*sigma*z*z)*norm.pdf(z)/4,
                    -t,t,epsabs=1e-12)[0]
        # The endpoint error is shared across the whole distribution, not
        # adversarially selected anew at each observation.
        def risk(e):
            Pin=norm.cdf(t)-norm.cdf(-t)
            return e*e*(1-Pin)+quad(lambda z:((e+sigma*z)**2+v)*norm.pdf(z)/4,
                                    -t,t,epsabs=1e-12)[0]
        risks=[risk(e) for e in (-1.,-.3,0.,.3,1.)]
        assert max(risks)<=1-formula+1e-12
        assert abs(risks[0]-(1-formula))<1e-12
    else:risks=[1.,.09,0.,.09,1.]
    assert abs(formula-direct)<1e-11
    assert (formula>0)==(v<3 and c>0)
    rows.append(dict(c=c,conditional_variance=v,robust_gain=formula,
                     robust_risk=1-formula,known_sign_possible=v<4 and c>0,
                     error_values=[-1.,-.3,0.,.3,1.],policy_risks=risks))
result=dict(status='PASS',cases=len(rows),rows=rows,new_scientific_samples=0,
            random_number_calls=0,cpu_seconds=time.process_time()-START)
with OUTPUT.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(dict(status=result['status'],cases=len(rows),cpu_seconds=result['cpu_seconds'])))
