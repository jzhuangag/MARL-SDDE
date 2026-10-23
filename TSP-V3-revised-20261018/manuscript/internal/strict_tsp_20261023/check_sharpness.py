from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import time
from scipy.integrate import quad

folder=Path(__file__).resolve().parent
target=folder/'sharpness_checks.json';assert not target.exists()
theta=F(1,3);rho=F(1,5);r2=F(49,50)
v=[F(1,q)+theta*(1+rho)/(1-rho) for q in (2,8)]
rates=(8,5)
assert v[0]/rates[0]==v[1]/rates[1]==F(1,8)
t=2*theta*rho/(1-rho)**2
second=[(r2-2*v[i]-t)/rates[i]**2 for i in (0,1)]
a=second[1]-second[0]
s2=(theta*(1-rho))**2/(2*(theta*(1-rho)+F(1,8)))
b=(rho/(1-rho))**2*(F(1,25)-F(1,64))*s2
cut=1-a/b
assert a==F(19,320000) and s2==F(64,705) and b==F(13,94000)
assert cut==F(1187,2080) and 0<a<b
coefficient=quad(lambda g:2*float(b)*(float(cut)-g*g)*math.exp(-g*g/2)/math.sqrt(2*math.pi),
                 0,math.sqrt(float(cut)),epsabs=1e-14)[0]
rows=[]
for B in (32,128,512,2048):
    risks=[];ks=[]
    for i,q in enumerate((2,8)):
        N=rates[i]*B
        risks.append((r2+N*float(v[i])-float(t)*(1-float(rho)**N))/(N+1)**2)
        ks.append(float(rho)*(1-float(rho)**N)/((N+1)*(1-float(rho))))
    gap=float(risks[1]-risks[0]);term=(ks[1]**2-ks[0]**2)*float(s2)
    assert gap>0 and gap<term
    value=quad(lambda g:2*max(0.,term-gap-term*g*g)*math.exp(-g*g/2)/math.sqrt(2*math.pi),
               0,math.sqrt(1-gap/term),epsabs=1e-14)[0]
    rows.append(dict(B=B,scaled_value=B*B*value))
assert abs(rows[-1]['scaled_value']/coefficient-1)<.005
record=dict(status='PASS',leading_risk='1/8',second_order_gap=str(a),state_variance=str(s2),
    state_coefficient=str(b),cutoff_square=str(cut),positive_limit_approx=coefficient,
    finite_budget_diagnostic=rows,exact_fractions_establish_positive_limit=True,
    numerical_quadrature_not_used_as_proof=True,new_samples_seeds=0,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    cpu_snapshot=time.process_time())
with target.open('x',encoding='utf-8') as out:json.dump(record,out,indent=2)
print(json.dumps(record))
