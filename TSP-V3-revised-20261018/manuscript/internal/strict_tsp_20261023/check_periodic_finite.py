from fractions import Fraction as F
from itertools import product
import hashlib
import json
from pathlib import Path
import time

folder=Path(__file__).resolve().parent;output=folder/'periodic_finite.json'
assert not output.exists()
reference=json.loads((folder/'analytic_checks.json').read_text(encoding='utf-8'))['witness']['rows']
band=json.loads((folder/'robust_band_corrected02.json').read_text(encoding='utf-8'))['robust_band']['rows']
actions=tuple(product((1,2,4,8),repeat=2));scales=(32,128,512,2048,8192)

def words(lam):
    out=[]
    for length in (1,2,3):
        for word in product(actions,repeat=length):
            rs=[lam**b for q,b in word];A=F(1);z=F(0)
            for rho in rs:A*=rho;z=rho*(1+z)
            z0=z/(1-A);z=z0;zs=[]
            for rho in rs:z=rho*(1+z);zs.append(z)
            assert z==z0
            private=sum(F(1,q) for q,b in word)/length
            common=1+2*sum(zs)/length
            pm=[0];pe=[0];dp=[F(0)];dc=[F(0)]
            for i,(q,b) in enumerate(word):
                pm.append(pm[-1]+8+q);pe.append(pe[-1]+b)
                dp.append(dp[-1]+F(1,q)-private)
                dc.append(dc[-1]+2*zs[i]-(common-1))
            out.append((word,pm,pe,private,common,dp,dc,2*zs[0]/(1-max(rs))))
    assert len(out)==4368
    return out

def minimum_lower(B,theta,prepared):
    best=None;early=0
    for word,pm,pe,private,common,dp,dc,transient in prepared:
        length=len(word)
        M=max(length*min((16*B-pm[s])//pm[-1],(4*B-pe[s])//pe[-1])+s
              for s in range(length))
        maximum=max(0,M-8)
        v=private+theta*common
        C=theta*transient-min(dp[s]+theta*dc[s] for s in range(length))
        if maximum==0:bound=F(1)
        else:
            first=(1+v-C)/4
            last=(1+maximum*v-C)/(maximum+1)**2
            bound=min(F(1),first,last)
            if first<=0:
                # For N>=n0, Nv-C>=Nv/2. For earlier N, initialization suffices.
                n0=max(1,-(-2*C.numerator*v.denominator//(C.denominator*v.numerator)))
                if maximum<n0:bound=F(1,(maximum+1)**2)
                else:bound=min(F(1,(n0+1)**2),(1+n0*v-C)/(n0+1)**2,last)
                early+=1
        if best is None or bound<best[0]:best=(bound,word,maximum)
    return best,early

rows=[];contexts=[('reference',F(1,5),((F(1,2),F(41,40)),)),
                 ('band',F(39,200),tuple((pi,(1-pi)*F(99,2000)+pi*F(99,50)) for pi in (F(9,20),F(11,20))))]
for label,lam,priors in contexts:
    prepared=words(lam)
    for index,B in enumerate(scales):
        for pi,theta in priors:
            (L,word,N),early=minimum_lower(B,theta,prepared)
            if label=='reference':U=F(reference[index]['paid_risk_upper_fraction'])
            else:U=F(next(x['upper_fraction'] for x in band[index]['prior_endpoints'] if x['prior']==str(pi)))
            rows.append(dict(context=label,B=B,prior=str(pi),all_words=4368,
                lower_fraction=str(L),upper_fraction=str(U),relative_gain_lower_fraction=str(1-U/L),
                relative_gain_lower=float(1-U/L),strict_positive=L>U,
                minimizing_lower_bound_word=word,max_applied=N,early_prefix_bound_cases=early))
    assert time.process_time()<110,'Preserve partial work; no silent budget expansion'

# Verify the periodic finite variance identity on all short words, every prefix.
checks=0
for word,pm,pe,private,common,dp,dc,transient in words(F(1,5))[:272]:
    theta=F(41,40);v=private+theta*common
    C=theta*transient-min(dp[s]+theta*dc[s] for s in range(len(word)))
    z=F(0);running=F(0)
    for N in range(1,13):
        q,b=word[(N-1)%len(word)]
        z=F(0) if N==1 else F(1,5)**b*(1+z)
        running+=F(1,q)+theta*(1+2*z)
        assert running>=N*v-C
        checks+=1
record=dict(status='EXACT_RATIONAL_FINITE_PERIODIC_PREFIX_CERTIFICATES',rows=rows,
    word_count=4368,all_five_budget_scales=True,finite_variance_identity_checks=checks,
    no_arbitrary_period_or_adaptive_estimator_optimality=True,no_new_samples_or_seeds=True,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cpu_snapshot=time.process_time())
with output.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k!='rows'}))
print(json.dumps([{k:r[k] for k in ('context','B','prior','relative_gain_lower','minimizing_lower_bound_word')} for r in rows]))
