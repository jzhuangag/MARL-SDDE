"""Essential independent dense-covariance check of the positive cyclic claim."""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import time

folder=Path(__file__).resolve().parent;target=folder/'periodic_dense_checks.json'
assert not target.exists()
def words():
    actions=tuple(product((1,2,4,8),repeat=2))
    for length in (1,2,3):
        for word in product(actions,repeat=length):
            rs=[F(1,5)**b for q,b in word];A=F(1);z=F(0)
            for rho in rs:A*=rho;z=rho*(1+z)
            z=z/(1-A);zs=[]
            for rho in rs:z=rho*(1+z);zs.append(z)
            private=sum(F(1,q) for q,b in word)/length
            common=1+2*sum(zs)/length
            pm=[0];pe=[0];dp=[F(0)];dc=[F(0)]
            for i,(q,b) in enumerate(word):
                pm.append(pm[-1]+8+q);pe.append(pe[-1]+b)
                dp.append(dp[-1]+F(1,q)-private)
                dc.append(dc[-1]+2*zs[i]-(common-1))
            yield word,pm,pe,private,common,dp,dc,2*zs[0]/(1-max(rs))
checks=0
for word,pm,pe,private,common,dp,dc,transient in words():
    theta=F(41,40);v=private+theta*common
    C=theta*transient-min(dp[s]+theta*dc[s] for s in range(len(word)))
    times=[];clock=0;noise=F(0)
    for i in range(7):
        q,b=word[i%len(word)];clock+=b;times.append(clock);noise+=F(1,q)
    dense=noise+theta*sum(F(1,5)**abs(x-y) for x in times for y in times)
    assert dense>=7*v-C
    m=e=M=0
    while True:
        q,b=word[M%len(word)]
        if m+8+q>512 or e+b>128:break
        m+=8+q;e+=b;M+=1
    formula=max(len(word)*min((512-pm[s])//pm[-1],(128-pe[s])//pe[-1])+s for s in range(len(word)))
    assert M==formula
    checks+=1
with target.open('x',encoding='utf-8') as out:
    json.dump(dict(status='PASS',all_periodic_words=checks,dense_covariance_and_independent_budget_loops=True,
                   no_samples_seeds=True,cpu_snapshot=time.process_time()),out,indent=2)
print(checks)
