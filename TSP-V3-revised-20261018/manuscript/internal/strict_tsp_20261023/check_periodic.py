from fractions import Fraction as F
from itertools import product
import hashlib
import json
from pathlib import Path
import time

folder=Path(__file__).resolve().parent;out=folder/'periodic_comparator.json'
assert not out.exists()
actions=tuple(product((1,2,4,8),repeat=2));theta=F(41,40);lam=F(1,5)
rows=[]
for length in (1,2,3):
    best=None;count=0
    for word in product(actions,repeat=length):
        rs=[lam**b for q,b in word];A=F(1);z=F(0)
        for rho in rs:A*=rho;z=rho*(1+z)
        z0=z/(1-A);z=z0;zs=[]
        for rho in rs:z=rho*(1+z);zs.append(z)
        assert z==z0
        private=sum(F(1,q) for q,b in word)/length
        v=private+theta*(1+2*sum(zs)/length)
        a=min(F(16)/(sum(8+q for q,b in word)/F(length)),
              F(4)/(sum(b for q,b in word)/F(length)))
        coefficient=v/a
        if best is None or coefficient<best[0]:best=(coefficient,word)
        count+=1
    rows.append(dict(period=length,all_words=count,minimum_coefficient=str(best[0]),
                     minimum_coefficient_float=float(best[0]),word=best[1]))
record=dict(status='EXACT_LEADING_COEFFICIENT_DIAGNOSTIC',rows=rows,
            all_periods_retained=True,not_all_nonadaptive_schedules=True,
            not_finite_budget_or_empirical_superiority=True,
            new_samples_seeds=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cpu_snapshot=time.process_time())
with out.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
