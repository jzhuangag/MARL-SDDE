from fractions import Fraction as F
from pathlib import Path
import json,time
here=Path(__file__).resolve().parent
source=(here.parent/'arbitrary_schedule_20261023/bound.py').read_text().split('rows=[]')[0]
ns={'__file__':str(here.parent/'arbitrary_schedule_20261023/bound.py')};exec(source,ns)
env=ns['envelope'];lower=ns['lower'];grid=(1,2,4,8)
l=F(175,19);c0=F(9,8);c1=F(4,5);delta=F(3,100)
assert l**(-17)*(1-c0/2+c0/(2*l))**(-34)<delta**4
assert l**17*(1+c1*(l-1)/4)**(-68)<delta**8
def upper(B,t,lam,j,td=False):
    vals=[]
    for q,b in ((8,4),(2,2)):
        N=min((16*B-288)//(8+q),(4*B-18)//b)-(0 if td else 8)
        rho=lam**b;v=F(1,q)+t*(1+rho)/(1-rho)
        if td:
            u=F(361,N*N)+4*v/N
            k=4*(rho/(N*(1-rho)))**2
        else:u=(1+N*v)/(N+1)**2;k=(rho/((N+1)*(1-rho)))**2
        vals.append((u,k))
    correct=vals[j][0];wrong=vals[1-j][0]
    return (1-delta)*correct+delta*max(correct,wrong)+t*sum(x[1] for x in vals)
def td_lower(B,theta,lam):
    m=200;beta=F(19,20);a=1-beta**(m+1)
    # N<=m bounded by the smallest initialization coefficient in this range.
    early=(beta*(1-beta**m)/(F(m,20)))**2
    maximum=F(16*B,9)
    # n=N-m; retain the first n observations at a uniform weight floor.
    points={F(1),maximum-m}
    points.update(F(16*B,8+q) for q in grid)
    points.update(F(4*B,b) for b in grid)
    points=[n for n in points if 1<=n<=maximum-m]
    values=[]
    for n in points:
        v=env(F(16*B)/n-8,[F(1,q) for q in grid])+theta*(1+2*env(F(4*B)/n,[lam**b for b in grid]))
        values.append(4*a*a*(n*v-2*theta*lam)/(n+m)**2)
    return min([early]+values)
rows=[]
for box,t0,t1,la,lb in [('old',F(99,2000),F(101,2000),F(39,200),F(41,200)),('wide',F(1,25),F(3,50),F(3,20),F(1,4))]:
    highlo,highhi=(F(99,50),F(101,50)) if box=='old' else (F(9,5),F(11,5))
    for B in (512,2048,8192,32768):
        for td in (False,True):
            gains=[]
            for pi in (F(9,20),F(11,20)):
                theta=(1-pi)*t0+pi*highlo
                L=td_lower(B,theta,la) if td else lower(B,theta,la)[0]
                U=(1-pi)*upper(B,t1,lb,0,td)+pi*upper(B,highhi,lb,1,td)
                gains.append(float(1-U/L))
            rows.append(dict(box=box,B=B,td=td,gain=min(gains)))
assert time.process_time()<120
with (here/'results.json').open('x') as f:json.dump(dict(rows=rows,cpu=time.process_time(),delta=str(delta)),f,indent=2)
print(rows)
