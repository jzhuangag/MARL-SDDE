from fractions import Fraction as F
from pathlib import Path
from itertools import product
import json,time
here=Path(__file__).resolve().parent
ns={'__file__':str(here/'calculate.py')};exec((here/'calculate.py').read_text().split('\nrows=[]\n')[0],ns)
env=ns['env'];upper=ns['upper'];grid=(1,2,4,8)
def lower(B,pi,t0,t1,la,lb,td):
    m=200 if td else 0;cut=512
    maximum=F(16*B,9);minimum=F(cut+1-m) if td else F(1)
    points={minimum,maximum-m}
    points.update(F(16*B,8+q) for q in grid)
    points.update(F(4*B,b) for b in grid)
    points=[n for n in points if minimum<=n<=maximum-m]
    vals=[]
    for n in points:
        v=env(F(16*B)/n-8,[F(1,q) for q in grid])
        for p,t,lam in ((1-pi,t0,la),(pi,t1,lb)):
            v+=p*t*(1+2*env(F(4*B)/n,[lam**b for b in grid]))
        boundary=2*((1-pi)*t0*la+pi*t1*lb)
        if td:val=4*(1-F(19,20)**201)**2*(n*v-boundary)/(n+m)**2
        else:val=(1+n*v-boundary)/(n+1)**2
        vals.append(val)
    early=(F(19,20)*(1-F(19,20)**cut)/(F(cut,20)))**2 if td else F(1)
    return min([early]+vals)
rows=[]
for B in (8192,32768):
    for td in (False,True):
        best=None
        # Entire box certified by concavity in each theta and prior;
        # only persistence is interval-partitioned, not sampled.
        for i,j in product(range(16),repeat=2):
            la=F(3,20)+F(i,160);lb=F(3,20)+F(j,160)
            for t0,t1,pi in product((F(1,25),F(3,50)),(F(9,5),F(11,5)),(F(9,20),F(11,20))):
                L=lower(B,pi,t0,t1,la,lb,td)
                U=(1-pi)*upper(B,t0,la+F(1,160),0,td)+pi*upper(B,t1,lb+F(1,160),1,td)
                ratio=1-U/L
                if best is None or ratio<best[0]:best=(ratio,(i,j,str(t0),str(t1),str(pi)),L,U)
        rows.append(dict(B=B,td=td,gain=float(best[0]),exact_gain=str(best[0]),worst=best[1],L=str(best[2]),U=str(best[3])))
        print(rows[-1],flush=True)
        assert time.process_time()<120
with (here/'refined.json').open('x') as f:json.dump(dict(rows=rows,cpu=time.process_time()),f,indent=2)
