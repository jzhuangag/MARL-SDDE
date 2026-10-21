"""Deterministic population-algebra checks; no trajectories, seeds or sampling."""
import hashlib
import json
import math
from pathlib import Path
import time
HERE=Path(__file__).resolve().parent

def risk(N,q,rho,theta,r):
    if N==0:return r*r
    v=1/q+theta*(1+rho)/(1-rho)
    return (r*r+N*v-2*theta*rho*(1-rho**N)/(1-rho)**2)/(N+1)**2

def run():
    checks=0;prefixes=0;largest=0;rounding=0
    for q in (1,2,8):
        for rho in (0.,.2,.8):
            for theta in (.05,2.):
                for r in (.5,1.):
                    v=1/q+theta*(1+rho)/(1-rho)
                    c=r*r+2*v+2*theta*rho/(1-rho)**2
                    n0=math.ceil(2*c/v)
                    for maximum in (math.ceil(v*(n0+1)**2/(r*r)),math.ceil(2*v*(n0+1)**2/(r*r))):
                        lower=v/maximum-16*c/maximum**2
                        largest=max(largest,maximum)
                        for N in range(maximum+1):
                            value=risk(N,q,rho,theta,r)
                            assert value>=lower-1e-12
                            if N:assert abs(value-v/N)<=c/N**2+1e-12
                            prefixes+=1
                        checks+=1
    for q in (1,2,4,8):
        for b in (1,2,4,8):
            a=min(16/(8+q),4/b);rho=.2**b
            for D in (0,8):
                for pm,pe in ((0,0),(640,40)):
                    L=D+1+max((1+pm)/(8+q),(1+pe)/b)
                    for B in (math.ceil(2*L/a),math.ceil(4*L/a)+1):
                        N=max(0,min((math.floor(16*B)-pm)//(8+q),(math.floor(4*B)-pe)//b)-D)
                        assert a*B-L-1e-12<=N<=a*B and N>=a*B/2-1e-12
                        for theta in (.05,2.):
                            v=1/q+theta*(1+rho)/(1-rho);c=1+2*v+2*theta*rho/(1-rho)**2
                            C=(4*c+2*v*L)/a**2
                            assert abs(risk(N,q,rho,theta,1)-v/(a*B))<=C/B**2+1e-12
                            rounding+=1
    report=dict(status='PASS',finite_prefix_models=checks,all_checked_prefixes=prefixes,
        largest_checked_full_prefix=largest,paid_rounding_remainder_checks=rounding,
        no_random_draws_or_seeds=True,new_scientific_trajectories=0,
        checks_support_not_replace_displayed_proof=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_snapshot_seconds=time.process_time())
    with (HERE/'explicit_threshold_checks.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report))

if __name__=='__main__':run()
