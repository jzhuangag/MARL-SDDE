"""Independent characteristic-function check of the classical positive series."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.integrate import quad
from population_risk import mixture
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'experiments/v3_observable_gate1h'))
import model

def fourier(a,b,c):
    d=len(a);s2=np.dot(b,b);logC=-.5*np.log(2*a).sum();eps=1e-10
    T=math.exp((logC-math.log(math.pi*(d/2)*eps))/(d/2))
    tail=math.exp(logC)/(math.pi*(d/2)*T**(d/2))
    mtail=math.exp(logC)*s2/(2*min(a)*math.pi*(d/2+1)*T**(d/2+1))
    def integrand(t,moment):
        if abs(t)<1e-14:
            return ((s2*np.sum(a)+2*np.dot(b*b,a)-c*s2) if moment else (np.sum(a)-c))
        denominator=1-2j*t*a
        value=np.exp(-1j*t*c-.5*np.log(denominator).sum())
        if moment:value*=np.sum(b*b/denominator)
        return value.imag/t
    p,ep=quad(lambda t:integrand(t,False),0,T,epsabs=1e-11,epsrel=1e-11,limit=600)
    m,em=quad(lambda t:integrand(t,True),0,T,epsabs=1e-11,epsrel=1e-11,limit=600)
    return .5+p/math.pi,s2/2+m/math.pi,tail+ep/math.pi,mtail+em/math.pi

def run():
    output=HERE/'population_fourier_checks.json';assert not output.exists()
    checks=[]
    for Q in (8,32):
        for th in (.5,2.):
            for lam in (.2,.8):
                f=model.old.Family(Q,th,lam,0);n,_=model.probe_plan(f);t=np.arange(n);D=np.diff(np.eye(n),axis=0)
                R=lam**abs(t[:,None]-t[None,:]);K=[D@(theta*R+np.eye(n)/Q)@D.T for theta in (.05,th)]
                L=np.linalg.cholesky(K[0]);white=np.linalg.solve(L,np.linalg.solve(L,K[1]).T).T
                eig,V=np.linalg.eigh((white+white.T)/2);c=np.log(eig).sum()
                for j,theta in enumerate((.05,th)):
                    if time.process_time()>180:raise TimeoutError('Independent model-check ceiling; preserve failure')
                    b=(V.T@np.linalg.solve(L,D@(theta*lam**(n-1-t))))/(np.sqrt(eig) if j else 1)
                    a=eig-1 if j else 1-1/eig
                    series=mixture(a,b,c);p,m,pe,me=fourier(a,b,c)
                    assert abs(p-series['p'])<=pe+2e-9
                    assert abs(m-series['moment'])<=me+2e-9
                    checks.append(dict(Q=Q,theta_high=th,lambda_value=lam,regime=j,probe_n=n,
                        probability_difference=abs(p-series['p']),state_moment_difference=abs(m-series['moment']),
                        fourier_quadrature_estimate_plus_analytic_tail_bound_probability=pe,
                        fourier_quadrature_estimate_plus_analytic_tail_bound_moment=me))
    report=dict(status='PASS',independent_cases=len(checks),checks=checks,new_scientific_trajectories=0,
        no_random_draws=True,quadrature_estimates_and_floating_roundoff_not_rigorous_intervals=True,
        exact_series_and_fourier_truncation_bounds_proved_separately=True,
        cpu_snapshot_seconds=time.process_time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(dict(status='PASS',independent_cases=len(checks),cpu=time.process_time(),
        maximum_probability_difference=max(x['probability_difference'] for x in checks),
        maximum_state_moment_difference=max(x['state_moment_difference'] for x in checks))))

if __name__=='__main__':run()
