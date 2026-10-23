"""Deterministic mathematical repairs; no samples or validation-study mutation."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path
import time
import sys
import numpy as np
from scipy.integrate import quad

HERE=Path(__file__).resolve().parent

def exact_witness():
    theta=(F(1,20),F(2));mean=sum(theta)/2;lam=F(1,5)
    delta=F(1,40);n=18;D=8;r=F(1)
    ratio=F(175,19)
    assert 2500*ratio<=256*(1+ratio)**2
    assert F(4,5)**(n-1)<delta
    actions=tuple((q,b) for q in (1,2,4,8) for b in (1,2,4,8))
    commits=((8,4),(2,2))
    def count(B,a,paid=False):
        q,b=a
        return max(0,min((16*B-288*paid)//(8+q),(4*B-18*paid)//b)-D)
    def v(a,t):
        q,b=a;rho=lam**b
        return F(1,q)+t*(1+rho)/(1-rho)
    def lower(B,a):
        q,b=a;N=count(B,a)
        if N==0:return r*r
        rho=lam**b;t=2*mean*rho/(1-rho)**2
        return min(r*r,(r*r+v(a,mean)-t)/4,
                   (r*r+N*v(a,mean)-t)/(N+1)**2)
    def upper(B,a,t):
        N=count(B,a,True)
        if N==0:return r*r,F(0)
        rho=lam**a[1]
        return (r*r+N*v(a,t))/(N+1)**2,(rho/((N+1)*(1-rho)))**2
    result=[]
    for B in (32,128,512,2048,8192):
        assert 16*B>=288 and 4*B>=18
        L=min(lower(B,a) for a in actions)
        U=F(0)
        for j in (0,1):
            c,kc=upper(B,commits[j],theta[j])
            w,kw=upper(B,commits[1-j],theta[j])
            U+=((1-delta)*c+delta*max(c,w)+theta[j]*(kc+kw))/2
        gain=L-U
        row=dict(B=B,probe_n=n,delay=D,all_constant_programs=len(actions),
                 fixed_prefix_lower_fraction=str(L),paid_risk_upper_fraction=str(U),
                 strict_positive=bool(gain>0),gap_fraction=str(gain),
                 relative_gain_lower_fraction=str(gain/L),
                 fixed_prefix_lower=float(L),paid_risk_upper=float(U),
                 relative_gain_lower=float(gain/L),
                 commitments_applied=[count(B,a,True) for a in commits])
        result.append(row)
    # Independent finite checks of the calculus/geometric lower bound.
    finite_checks=0
    for a in actions:
        q,b=a;rho=lam**b;t=2*mean*rho/(1-rho)**2
        for maximum in (1,2,7,32):
            bound=min(r*r,(r*r+v(a,mean)-t)/4,
                      (r*r+maximum*v(a,mean)-t)/(maximum+1)**2)
            for N in range(maximum+1):
                actual=(r*r+N*v(a,mean)-t*(1-rho**N))/(N+1)**2
                assert actual>=bound
                finite_checks+=1
    return dict(status='EXACT_RATIONAL_CERTIFICATES',rows=result,
                spectral_ratio_lower=str(ratio),affinity_upper_fraction=str(F(4,5)**17),
                each_directional_error_upper=str(delta),
                finite_rational_lower_bound_checks=finite_checks,
                no_floating_arithmetic_used_in_certification=True,
                model_chosen_as_existing_20261021_theorem_witness=True,
                not_new_or_independent_empirical_validation=True)

def general_ceiling():
    cases=[]
    for regimes in (1,2,3):
        for offset in (0.,.3,.9):
            sigma=np.array([.6,1.,1.7])[:regimes]
            slope=np.array([.4,-.6,.8])[:regimes]
            pi=np.arange(1,regimes+1,dtype=float);pi/=pi.sum()
            s2=slope*slope*sigma*sigma
            k=np.array([[.03,.6,.15],[.8,.02,.3],[.1,.4,.7]])[:regimes]+offset
            d=np.array([.2,.5,.1]);J=d*d+np.array([.08,.05,.03])+k*k*s2[:,None]
            K=4*math.exp(-.5)/math.sqrt(2*math.pi)*np.dot(pi,np.ptp(k*k,axis=1)*s2)
            def integrand(y):
                density=np.exp(-.5*(y/sigma)**2)/(sigma*math.sqrt(2*math.pi))
                joint=pi*density;g=joint.sum()
                Fv=joint@J
                correction=joint@(k*k*((slope*y)**2-s2)[:,None])
                Lv=Fv+correction
                c=int(np.argmin(Fv));u=int(np.argmin(Lv))
                mid=(np.max(k*k,axis=1)+np.min(k*k,axis=1))/2
                centered=np.dot(joint,(k[:,c]**2-mid)*((slope*y)**2-s2))
                return Lv[c]-Lv[u],centered,float(Lv[u]),float(Fv[c])
            values=[quad(lambda y:integrand(y)[i],-12.,12.,epsabs=2e-9,
                         epsrel=1e-8,limit=300)[0] for i in range(4)]
            gain,T,rob,nom=values
            assert -1e-7<=gain<=K+1e-7
            assert abs(T)<=K/2+1e-7
            # Midpoint centering integrates to zero under each regime.
            assert abs((gain+rob)-nom-T)<2e-7
            cases.append(dict(regimes=regimes,offset=offset,gain=gain,
                              ceiling=K,transport=T,identity_residual=(gain+rob)-nom-T))
    return dict(status='PASS',deterministic_quadrature_cases=cases,
                regime_specific_common_state_coefficients=True,
                new_samples_or_seeds=0,checks_support_not_replace_proof=True)

def robust_band():
    low=(F(99,2000),F(99,50));high=(F(101,2000),F(101,50))
    lamlo,lamhi,ref=F(39,200),F(41,200),F(1,5)
    eps=F(1,20);delta=F(1,10);D=8
    spectral=lambda lam:(1+lam)/(1-lam)
    factor_lo=(1-F(1,100))*spectral(lamlo)/spectral(ref)
    factor_hi=(1+F(1,100))*spectral(lamhi)/spectral(ref)
    assert factor_lo>=1-eps and factor_hi<=1+eps
    # Addition of identical independent private noise preserves these bounds;
    # congruence by D preserves them for the target-invariant contrasts.
    assert eps*eps*17/(8*(1-eps))<=F(3,40)**2
    actions=tuple((q,b) for q in (1,2,4,8) for b in (1,2,4,8))
    commits=((8,4),(2,2));rows=[]
    def count(B,a,paid):
        q,b=a
        return max(0,min((16*B-288*paid)//(8+q),(4*B-18*paid)//b)-D)
    def variance(a,theta,lam):
        q,b=a;rho=lam**b
        return F(1,q)+theta*(1+rho)/(1-rho)
    for B in (32,128,512,2048,8192):
        endpoint=[]
        for prior in (F(9,20),F(11,20)):
            mean=(1-prior)*low[0]+prior*low[1]
            lowers=[]
            for a in actions:
                N=count(B,a,False)
                if N==0:lowbound=F(1)
                else:
                    v=variance(a,mean,lamlo);rho=lamlo**a[1]
                    t=2*mean*rho/(1-rho)**2
                    lowbound=min(F(1),(1+v-t)/4,(1+N*v-t)/(N+1)**2)
                lowers.append(lowbound)
            L=min(lowers);U=F(0)
            for j,weight in enumerate((1-prior,prior)):
                vals=[];ks=[]
                for a in commits:
                    N=count(B,a,True);rho=lamhi**a[1]
                    vals.append((1+N*variance(a,high[j],lamhi))/(N+1)**2)
                    ks.append((rho/((N+1)*(1-rho)))**2 if N else F(0))
                U+=weight*((1-delta)*vals[j]+delta*max(vals)+high[j]*sum(ks))
            endpoint.append(dict(prior=str(prior),lower_fraction=str(L),upper_fraction=str(U),
                relative_gain_lower_fraction=str(1-U/L),relative_gain_lower=float(1-U/L),
                gap_fraction=str(L-U),strict_positive=L>U))
        rows.append(dict(B=B,prior_endpoints=endpoint,
            all_band_strict_positive=all(e['strict_positive'] for e in endpoint),
            uniform_relative_gain_lower=min(e['relative_gain_lower'] for e in endpoint)))
    return dict(status='EXACT_RATIONAL_UNIFORM_BAND_CERTIFICATE',rows=rows,
        spectral_covariance_lower_factor=str(factor_lo),upper_factor=str(factor_hi),
        relative_probe_covariance_band=str(eps),actual_directional_error_upper=str(delta),
        two_persistences_may_differ=True,prior_band=['9/20','11/20'],
        no_sampling_no_model_fitting_no_scientific_policy_retry=True,
        no_floating_arithmetic_used_in_certification=True)

if __name__=='__main__':
    target=HERE/('robust_band.json' if '--band' in sys.argv else 'analytic_checks.json');assert not target.exists()
    value=dict(robust_band=robust_band()) if '--band' in sys.argv else dict(witness=exact_witness(),generalized_ceiling=general_ceiling())
    value.update(
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               child_cpu_snapshot=time.process_time())
    assert time.process_time()<120
    with target.open('x',encoding='utf-8') as out:json.dump(value,out,indent=2)
    print(json.dumps(value,indent=2))
