"""Deterministic algebra/quadrature checks; not scientific trajectories or seeds."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from fractions import Fraction as F
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'experiments/v3_observable_gate1h'))
from model import invariant_filter
checks=[]
def check(label,condition):
    checks.append(dict(label=label,passed=bool(condition)))
    if not condition:raise AssertionError(label)

def run():
    # General p-dimensional nuisance projection; no random matrices.
    for n,p in ((4,1),(6,2),(7,3)):
        grid=np.arange(n,dtype=float);X=np.column_stack([(grid/(n+1))**k for k in range(p)])
        U,_,_=np.linalg.svd(X,full_matrices=True);D=U[:,p:].T
        A=np.fromfunction(lambda i,j:np.cos((i+1)*(j+1)),(n,n))
        S=A@A.T+np.eye(n);inv=np.linalg.inv(S)
        Pi=inv-inv@X@np.linalg.solve(X.T@inv@X,X.T@inv)
        check('general_projection_identity',np.allclose(Pi,D.T@np.linalg.solve(D@S@D.T,D),rtol=1e-11,atol=1e-11))
        lhs=np.linalg.slogdet(D@S@D.T)[1]
        rhs=np.linalg.slogdet(S)[1]+np.linalg.slogdet(X.T@inv@X)[1]+np.linalg.slogdet(D@D.T)[1]-np.linalg.slogdet(X.T@X)[1]
        check('general_determinant_identity',abs(lhs-rhs)<1e-10)
        S0=S;S1=2.3*S+.7*np.eye(n);Sm=(S0+S1)/2
        logfull=.25*(np.linalg.slogdet(S0)[1]+np.linalg.slogdet(S1)[1])-.5*np.linalg.slogdet(Sm)[1]
        logcontrast=.25*(np.linalg.slogdet(D@S0@D.T)[1]+np.linalg.slogdet(D@S1@D.T)[1])-.5*np.linalg.slogdet(D@Sm@D.T)[1]
        a0=np.linalg.eigvalsh(S0).min();a1=np.linalg.eigvalsh(S1).min();bm=np.linalg.eigvalsh(Sm).max()
        logC=p/2*math.log(bm)-p/4*math.log(a0*a1)
        check('affinity_finite_nuisance_bound',-1e-12<=logcontrast-logfull<=logC+1e-12)
    phi=lambda x:math.exp(-x*x/2)/math.sqrt(2*math.pi)
    absolute=quad(lambda x:abs(x*x-1)*phi(x),-12,12,points=[-1,1],epsabs=1e-12)[0]
    check('absolute_centered_gaussian_square',abs(absolute-4*phi(1))<1e-11)
    # Deterministic integrals check the four-term decomposition including signed T.
    for variance in (1.,3.):
        sd=math.sqrt(variance);density=lambda y:phi(y/sd)/sd
        for offset in (-.7,.3):
            def fields(y):
                # One known Gaussian regime is enough to test state-value algebra.
                F=np.array([1.,1.+offset]);k2=np.array([0.,.8]);S=y*y-variance
                L=F+k2*S;c=int(np.argmin(F));u=int(np.argmin(L))
                return np.array([F[c],L[c],L[u],k2[c]*S,L[c]-L[u]])
            values=np.array([quad(lambda y,i=i:fields(y)[i]*density(y),-12*sd,12*sd,epsabs=1e-9,limit=200)[0] for i in range(5)])
            nom,actual,rob,T,state=values;K=4*phi(1)*.8*variance
            check('actual_nominal_signed_carry',abs(actual-nom-T)<1e-8)
            check('state_value_ceiling',-1e-9<=state<=K+1e-8)
            check('four_term_identity',abs((1.-rob)-((1.-nom)+state-T))<1e-8)
            check('oracle_plus_half_ceiling',rob>=min(1.,1.+offset)-K/2-1e-8)
    # Finite symmetric, two-regime fixture with nominal state-selection carry.
    # Both signs of each probe atom occur; the moment normalization is exact.
    y=np.array([-2.,-1.,0.,1.,2.]);weights=np.array([[.05,.2,.5,.2,.05],[.2,.2,.2,.2,.2]])
    pi=np.array([.5,.5]);g=pi@weights;post=(pi[:,None]*weights/g).T
    m=np.array([.3*y,.8*y]);s2=np.array([np.sum(weights[j]*m[j]**2) for j in range(2)])
    Js=np.array([[.7,1.3],[1.7,.9]]);k2=np.array([.1,.8])
    nominal_scores=post@Js;S=np.sum(post*(m*m-s2[:,None]).T,axis=1);L=nominal_scores+S[:,None]*k2
    c=np.argmin(nominal_scores,axis=1);u=np.argmin(L,axis=1);indices=np.arange(len(y))
    nominal=g@nominal_scores[indices,c];actual=g@L[indices,c];rob=g@L[indices,u];T=g@(k2[c]*S);state=actual-rob
    check('mixture_S_centered',abs(g@S)<1e-14)
    check('mixture_carry_nonzero',abs(T)>1e-4)
    check('mixture_four_term_identity',abs((1.-rob)-((1.-nominal)+state-T))<1e-13)
    check('generic_half_range_bound',abs(T)<=.5*np.ptp(k2)*(g@abs(S))+1e-14)
    # Matrix tie example: arbitrary pure dual argmin is not minimax.
    H0=np.diag([1.,0.]);H1=np.diag([0.,1.]);S=np.eye(2)/2
    dual=min(np.trace(H0@S),np.trace(H1@S));pure=np.linalg.eigvalsh(H0).max();mixed=np.linalg.eigvalsh((H0+H1)/2).max()
    check('matrix_tie_counterexample',dual==mixed==.5 and pure==1.)
    # Registered delay stability is a symbolic crossing argument; roots are checks only.
    for D in (0,1,8):
        alpha=.1/(D+1);coeff=np.zeros(D+2);coeff[0]=1;coeff[1]-=1;coeff[-1]+=alpha
        check('registered_delay_roots',max(abs(np.roots(coeff)))<1)
        check('strict_first_crossing_bound',alpha<2*math.sin(math.pi/(4*D+2)))
    actions=[(q,b) for q in (1,2,4,8) for b in (1,2,4,8)]
    def coefficient(theta,a):
        q,b=a;rho=F(1,5)**b;rate=min(F(16,8+q),F(4,b))
        return (F(1,q)+theta*(1+rho)/(1-rho))/rate
    low=min(actions,key=lambda a:coefficient(F(1,20),a))
    high=min(actions,key=lambda a:coefficient(F(2),a))
    balanced=min(actions,key=lambda a:coefficient(F(41,40),a))
    check('exact_leading_low_program',low==(8,4) and coefficient(F(1,20),low)==F(1093,6240))
    check('exact_leading_high_program',high==(2,2) and coefficient(F(2),high)==F(5,3))
    K=coefficient(F(41,40),balanced)
    check('exact_balanced_leading_program',balanced==(2,2) and K==F(773,768))
    A=((coefficient(F(1,20),high)-coefficient(F(1,20),low))+(coefficient(F(2),low)-coefficient(F(2),high)))/2
    Delta=K-(coefficient(F(1,20),low)+coefficient(F(2),high))/2
    check('exact_wrong_commitment_coefficient',A==F(407,1280))
    margin=Delta-A/40
    check('paid_leading_margin_positive',margin>0 and .07764<float(margin)<.07766)
    check('relative_leading_margin',.0771<float(margin/K)<.0772)
    output=HERE/'extension_checks.json'
    if output.exists():raise RuntimeError('Preserve prior attempt')
    report=dict(status='PASS',checks=checks,count=len(checks),scientific_samples_generated=0,
        cpu_seconds=time.process_time(),not_a_proof_substitute=True,
        rejected_nominal_decomposition_preserved='value_decomposition_failure_01.json',
        exact_leading_witness=dict(K=str(K),Delta=str(Delta),wrong_coefficient=str(A),margin=str(margin),relative_margin=float(margin/K)),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'matrix_extension.tex',HERE/'observable_probe.tex',HERE/'state_value_ceiling.tex',HERE/'value_decomposition.tex',HERE/'leading_fixed_separation.tex',Path(__file__))})
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',count=len(checks),cpu_seconds=time.process_time())))

if __name__=='__main__':run()
