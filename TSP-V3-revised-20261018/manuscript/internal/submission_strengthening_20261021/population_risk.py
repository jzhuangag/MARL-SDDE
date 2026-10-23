"""Outcome-known deterministic model analysis; never another trajectory study."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import chi2
HERE=Path(__file__).resolve().parent;PAPER=HERE.parents[1];ROOT=PAPER.parents[1]

def guard():
    if time.process_time()>270:raise TimeoutError('Bounded deterministic model evaluation; preserve partial work, no expansion')

def mixture(weights,state,threshold,tol=1e-13):
    """Exact-series truncation mass; floating point is NOT interval arithmetic."""
    a=np.asarray(weights,dtype=np.longdouble);b2=np.asarray(state,dtype=np.longdouble)**2
    assert np.all(a>0) and threshold>0
    beta=a.min();ratios=1-beta/a;n=len(a);s2=b2.sum()
    c=np.zeros(12001,dtype=np.longdouble);d=np.zeros_like(c);H=np.zeros_like(c)
    c[0]=np.exp(np.sum(np.log(beta/a))/2);conv=np.full(n,c[0],dtype=np.longdouble)
    H[0]=np.dot(b2*beta/a,conv);mass=c[0];hmass=H[0];power=np.ones(n,dtype=np.longdouble)
    for k in range(12001):
        if k:
            power*=ratios;d[k]=power.sum()/2
            c[k]=np.dot(d[1:k+1],c[k-1::-1])/k
            conv=ratios*conv+c[k];H[k]=np.dot(b2*beta/a,conv)
            mass+=c[k];hmass+=H[k]
        if k%128==0:guard()
        assert c[k]>=0 and H[k]>=0
        if 1-mass<tol and s2-hmass<tol*max(1.,s2):break
    else:raise ArithmeticError('Positive mixture did not converge within fixed term ceiling')
    assert mass<=1+1e-12 and hmass<=s2+1e-12
    degrees=n+2*np.arange(k+1)
    p=float(np.dot(c[:k+1],chi2.sf(float(threshold/beta),degrees)))
    moment=float(np.dot(H[:k+1],chi2.sf(float(threshold/beta),degrees+2)))
    return dict(p=p,moment=moment,probability_unretained_mass=float(max(0.,1-mass)),
        moment_unretained_mass=float(max(0.,s2-hmass)),terms=k+1,explained_variance=float(s2),
        condition_ratio=float(a.max()/beta),floating_roundoff_not_rigorously_enclosed=True)

def tests():
    checks=[]
    for n in (1,2,5):
        state=np.arange(1,n+1)/7
        for threshold in (.2,3.,12.):
            out=mixture(np.full(n,.7),state,threshold)
            assert abs(out['p']-chi2.sf(threshold/.7,n))<1e-13
            assert abs(out['moment']-np.dot(state,state)*chi2.sf(threshold/.7,n+2))<1e-13
            checks.append('equal-weight probability and state moment')
    for a1,a2 in ((.4,1.7),(.7,.8),(2.,.3)):
        for threshold in (.5,3.,9.):
            b1,b2=.3,-.7;limit=math.sqrt(threshold/a1)
            phi=lambda x:math.exp(-x*x/2)/math.sqrt(2*math.pi)
            t=lambda x:math.sqrt(max(0.,(threshold-a1*x*x)/a2))
            low=quad(lambda x:2*phi(x)*(2*ndtr(t(x))-1),0,limit,epsabs=1e-12,epsrel=1e-12)[0]
            mlow=quad(lambda x:2*phi(x)*(b1*b1*x*x*(2*ndtr(t(x))-1)+b2*b2*(2*ndtr(t(x))-1-2*t(x)*phi(t(x)))),0,limit,epsabs=1e-12,epsrel=1e-12)[0]
            out=mixture([a1,a2],[b1,b2],threshold)
            assert abs(out['p']-(1-low))<2e-11
            assert abs(out['moment']-(b1*b1+b2*b2-mlow))<2e-11
            checks.append('unequal-weight independent two-dimensional quadrature')
    return dict(status='PASS',deterministic_cases=len(checks),new_scientific_trajectories=0,no_random_draws=True,
        independent_quadrature_and_scaled_chi_square_identities=True,roundoff_not_a_rigorous_interval=True)

def evaluate():
    sys.path.insert(0,str(ROOT/'experiments/v3_observable_gate1h'))
    import model
    cfg=json.loads((ROOT/'experiments/v3_observable_gate1h/protocol.json').read_text())
    if not (ROOT/'experiments/v3_observable_gate1h/results/formal_20261021/independent_audited_terminal.json').exists():raise RuntimeError('Audited closed data prerequisite')
    @lru_cache(None)
    def probabilities(f):
        n,_=model.probe_plan(f);t=np.arange(n);R=f.lam**abs(t[:,None]-t[None,:]);D=np.diff(np.eye(n),axis=0)
        K=[D@(theta*R+np.eye(n)/f.Q)@D.T for theta in (f.theta_low,f.theta_high)]
        L=np.linalg.cholesky(K[0]);white=np.linalg.solve(L,np.linalg.solve(L,K[1]).T).T
        eig,V=np.linalg.eigh((white+white.T)/2);assert np.all(eig>1)
        threshold=float(np.log(eig).sum());out=[]
        for j,theta in enumerate((f.theta_low,f.theta_high)):
            cov=D@(theta*f.lam**(n-1-t));state=(V.T@np.linalg.solve(L,cov))/(np.sqrt(eig) if j else 1)
            expected=model.invariant_filter(np.zeros(n),theta,f.lam,f.Q).explained_variance
            assert abs(np.dot(state,state)-expected)<2e-12
            a=eig-1 if j else 1-1/eig
            result=mixture(a,state,threshold);result.update(regime=j,probe_n=n)
            out.append(result)
        return out
    rows=[];largest_terms=0
    for learner in cfg['learners']:
        for Q in cfg['public_grid']['Q']:
            for theta in cfg['public_grid']['theta_high']:
                for lam in cfg['public_grid']['lambda']:
                    for delay in cfg['public_grid']['D']:
                        f=model.old.Family(Q,theta,lam,delay)
                        for scale in cfg['public_grid']['scales']:
                            guard();pub=model.public_design(learner,f,scale);p=pub['legacy'];fixed=[]
                            for th in (f.theta_low,f.theta_high):
                                action=pub['best_fixed'];M=pub['best_fixed_M']
                                fixed.append(1. if action==model.STOP else model.risk(learner,action,M*(f.h+action.q),M*action.b,f,th)[0])
                            assert abs(sum(fixed)/2-pub['normalizer'])<2e-12
                            probe=[];errors=[]
                            if p['feasible']:
                                probs=probabilities(f)
                                for j in (0,1):
                                    J0,d0,k0=pub['after'][j][pub['indices'][0]];J1,d1,k1=pub['after'][j][pub['indices'][1]]
                                    m=probs[j];largest_terms=max(largest_terms,m['terms'])
                                    probe.append(J0+(J1-J0)*m['p']+(k1*k1-k0*k0)*(m['moment']-m['explained_variance']*m['p']))
                                    errors.append(abs(J1-J0-(k1*k1-k0*k0)*m['explained_variance'])*m['probability_unretained_mass']+abs(k1*k1-k0*k0)*m['moment_unretained_mass'])
                            else:probe=fixed;errors=[0.,0.]
                            gate=probe if p['corrected'] else fixed;den=pub['normalizer']
                            improvement=(sum(fixed)-sum(gate))/(2*den)
                            lr_improvement=(sum(probe)-sum(gate))/(2*den)
                            if p['corrected']:assert sum(gate)/2<=sum(p['U'])/2+1e-10
                            row=dict(learner=learner,family=f.key,scale=scale,admit=p['corrected'],
                                normalized_fixed_minus_gate=improvement,normalized_uncertified_minus_gate=lr_improvement,
                                normalized_truncation_error_only=sum(errors)/(2*den),fixed_prior_risk=sum(fixed)/2,
                                gate_prior_risk=sum(gate)/2,probe_prior_risk=sum(probe)/2)
                            rows.append(row)
                            with (HERE/'population_risk_partial.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
    assert len(rows)==128
    return dict(status='COMPUTED_OUTCOME_KNOWN_MODEL_ANALYSIS_NOT_NEW_VALIDATION',designs=rows,
        all_designs=128,largest_mixture_terms=largest_terms,normalized_fixed_minus_gate=sum(r['normalized_fixed_minus_gate'] for r in rows)/128,
        normalized_uncertified_minus_gate=sum(r['normalized_uncertified_minus_gate'] for r in rows)/128,
        maximum_normalized_series_truncation_error_only=max(r['normalized_truncation_error_only'] for r in rows),
        no_trajectory_seed_bootstrap_or_policy_change=True,registered_negative_decision_unchanged=True,
        floating_point_roundoff_not_rigorously_enclosed=True,not_a_confidence_interval=True)

def run():
    mode=sys.argv[1];output=HERE/('population_tests.json' if mode=='--test' else 'population_model_risk.json')
    assert not output.exists()
    if mode=='--evaluate':assert not (HERE/'population_risk_partial.jsonl').exists()
    value=tests() if mode=='--test' else evaluate()
    value.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cpu_snapshot_seconds=time.process_time())
    with output.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)
    print(json.dumps({key:val for key,val in value.items() if key!='designs'}))

if __name__=='__main__':run()
