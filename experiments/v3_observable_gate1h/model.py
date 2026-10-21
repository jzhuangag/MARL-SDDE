"""Observable location-invariant probe and prospective strongest-fixed backstop."""
from functools import lru_cache
import math
import numpy as np
from scipy.special import ndtri
from scipy.signal import lfilter
from support import plan,previous_module
from controller import Posterior,filter_probe,conditional_loss
base=previous_module('model')
old=base.old
STOP=base.STOP
risk=base.risk
scheduled=base.scheduled
digest=base.digest
audit_row=base.audit_row

@lru_cache(None)
def prefix_risks(learner,action,bm,be,f,theta):
    """All scheduled prefixes; only the declared fixed learner and step vary in length."""
    Mmax=old.scheduled(action,bm,be,f.h);Nmax=max(0,Mmax-f.D)
    result=np.ones(Mmax+1)
    if Nmax:
        N=np.arange(1,Nmax+1,dtype=float);rho=f.lam**action.b
        if learner=='running_mean':
            G=np.ones(Nmax);H=lfilter([0.,rho],[1.,-rho],G)
            common=np.cumsum(1+2*H)
            result[f.D+1:]=(1+theta*common+N/action.q)/(N+1)**2
        else:
            eta=.2/(f.D+1);mu=.5
            _,w=old.weights(learner,Mmax,f.D);G=w[::-1]*(Nmax/eta)
            H=lfilter([0.,rho],[1.,-rho],G)
            diagonal=np.cumsum(G*G);common=np.cumsum(G*G+2*G*H)
            d=1-mu*eta*np.cumsum(G)/N
            result[f.D+1:]=d*d+(eta/N)**2*(theta*common+diagonal/action.q)
    result.flags.writeable=False
    return result

def best_fixed_prefix(learner,f,bm,be,theta):
    best=(1.,STOP,0)
    for action in old.catalogue(f):
        values=prefix_risks(learner,action,bm,be,f,theta);M=int(np.argmin(values))
        candidate=(float(values[M]),action,M)
        if candidate<best:best=candidate
    return best

def invariant_filter(values,theta,lam,q):
    y=np.asarray(values,dtype=float)
    if y.ndim!=1 or len(y)<2 or not np.isfinite(y).all():raise ValueError('At least two finite raw observations required')
    # Observable anchoring improves stability; no true target is subtracted.
    y=y-y[0]
    my=m1=0.;P=float(theta);S=T=U=ld=0.
    for i,value in enumerate(y):
        F=P+1/q;ey=value-my;e1=1-m1
        S+=e1*e1/F;T+=ey*e1/F;U+=ey*ey/F;ld+=math.log(F)
        K=P/F;my+=K*ey;m1+=K*e1;P*=1-K
        if i+1<len(y):my*=lam;m1*=lam;P=lam*lam*P+(1-lam*lam)*theta
    explained=max(0.,theta-P-m1*m1/S)
    mean=my-m1*T/S
    log=-.5*((len(y)-1)*math.log(2*math.pi)+ld+math.log(S)+max(0.,U-T*T/S))
    return Posterior(float(mean),float(theta-explained),float(explained),float(log))

@lru_cache(None)
def probe_plan(f):
    theta=(f.theta_low,f.theta_high,(f.theta_low+f.theta_high)/2)
    P=list(theta);m=[0.]*3;S=[0.]*3;ld=[0.]*3
    for n in range(1,4097):
        logs=[]
        for j,th in enumerate(theta):
            F=P[j]+1/f.Q;e=1-m[j];S[j]+=e*e/F;ld[j]+=math.log(F)
            K=P[j]/F;m[j]+=K*e;P[j]*=1-K
            logs.append(ld[j]+math.log(S[j]))
            m[j]*=f.lam;P[j]=f.lam**2*P[j]+(1-f.lam**2)*th
        affinity=min(1.,math.exp(.25*(logs[0]+logs[1])-.5*logs[2]))
        if n>=2 and affinity<=.025:return n,affinity
    return 0,1.

@lru_cache(None)
def public_design(learner,f,scale):
    bm=(f.h+f.Q)*scale;be=4*scale;actions=(STOP,)+old.catalogue(f)
    full=tuple(tuple(risk(learner,a,bm,be,f,t) for a in actions) for t in (f.theta_low,f.theta_high))
    normalizer,fixed,fixed_M=best_fixed_prefix(learner,f,bm,be,(f.theta_low+f.theta_high)/2)
    n,delta=probe_plan(f);pm=n*(f.h+f.Q)
    feasible=n>0 and pm<=bm and n<=be
    p=dict(bm=bm,be=be,probe_n=n,delta=delta,probe_messages=pm,probe_environment=n,feasible=feasible,fallback=fixed)
    after=None;indices=None;upper=None;uncorrected=None
    if feasible:
        after=tuple(tuple(risk(learner,a,bm-pm,be-n,f,t) for a in actions) for t in (f.theta_low,f.theta_high))
        indices=tuple(min(range(len(actions)),key=lambda i:(after[j][i][0],i)) for j in (0,1))
        s2=[invariant_filter(np.zeros(n),t,f.lam,f.Q).explained_variance for t in (f.theta_low,f.theta_high)]
        delta_k=after[0][indices[1]][2]**2-after[0][indices[0]][2]**2
        t=ndtri(1-delta/2);a=ndtri((1+delta)/2)
        phi=lambda z:math.exp(-z*z/2)/math.sqrt(2*math.pi)
        h=(2*t*phi(t),2*a*phi(a))
        uncorrected=tuple(after[j][indices[j]][0]+delta*max(0.,after[j][indices[1-j]][0]-after[j][indices[j]][0]) for j in (0,1))
        upper=tuple(float(uncorrected[j]+max(0.,delta_k)*s2[j]*h[j]) for j in (0,1))
        p.update(low=actions[indices[0]],high=actions[indices[1]])
    p.update(corrected=bool(feasible and sum(upper)/2<normalizer),
             no_carryover=bool(feasible and sum(uncorrected)/2<normalizer),U=upper,uncorrected_U=uncorrected)
    return dict(legacy=p,actions=actions,full=full,after=after,indices=indices,best_fixed=fixed,best_fixed_M=fixed_M,normalizer=float(normalizer))

def posterior(pilot,f):
    posts=tuple(invariant_filter(pilot,t,f.lam,f.Q) for t in (f.theta_low,f.theta_high))
    logs=np.array([p.log_density for p in posts]);prob=np.exp(logs-logs.max());prob/=prob.sum()
    return posts,prob

def deployment(public,pilot,f):
    """Only public specification and raw observable probe; no nuisance target."""
    p=public['legacy'];fallback=public['best_fixed']
    if not p['feasible']:return {k:(False,fallback) for k in ('certified','uncertified','no_carryover','marginal')},None
    posts,prob=posterior(pilot,f);j=int(posts[1].log_density>posts[0].log_density)
    scores=[sum(prob[h]*public['after'][h][i][0] for h in (0,1)) for i in range(len(public['actions']))]
    marginal=public['actions'][min(range(len(scores)),key=lambda i:(scores[i],i))]
    selected=p['high'] if j else p['low']
    return dict(certified=(True,selected) if p['corrected'] else (False,fallback),
        uncertified=(True,selected),no_carryover=(True,selected) if p['no_carryover'] else (False,fallback),
        marginal=(True,marginal)),(posts,prob)

def known_target_evaluation_only(public,centered,f,error_sign):
    if not public['legacy']['feasible']:return False,public['best_fixed']
    posts=tuple(filter_probe(centered,t,f.lam,f.Q) for t in (f.theta_low,f.theta_high))
    logs=np.array([p.log_density for p in posts]);prob=np.exp(logs-logs.max());prob/=prob.sum()
    scores=[sum(prob[j]*conditional_loss((public['after'][j][i][0],error_sign*public['after'][j][i][1],public['after'][j][i][2]),posts[j],'full') for j in (0,1)) for i in range(len(public['actions']))]
    return True,public['actions'][min(range(len(scores)),key=lambda i:(scores[i],i))]

def rows(seed,learner,f,regime,scales,check=lambda:None):
    cfg=plan();world=old.potential_observations(seed,f,regime,learner,4*max(scales)+1)
    n,_=probe_plan(f)
    noise_pilot=old.observations(world,old.Action(f.Q,1),0,n) if 0<n<=4*max(scales) else None
    for scale in scales:
        check();pub=public_design(learner,f,scale);p=pub['legacy'];cached={}
        j=0 if regime=='low' else 1
        _,known,known_M=best_fixed_prefix(learner,f,p['bm'],p['be'],f.theta_low if j==0 else f.theta_high)
        for error_sign in cfg['error_signs']:
            mu=1. if learner=='running_mean' else .5
            raw=mu*(-error_sign)+noise_pilot if p['feasible'] else None
            choices,_=deployment(pub,raw,f)
            selected=[choices['certified'],known_target_evaluation_only(pub,noise_pilot,f,error_sign),choices['uncertified'],choices['no_carryover'],choices['marginal'],(False,old.Action(f.Q,1)),(False,old.Action(2,1)),(False,pub['best_fixed']),(False,known)]
            for policy,(used,action) in zip(cfg['policies'],selected):
                pm,pe=(p['probe_messages'],n) if used else (0,0)
                # Every non-probing adaptive fallback uses the strongest fixed prefix.
                prefix=known_M if policy==cfg['policies'][8] else (pub['best_fixed_M'] if not used and policy not in cfg['policies'][5:7] else None)
                key=(action,pe,prefix)
                if key not in cached:
                    bm,be=(p['bm']-pm,p['be']-pe) if prefix is None else ((f.h+action.q)*prefix,action.b*prefix)
                    M,N,d,w=scheduled(learner,action,bm,be,f)
                    values=old.observations(world,action,pe,M) if action!=STOP else np.zeros(0)
                    cached[key]=(M,N,d,float(w@values[:N]),digest(values))
                M,N,d,noise,obs_sha=cached[key];error=float(d*error_sign+noise)
                sm=pm+M*(f.h+action.q) if action!=STOP else pm
                se=pe+M*action.b if action!=STOP else pe
                yield dict(seed=seed,learner=learner,family=f.key,scale=scale,regime=regime,error_sign=error_sign,policy=policy,q=action.q,b=action.b,stop=action==STOP,probe_used=bool(used),probe_n=n if used else 0,M=M,applied=N,message_budget=p['bm'],clock_budget=p['be'],messages=sm,clock=se,probe_messages=pm,training_messages=sm-pm,training_clock=se-pe,terminal_squared_risk=error*error,terminal_error=error,normalized_risk=error*error/pub['normalizer'],normalizer=pub['normalizer'],pilot_sha256=digest(raw) if used else None,observable_contrast_sha256=digest(np.diff(raw)) if used else None,scheduled_noise_sha256=obs_sha,potential_stream_seed=world[2],initial_parameter=0.,hidden_target=-error_sign,oracle_evaluation_only='evaluation_only' in policy,public_radius=1.,candidate_admit=p['corrected'],candidate_no_carryover=p['no_carryover'],projected_directional_bound=p['delta'],certified_prior_upper=sum(p['U'])/2 if p['feasible'] else None)
