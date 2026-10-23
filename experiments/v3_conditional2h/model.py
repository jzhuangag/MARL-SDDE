"""One paid pilot and actual new Gaussian streams; no hidden state in deployment."""
from functools import lru_cache
import hashlib
import math
import numpy as np
from support import plan
import core as old
from controller import filter_probe,conditional_loss

STOP=old.Action(0,0)


def digest(values):return hashlib.sha256(np.asarray(values,dtype='<f8').tobytes()).hexdigest()


def risk(learner,action,bm,be,f,theta):
    return (1.,1.,0.) if action==STOP else old.program_risk(learner,action,bm,be,f,theta)


@lru_cache(None)
def public_design(learner,f,scale):
    p=old.design(learner,f,scale)
    actions=(STOP,)+old.catalogue(f)
    full=tuple(tuple(risk(learner,a,p['bm'],p['be'],f,theta) for a in actions)
               for theta in (f.theta_low,f.theta_high))
    best=min(range(len(actions)),key=lambda i:((full[0][i][0]+full[1][i][0])/2,i))
    denominator=(full[0][best][0]+full[1][best][0])/2
    if not denominator>0:raise AssertionError('Nonpositive registered normalizer')
    after=tuple(tuple(risk(learner,a,p['bm']-p['probe_messages'],p['be']-p['probe_environment'],f,theta)
                      for a in actions) for theta in (f.theta_low,f.theta_high)) if p['feasible'] else None
    return dict(legacy=p,actions=actions,full=full,after=after,best_fixed=actions[best],
                normalizer=float(denominator))


def posterior(pilot,f):
    posts=tuple(filter_probe(pilot,theta,f.lam,f.Q) for theta in (f.theta_low,f.theta_high))
    log=np.array([x.log_density for x in posts]);prob=np.exp(log-log.max());prob/=prob.sum()
    return posts,prob


def deployment(public,pilot,f):
    """No true regime, seed, hidden target/error sign or future samples accepted."""
    p=public['legacy'];fallback=p['fallback']
    if not p['feasible']:
        return dict(robust=(False,fallback),marginal=(False,fallback),
                    legacy=(False,fallback),information=(False,fallback)),None
    posts,prob=posterior(pilot,f)
    scores={mode:[sum(prob[j]*conditional_loss(public['after'][j][i],posts[j],mode)
                     for j in (0,1)) for i in range(len(public['actions']))]
            for mode in ('even','marginal')}
    chosen={mode:public['actions'][min(range(len(scores[mode])),key=lambda i:(scores[mode][i],i))]
            for mode in scores}
    identified='high' if posts[1].log_density>posts[0].log_density else 'low'
    return dict(robust=(True,chosen['even']),marginal=(True,chosen['marginal']),
                legacy=(bool(p['corrected']),p[identified] if p['corrected'] else fallback),
                information=(True,p[identified])),(posts,prob)


def known_error_evaluation_only(public,posts_and_prob,error_sign):
    if posts_and_prob is None:return False,public['legacy']['fallback']
    posts,prob=posts_and_prob
    scores=[]
    for i in range(len(public['actions'])):
        scores.append(sum(prob[j]*conditional_loss((public['after'][j][i][0],
            error_sign*public['after'][j][i][1],public['after'][j][i][2]),posts[j],'full')
            for j in (0,1)))
    return True,public['actions'][min(range(len(scores)),key=lambda i:(scores[i],i))]


def scheduled(learner,action,bm,be,f):
    if action==STOP:return 0,0,1.,np.zeros(0)
    M=old.scheduled(action,bm,be,f.h);d,w=old.weights(learner,M,f.D)
    return M,len(w),d,w


def rows(seed,learner,f,regime,scales,check=lambda:None):
    """Evaluator generates noise, executes full programs and discloses outcomes only in saved rows."""
    cfg=plan()
    world=old.potential_observations(seed,f,regime,learner,4*max(scales)+1)
    n,_=old.probe_plan(f)
    pilot=old.observations(world,old.Action(f.Q,1),0,n) if n>0 and n<=4*max(scales) else None
    for scale in scales:
        check();pub=public_design(learner,f,scale);p=pub['legacy']
        choices,post=deployment(pub,pilot if p['feasible'] else None,f)
        fullbest=pub['actions'][min(range(len(pub['actions'])),
                   key=lambda i:(pub['full'][0 if regime=='low' else 1][i][0],i))]
        cached={}
        for error_sign in cfg['error_signs']:
            policy_choices=[choices['robust'],known_error_evaluation_only(pub,post,error_sign),
                choices['marginal'],choices['legacy'],choices['information'],
                (False,old.Action(f.Q,1)),(False,old.Action(2,1)),
                (False,pub['best_fixed']),(False,fullbest)]
            for policy,(used,action) in zip(cfg['policies'],policy_choices):
                pm,pe=(p['probe_messages'],n) if used else (0,0)
                key=(action,pe)
                if key not in cached:
                    M,N,d,w=scheduled(learner,action,p['bm']-pm,p['be']-pe,f)
                    values=old.observations(world,action,pe,M) if action!=STOP else np.zeros(0)
                    noise=float(w@values[:N])
                    cached[key]=(M,N,d,noise,digest(values))
                M,N,d,noise,obs_sha=cached[key]
                error=float(d*error_sign+noise)
                spent_m=pm+M*(f.h+action.q) if action!=STOP else pm
                spent_e=pe+M*action.b if action!=STOP else pe
                yield dict(seed=seed,learner=learner,family=f.key,scale=scale,regime=regime,
                    error_sign=error_sign,policy=policy,q=action.q,b=action.b,stop=action==STOP,
                    probe_used=bool(used),probe_n=n if used else 0,M=M,applied=N,
                    message_budget=p['bm'],clock_budget=p['be'],messages=spent_m,clock=spent_e,
                    probe_messages=pm,training_messages=spent_m-pm,training_clock=spent_e-pe,
                    terminal_squared_risk=error*error,terminal_error=error,
                    normalized_risk=error*error/pub['normalizer'],normalizer=pub['normalizer'],
                    pilot_sha256=digest(pilot) if used else None,scheduled_noise_sha256=obs_sha,
                    potential_stream_seed=world[2],initial_parameter=0.,hidden_target=-error_sign,
                    oracle_evaluation_only=('evaluation_only' in policy),public_radius=1.)


def audit_row(row):
    if not all(math.isfinite(row[k]) for k in ('terminal_squared_risk','terminal_error','normalizer','normalized_risk')):
        raise AssertionError('Nonfinite scientific row')
    if not 0<=row['messages']<=row['message_budget'] or not 0<=row['clock']<=row['clock_budget']:
        raise AssertionError('Unpaid budget')
    if row['terminal_squared_risk']<0 or row['normalizer']<=0:raise AssertionError('Invalid risk')
    if row['messages']!=row['probe_messages']+row['training_messages']:
        raise AssertionError('Message ledger')
    if row['clock']!=row['probe_n']+row['training_clock']:raise AssertionError('Clock ledger')
    if row['stop'] and (row['M'] or row['applied'] or row['training_messages'] or row['training_clock']):
        raise AssertionError('Stop ledger')
    if not row['stop'] and (row['training_messages']!=row['M']*(8+row['q']) or row['training_clock']!=row['M']*row['b']):
        raise AssertionError('Scheduled block ledger')
    D=int(row['family'].rsplit('-D',1)[1])
    if row['applied']!=max(0,row['M']-D):raise AssertionError('Unpaid withheld updates')
    if row['initial_parameter']!=0 or row['hidden_target']!=-row['error_sign']:
        raise AssertionError('Initialization/target mismatch')
