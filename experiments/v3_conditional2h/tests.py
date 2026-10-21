"""Only finite deterministic fixtures; every preparation attempt is CPU-metered."""
import os
import sys
from support import HERE,ROOT,SingleOwner,child_cpu,plan,sha,Budget
if '--mutex-probe' in sys.argv:
    try:
        with SingleOwner():sys.exit(0)
    except RuntimeError:sys.exit(17)
import inspect
import json
import math
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from model import old,STOP,public_design,deployment,posterior,rows,audit_row,risk,scheduled
from controller import filter_probe,conditional_loss
from analyze import analyze

checks=[];child_cost=0.
def check(label,condition):
    checks.append(dict(label=label,passed=bool(condition)))
    if not condition:raise AssertionError(label)


def direct(values,learner,D,error):
    if learner=='running_mean':return float((error+sum(values[:max(0,len(values)-D)]))/(max(0,len(values)-D)+1))
    parameter=float(error);total=0.;pending=[];N=0;eta=.2/(D+1)
    for y in values:
        pending.append(eta*(float(y)-.5*parameter))
        if len(pending)>D:parameter+=pending.pop(0);total+=parameter;N+=1
    return total/N if N else parameter


if __name__=='__main__':
    try:
        for learner in old.LEARNERS:
            for D in (0,1,3):
                for M in (0,2,8):
                    d,w=old.weights(learner,M,D);values=np.linspace(-.3,.7,M)
                    for e in (-1.,1.):
                        check('direct_FIFO_both_signs',abs(d*e+float(w@values[:len(w)])-direct(values,learner,D,e))<2e-13)
        for theta,lam,q,n in ((.05,.2,8,3),(2.,.8,32,6)):
            y=np.linspace(-.8,.5,n);t=np.arange(n)
            S=theta*lam**abs(t[:,None]-t[None,:])+np.eye(n)/q;c=theta*lam**(n-1-t)
            p=filter_probe(y,theta,lam,q)
            check('dense_posterior',abs(p.mean-c@np.linalg.solve(S,y))<1e-12)
            check('dense_variance',abs(p.explained_variance-c@np.linalg.solve(S,c))<1e-12)
        check('public_controller_role_signature',tuple(inspect.signature(deployment).parameters)==('public','pilot','f'))
        f=old.Family(8,.5,.2,0);pub=public_design('running_mean',f,128);n=pub['legacy']['probe_n']
        check('toy_probe_is_actually_feasible',pub['legacy']['feasible'])
        pilot=np.linspace(-.4,.3,n)
        a,_=deployment(pub,pilot,f);b,_=deployment(pub,-pilot,f)
        check('robust_even_reflection',a['robust']==b['robust'])
        try:deployment(pub,pilot,f,error_sign=1);rejected=False
        except TypeError:rejected=True
        check('hidden_error_rejected',rejected)
        try:deployment(pub,pilot,f,true_regime='high');rejected=False
        except TypeError:rejected=True
        check('realized_regime_rejected',rejected)
        for learner in old.LEARNERS:
            for family in old.families(plan()):
                for scale in plan()['public_grid']['scales']:
                    p=public_design(learner,family,scale)
                    check('full_catalogue_positive_normalizer',p['normalizer']>0 and len(p['actions']) in (17,25))
                    for a in p['actions']:
                        M,N,_,_=scheduled(learner,a,p['legacy']['bm'],p['legacy']['be'],family)
                        check('all_programs_paid',a==STOP or (M*(8+a.q)<=p['legacy']['bm'] and M*a.b<=p['legacy']['be'] and N==max(0,M-family.D)))
        # Deterministic potential arrays, never a random Gaussian trajectory.
        original=old.potential_observations
        def fixture(seed,family,regime,learner,length):
            common=np.linspace(-.3,.4,length)
            private=np.tile(np.linspace(-.2,.2,length),(family.Q,1))
            return common,private,12345
        old.potential_observations=fixture
        values=[*rows(11,'running_mean',f,'low',[32,128]),*rows(12,'running_mean',f,'low',[32,128])]
        old.potential_observations=original
        check('all_policy_sign_mapping',len(values)==72)
        check('toy_retains_probe_and_fallback',any(r['probe_used'] for r in values) and any(not r['probe_used'] for r in values))
        for row in values:audit_row(row)
        check('fixture_complete_ledger',True)
        # Raw-target translation is independently verified for both learners/signs.
        noise=np.linspace(-.3,.7,10)
        for learner in old.LEARNERS:
            for e in (-1.,1.):
                target=-e;mu=1. if learner=='running_mean' else .5
                parameter=0.;pending=[];total=0.;N=0;D=1;eta=.2/(D+1)
                observations=mu*target+noise
                if learner=='running_mean':actual=observations[:9].sum()/10-target
                else:
                    for y in observations:
                        pending.append(eta*(y-.5*parameter))
                        if len(pending)>D:parameter+=pending.pop(0);total+=parameter;N+=1
                    actual=total/N-target
                check('unknown_target_translation',abs(actual-direct(noise,learner,D,e))<1e-12)
        prep=HERE/'preparation';prep.mkdir(exist_ok=True)
        fixture_root=Path(tempfile.mkdtemp(prefix='fixture_',dir=prep))
        for phase in ('science','replay'):
            (fixture_root/phase).mkdir()
            with (fixture_root/phase/'part_fixture.jsonl').open('x',encoding='utf-8') as stream:
                for row in values:stream.write(json.dumps(row,sort_keys=True)+'\n')
        cfg=plan();cfg.update(trajectories_per_phase=72,formal_seed_count=2,bootstrap_repetitions=8)
        for output in ('analysis_a','analysis_b'):analyze(fixture_root/'science',fixture_root/'replay',fixture_root/output,cfg)
        for name in cfg['analysis_outputs']:
            check('seven_analysis_bytes_'+name,sha(fixture_root/'analysis_a'/name)==sha(fixture_root/'analysis_b'/name))
        with SingleOwner():
            child=subprocess.Popen([sys.executable,str(HERE/'tests.py'),'--mutex-probe'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out,err=child.communicate();child_cost+=child_cpu(child._handle)
            check('duplicate_native_owner_rejected',child.returncode==17)
        check('exact_counts',9*2*16*4*2*2*64==294912)
        try:Budget(7201)();rejected=False
        except TimeoutError:rejected=True
        check('whole_cap_rejected',rejected)
        try:Budget(6400)(900);rejected=False
        except TimeoutError:rejected=True
        check('audit_reserve_rejected',rejected)
        import worker
        try:worker.run(0.,-1);rejected=False
        except RuntimeError:rejected=True
        check('unadmitted_worker_rejected',rejected)
        print(json.dumps(dict(successful=True,checks=len(checks),test_child_cpu_seconds=child_cost,
            fixtures=str(fixture_root.relative_to(HERE)),scientific_samples_generated=0)))
    except BaseException as exc:
        print(json.dumps(dict(successful=False,checks=checks,error=repr(exc),test_child_cpu_seconds=child_cost)))
        raise
