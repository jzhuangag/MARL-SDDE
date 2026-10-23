"""Deterministic fixtures only; no scientific random draws or pilot outcomes."""
import sys
from support import *
if '--mutex-probe' in sys.argv:
    try:
        with SingleOwner():sys.exit(0)
    except RuntimeError:sys.exit(17)
import inspect
import math
import tempfile
import numpy as np
from model import *
from analyze import analyze
checks=[];child_cost=0.
def check(name,condition):
    checks.append(dict(label=name,passed=bool(condition)))
    if not condition:raise AssertionError(name)

def run():
    global child_cost
    for n in (2,4,9):
        for theta,lam,q in ((.05,.2,8),(2.,.8,32)):
            t=np.arange(n);S=theta*lam**abs(t[:,None]-t[None,:])+np.eye(n)/q
            D=np.diff(np.eye(n),axis=0);K=D@S@D.T;c=theta*lam**(n-1-t);b=c@D.T
            y=np.linspace(-.7,.9,n)**3
            p=invariant_filter(y,theta,lam,q);v=D@y
            expected=-.5*((n-1)*math.log(2*math.pi)+np.linalg.slogdet(K)[1]+v@np.linalg.solve(K,v))
            check('dense_invariant_density',abs(p.log_density-expected)<1e-11)
            check('dense_invariant_mean',abs(p.mean-b@np.linalg.solve(K,v))<1e-11)
            check('dense_invariant_explained_variance',abs(p.explained_variance-b@np.linalg.solve(K,b))<1e-11)
            shifted=invariant_filter(y+13.2,theta,lam,q)
            check('observable_shift_invariance',max(abs(p.mean-shifted.mean),abs(p.log_density-shifted.log_density))<1e-11)
    design_records=[]
    for f in old.families(plan()):
        n,delta=probe_plan(f);check('valid_projected_probe',n>=2 and 0<delta<=.025)
        def affinity(m):
            t=np.arange(m);D=np.diff(np.eye(m),axis=0)
            ld=[np.linalg.slogdet(D@(th*f.lam**abs(t[:,None]-t[None,:])+np.eye(m)/f.Q)@D.T)[1] for th in (f.theta_low,f.theta_high,(f.theta_low+f.theta_high)/2)]
            return math.exp(.25*(ld[0]+ld[1])-.5*ld[2])
        check('dense_projected_affinity',abs(affinity(n)-delta)<1e-11)
        check('smallest_projected_probe',n==2 or affinity(n-1)>.025)
        for learner in old.LEARNERS:
            for scale in plan()['public_grid']['scales']:
                pub=public_design(learner,f,scale);p=pub['legacy']
                check('strongest_fixed_including_stop',pub['normalizer']==min([1.]+[float(prefix_risks(learner,a,p['bm'],p['be'],f,(f.theta_low+f.theta_high)/2).min()) for a in old.catalogue(f)]))
                for action in old.catalogue(f):
                    profile=prefix_risks(learner,action,p['bm'],p['be'],f,(f.theta_low+f.theta_high)/2)
                    for M in sorted({0,1,len(profile)//2,len(profile)-1}):
                        exact=old.program_risk(learner,action,M*(f.h+action.q),M*action.b,f,(f.theta_low+f.theta_high)/2)[0]
                        check('prefix_profile_direct_weights',abs(profile[M]-exact)<1e-11)
                check('certificate_backstop',not p['corrected'] or (sum(p['U'])/2<pub['normalizer'] and p['feasible']))
                check('carryover_not_negative',not p['feasible'] or all(a>=b for a,b in zip(p['U'],p['uncorrected_U'])))
                design_records.append(dict(learner=learner,family=f.key,scale=scale,feasible=p['feasible'],certified=p['corrected'],no_carryover=p['no_carryover'],full_fixed_risk=pub['normalizer'],fixed_q=pub['best_fixed'].q,fixed_b=pub['best_fixed'].b,fixed_M=pub['best_fixed_M'],certified_upper=sum(p['U'])/2 if p['feasible'] else None))
                y=np.linspace(-.8,.7,n)
                a,_=deployment(pub,y if p['feasible'] else None,f)
                b,_=deployment(pub,y+7 if p['feasible'] else None,f)
                c,_=deployment(pub,-y if p['feasible'] else None,f)
                check('unknown_target_action_invariance',a==b)
                check('even_action_invariance',a==c)
    check('public_role_signature',tuple(inspect.signature(deployment).parameters)==('public','pilot','f'))
    for hidden in ('target','error_sign','true_regime','future'):
        try:deployment(pub,None,f,**{hidden:1});rejected=False
        except TypeError:rejected=True
        check('hidden_argument_rejected_'+hidden,rejected)
    previous_tests=previous_module('tests')
    for learner in old.LEARNERS:
        for D in (0,1,8):
            for M in (0,2,19):
                d,w=old.weights(learner,M,D);noise=np.linspace(-.5,.9,M)
                for e in (-1.,1.):check('direct_FIFO_weights',abs(d*e+w@noise[:len(w)]-previous_tests.direct(noise,learner,D,e))<1e-12)
    # Execute raw unknown-target learner observations independently of error coordinates.
    for learner in old.LEARNERS:
        for D in (0,8):
            for e in (-1.,1.):
                noise=np.linspace(-.4,.7,21);target=-e;mu=1. if learner=='running_mean' else .5
                raw=mu*target+noise;N=len(raw)-D
                if learner=='running_mean':actual=raw[:N].sum()/(N+1)-target
                else:
                    parameter=total=0.;pending=[];applied=0;eta=.2/(D+1)
                    for y in raw:
                        pending.append(eta*(y-.5*parameter))
                        if len(pending)>D:parameter+=pending.pop(0);total+=parameter;applied+=1
                    actual=total/applied-target
                check('raw_unknown_target_translation',abs(actual-previous_tests.direct(noise,learner,D,e))<1e-12)
    original=old.potential_observations
    def fixture(seed,family,regime,learner,length):return np.linspace(-.3,.4,length),np.tile(np.linspace(-.2,.2,length),(family.Q,1)),12345
    old.potential_observations=fixture;f=old.Family(8,.5,.2,0)
    values=[*rows(11,'running_mean',f,'low',[32,128]),*rows(12,'running_mean',f,'low',[32,128])]
    old.potential_observations=original
    check('complete_deterministic_mapping',len(values)==72)
    for row in values:audit_row(row)
    check('fixture_probe_and_fallback',any(r['probe_used'] for r in values) and any(not r['probe_used'] for r in values))
    prep=HERE/'preparation';prep.mkdir(exist_ok=True);root=Path(tempfile.mkdtemp(prefix='fixture_',dir=prep))
    for phase in ('science','replay'):
        (root/phase).mkdir()
        with (root/phase/'part_fixture.jsonl').open('x',encoding='utf-8') as stream:
            for row in values:stream.write(json.dumps(row,sort_keys=True)+'\n')
    cfg=plan();cfg.update(trajectories_per_phase=72,formal_seed_count=2,bootstrap_repetitions=8)
    for out in ('analysis_a','analysis_b'):analyze(root/'science',root/'replay',root/out,cfg)
    for name in cfg['analysis_outputs']:check('double_analysis_'+name,sha(root/'analysis_a'/name)==sha(root/'analysis_b'/name))
    with SingleOwner():
        child=subprocess.Popen([sys.executable,str(HERE/'tests.py'),'--mutex-probe'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        child.communicate();child_cost+=child_cpu(child._handle);check('duplicate_owner',child.returncode==17)
    for earlier,reserve in ((3601,0),(2800,900)):
        try:Budget(earlier)(reserve);rejected=False
        except TimeoutError:rejected=True
        check('whole_cap_and_reserve',rejected)
    import worker
    try:worker.run(0,-1);rejected=False
    except RuntimeError:rejected=True
    check('unadmitted_worker',rejected)
    dump(root/'data_free_designs.json',dict(designs=design_records,certificate_admissions=sum(p['certified'] for p in design_records),scientific_samples_generated=0,not_an_outcome_based_screen=True))
    print(json.dumps(dict(successful=True,checks=len(checks),test_child_cpu_seconds=child_cost,fixtures=str(root.relative_to(HERE)),scientific_samples_generated=0)))

if __name__=='__main__':
    try:run()
    except BaseException as exc:
        print(json.dumps(dict(successful=False,checks=checks,error=repr(exc),test_child_cpu_seconds=child_cost)));raise
