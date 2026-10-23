"""Independent algebra/frontier checks, deterministic quadrature; no simulation."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

START=time.process_time()
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(REPO/"experiments/v3_risk_gate8h"))
import core
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

def stop(*a,**kw):raise AssertionError("No scientific observations allowed")
for name in ("potential_observations","observations","direct_learner","group_rows"):setattr(core,name,stop)
rows=list(csv.DictReader((HERE/"complete/design_diagnostics.csv").open(encoding="utf-8")))
assert len(rows)==128
checks=[];frontier=[]
for row in rows:
    # Parse the original family identifier, no newly searched family.
    q,th,lam,delay=row["family"].split("-")
    f=core.Family(int(q[1:]),float(th[2:]),float(lam[1:]),int(delay[1:]))
    learner=row["learner"];s=int(row["scale"]);bm=(8+f.Q)*s;be=4*s
    values=[]
    for a in core.catalogue(f):
        j0=core.program_risk(learner,a,bm,be,f,f.theta_low)[0]
        j1=core.program_risk(learner,a,bm,be,f,f.theta_high)[0]
        values.append((a,j0,j1))
    fixed=min((j0+j1)/2 for _,j0,j1 in values)
    worst=min(max(j0,j1) for _,j0,j1 in values)
    if row["frozen_corrected_admit"]=="True":
        B0=float(row["j0_old_upper"]);B1=float(row["j1_old_upper"])
    else:B0=float(row["j0_fallback"]);B1=float(row["j1_fallback"])
    gap=fixed-(B0+B1)/2
    assert abs(gap-float(row["balanced_frozen_certificate_advantage"]))<1e-13
    if gap>1e-12:
        assert all((j0+j1)/2-(B0+B1)/2>=gap-1e-13 for _,j0,j1 in values)
    weights=sorted(set([0.,1.]+[(v[1]-u[1])/((u[2]-u[1])-(v[2]-v[1]))
                   for i,u in enumerate(values) for v in values[i+1:]
                   if abs((u[2]-u[1])-(v[2]-v[1]))>1e-15
                   and 0<(v[1]-u[1])/((u[2]-u[1])-(v[2]-v[1]))<1]))
    for left,right in zip(weights,weights[1:]):
        pi=(left+right)/2
        best=min(values,key=lambda z:((1-pi)*z[1]+pi*z[2],z[0]))
        # Linear certificate-minus-fixed margin throughout this exact action interval.
        margin0=best[1]-B0;margin1=best[2]-B1
        slope=margin1-margin0
        if abs(slope)<1e-15:
            good_left,good_right=(left,right) if margin0>0 else (0.,0.)
        else:
            crossing=-margin0/slope
            good_left,good_right=(max(left,crossing),right) if slope>0 else (left,min(right,crossing))
        if good_right>good_left:
            frontier.append(dict(family=f.key,learner=learner,scale=s,pi_low=good_left,pi_high=good_right,
                                 best_fixed_q=best[0].q,best_fixed_b=best[0].b))
    checks.append(dict(family=f.key,learner=learner,scale=s,all_fixed_actions=len(values),
                       balanced_certificate_gap=gap,relative_gap=gap/fixed,
                       worst_case_certificate_gap=worst-max(B0,B1)))

# Positive quadratic likelihood event, integrating away the second independent normal.
quadratures=[]
for threshold in (.5,2.,6.):
    cdf=lambda z:2*norm.sf(math.sqrt(max(0.,(threshold-z*z)/.2))) if z*z<threshold else 1.
    p=2*quad(lambda z:norm.pdf(z)*cdf(z),0,np.inf,points=None,epsabs=1e-10)[0]
    e1=2*quad(lambda z:(z*z-1)*norm.pdf(z)*cdf(z),0,np.inf,epsabs=1e-10)[0]
    def second(z):
        a=math.sqrt(max(0.,(threshold-z*z)/.2))
        return 0. if a==0 else 2*a*norm.pdf(a)
    e2=2*quad(lambda z:norm.pdf(z)*second(z),0,np.inf,epsabs=1e-10)[0]
    assert e1>=-1e-9 and e2>=-1e-9
    for c1,c2 in ((1.,0.),(0.,1.),(1.,-2.)):
        s2=c1*c1+c2*c2;cov=c1*c1*e1+c2*c2*e2
        hp=2*norm.isf(p/2)*norm.pdf(norm.isf(p/2))
        hm=2*norm.ppf((1+p)/2)*norm.pdf(norm.ppf((1+p)/2))
        assert -s2*hm-1e-8<=cov<=s2*hp+1e-8
        quadratures.append(dict(threshold=threshold,c1=c1,c2=c2,probability=p,covariance=cov,positive=True,bound_pass=True))

with (HERE/"prior_frontier.csv").open("x",encoding="utf-8",newline="") as file:
    writer=csv.DictWriter(file,fieldnames=list(frontier[0]));writer.writeheader();writer.writerows(frontier)
groups=[]
for learner in core.LEARNERS:
    for scale in (32,128,512,2048):
        part=[r for r in checks if r["learner"]==learner and r["scale"]==scale]
        groups.append(dict(learner=learner,scale=scale,total=len(part),balanced_certified=sum(r["balanced_certificate_gap"]>1e-12 for r in part)))
report=dict(status="DETERMINISTIC_PROOF_AND_ORIGINAL_CATALOGUE_CHECK_NOT_NEW_SCIENCE",all_128_checked=len(checks),
    checked_all_fixed_actions=True,balanced_groups=groups,
    balanced_certificate_passes=sum(r["balanced_certificate_gap"]>1e-12 for r in checks),
    worst_case_certificate_passes=sum(r["worst_case_certificate_gap"]>1e-12 for r in checks),
    full_prior_intervals_retained=True,prior_frontier_interval_count=len(frontier),
    scalar_quadrature_fixtures=quadratures,
    positive_relative_gap_range=[min(r["relative_gap"] for r in checks if r["balanced_certificate_gap"]>1e-12),
                                 max(r["relative_gap"] for r in checks if r["balanced_certificate_gap"]>1e-12)],
    all_design_gaps=checks,observed_analysis_cpu_seconds=time.process_time()-START,
    inputs={name:hashlib.sha256((HERE/"complete"/name).read_bytes()).hexdigest() for name in ("design_diagnostics.csv","fixed_q2_strata.csv")})
with (HERE/"mechanism_checks.json").open("x",encoding="utf-8") as file:json.dump(report,file,indent=2)
print(json.dumps({k:v for k,v in report.items() if k not in ("all_design_gaps","scalar_quadrature_fixtures")},indent=2))
