"""Deterministic risk/decision diagnostics on the ORIGINAL public grid, no RNG."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

START=time.process_time()
for thread_option in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[thread_option]="1"
BASE=Path(__file__).resolve().parent
HERE=BASE/sys.argv[1] if len(sys.argv)>1 else BASE
if HERE!=BASE:
    HERE.mkdir(exist_ok=False)
PAPER=BASE.parents[1]
REPO=PAPER.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(REPO/"experiments/v3_risk_gate8h"))
import core
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

def forbidden(*args,**kwargs):
    raise AssertionError("No observations, RNG, trajectories or direct learners allowed")
for name in ("potential_observations","observations","direct_learner","group_rows"):
    setattr(core,name,forbidden)

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol_path=REPO/"experiments/v3_risk_gate8h/protocol.json"
source_path=REPO/"experiments/v3_risk_gate8h/core.py"
cells_path=REPO/"experiments/v3_risk_gate8h/results/formal_20261019/analysis_a/cell_results.csv"
inputs={str(p.relative_to(REPO)):sha(p) for p in (source_path,protocol_path,cells_path)}
protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
cells=list(csv.DictReader(cells_path.open(encoding="utf-8")))
assert len(cells)==1536

def action(a): return f"q{a.q}/b{a.b}"
def write_csv(name,rows):
    with (HERE/name).open("x",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

designs=[];matrix=[]
for family in core.families(protocol):
    n,delta=core.probe_plan(family)
    # Verify the actual likelihood precision difference is positive definite.
    indices=np.arange(n)
    ar=family.lam**np.abs(indices[:,None]-indices[None,:])
    cov0=np.eye(n)+family.Q*family.theta_low*ar
    cov1=np.eye(n)+family.Q*family.theta_high*ar
    precision=np.linalg.inv(cov0)-np.linalg.inv(cov1)
    eigmin=float(np.linalg.eigvalsh(precision).min())
    assert eigmin>0
    for learner in core.LEARNERS:
        for scale in protocol["public_grid"]["scales"]:
            d=core.design(learner,family,scale)
            bm,be=max(0,d["bm"]-d["probe_messages"]),max(0,d["be"]-d["probe_environment"])
            low,high=d["low"],d["high"]
            k0=core.program_risk(learner,low,bm,be,family,family.theta_low)[2]
            k1=core.program_risk(learner,high,bm,be,family,family.theta_high)[2]
            difference=k1*k1-k0*k0
            qp=norm.isf(delta/2); qm=norm.ppf((1+delta)/2)
            hp=2*qp*norm.pdf(qp);hm=2*qm*norm.pdf(qm)
            bounds=[];row={"family":family.key,"learner":learner,"scale":scale,"probe_n":n,"probe_feasible":bool(d["feasible"]),
                "delta":delta,"low_action":action(low),"high_action":action(high),
                "same_action":low==high if d["feasible"] else None,"k_low":k0 if d["feasible"] else None,"k_high":k1 if d["feasible"] else None,"k_high_sq_minus_low_sq":difference if d["feasible"] else None,
                "selection_correction_nonpositive_both_regimes":difference<=0 if d["feasible"] else None,"likelihood_precision_min_eigenvalue":eigmin,
                "frozen_corrected_admit":bool(d["corrected"]),"frozen_no_carryover_admit":bool(d["no_carryover"])}
            for j,(theta,correct,wrong,h) in enumerate(((family.theta_low,low,high,hp),(family.theta_high,high,low,hm))):
                jc=core.program_risk(learner,correct,bm,be,family,theta)[0]
                jw=core.program_risk(learner,wrong,bm,be,family,theta)[0]
                fallback=core.program_risk(learner,d["fallback"],d["bm"],d["be"],family,theta)[0]
                s2=theta-core.posterior_variance(theta,family.lam,family.Q,n)
                independent=jc+delta*max(0,jw-jc)
                signed=independent+max(0,difference)*s2*h
                old=d["U0" if j==0 else "U1"]
                row.update({f"j{j}_correct_J":jc,f"j{j}_wrong_J":jw,f"j{j}_fallback":fallback,
                    f"j{j}_old_upper":old,f"j{j}_independent_upper":independent,f"j{j}_signed_upper":signed,
                    f"j{j}_old_carry":d["carry0" if j==0 else "carry1"],
                    f"j{j}_gate_slack_without_carry":fallback+(.1 if j==0 else 0)-independent})
                bounds.append(signed)
            if not d["feasible"]:
                for key in tuple(row):
                    if key.startswith(("j0_","j1_")) and not key.endswith("_fallback"):
                        row[key]=None
            # All original full-budget fixed (q,b) programs, no observed regime supplied to selection.
            fixed=[]
            for a in core.catalogue(family):
                r0=core.program_risk(learner,a,d["bm"],d["be"],family,family.theta_low)[0]
                r1=core.program_risk(learner,a,d["bm"],d["be"],family,family.theta_high)[0]
                fixed.append((a,r0,r1))
            best0=min(fixed,key=lambda v:(v[1],v[0]))
            best1=min(fixed,key=lambda v:(v[2],v[0]))
            bestmix=min(fixed,key=lambda v:((v[1]+v[2])/2,v[0]))
            I=(bestmix[1]+bestmix[2])/2
            oracle=(best0[1]+best1[2])/2
            signed_admit=d["feasible"] and bounds[1]<row["j1_fallback"] and bounds[0]<=row["j0_fallback"]+.1
            frozen_bound=(row["j0_old_upper"]+row["j1_old_upper"])/2 if d["corrected"] else (row["j0_fallback"]+row["j1_fallback"])/2
            hypothetical_bound=(bounds[0]+bounds[1])/2 if signed_admit else (row["j0_fallback"]+row["j1_fallback"])/2
            row.update(best_low_full_action=action(best0[0]),best_high_full_action=action(best1[0]),
                best_balanced_fixed_action=action(bestmix[0]),balanced_best_fixed_risk=I,
                balanced_oracle_fixed_program_risk=oracle,balanced_adaptation_opportunity=I-oracle,
                balanced_frozen_certificate_advantage=I-frozen_bound,
                signed_bound_public_admit_diagnostic=bool(signed_admit),
                balanced_unexecuted_signed_bound_advantage=I-hypothetical_bound)
            designs.append(row)

for learner in core.LEARNERS:
    for regime in ("low","high"):
        for scale in protocol["public_grid"]["scales"]:
            part=[r for r in cells if r["learner"]==learner and r["regime"]==regime and int(r["scale"])==scale]
            gate=[r for r in part if r["policy"]=="risk_corrected_gate"]
            fixed=[r for r in part if r["policy"]=="fixed_q2_b1"]
            fg={r["family"]:r for r in fixed}
            assert len(gate)==len(fixed)==16
            diffs=[float(fg[r["family"]]["mean_risk"])-float(r["mean_risk"]) for r in gate]
            matrix.append(dict(learner=learner,regime=regime,scale=scale,
                mean_fixed_q2_minus_gate_risk=float(np.mean(diffs)),
                families_gate_lower_risk=sum(v>1e-14 for v in diffs),
                families_fixed_q2_lower_risk=sum(v< -1e-14 for v in diffs),
                ties=sum(abs(v)<=1e-14 for v in diffs),
                mean_gate_risk=float(np.mean([float(r["mean_risk"]) for r in gate])),
                mean_fixed_q2_risk=float(np.mean([float(r["mean_risk"]) for r in fixed]))))

# Exact elementary counterexample identity, verified by deterministic integration, not simulation.
t=3.;rho=.5
p=2*norm.sf(t)
carry=.75*rho*rho*2*t*norm.pdf(t)
integral=2*quad(lambda g:.75*rho*rho*(g*g-1)*norm.pdf(g),t,np.inf,epsabs=1e-13)[0]
assert abs(carry-integral)<1e-12 and carry>0
assert len(designs)==128
write_csv("design_diagnostics.csv",designs);write_csv("fixed_q2_strata.csv",matrix)
summary=dict(status="OUTCOME_KNOWN_DETERMINISTIC_ANALYSIS_NOT_NEW_VALIDATION",designs=len(designs),
    feasible_designs=sum(r["probe_feasible"] for r in designs),
    carry_order_nonpositive=sum(r["selection_correction_nonpositive_both_regimes"] is True for r in designs),
    carry_order_positive=sum(r["selection_correction_nonpositive_both_regimes"] is False for r in designs),
    same_action_designs=sum(r["same_action"] is True for r in designs),
    corrected_admissions=sum(r["frozen_corrected_admit"] for r in designs),
    signed_bound_admissions_unexecuted=sum(r["signed_bound_public_admit_diagnostic"] for r in designs),
    balanced_positive_adaptation_opportunity=sum(r["balanced_adaptation_opportunity"]>1e-12 for r in designs),
    balanced_frozen_certified_advantage_over_all_fixed=sum(r["balanced_frozen_certificate_advantage"]>1e-12 for r in designs),
    balanced_signed_bound_advantage_unexecuted=sum(r["balanced_unexecuted_signed_bound_advantage"]>1e-12 for r in designs),
    empirical_gate_lower_mean_than_fixed_q2_strata=sum(r["mean_fixed_q2_minus_gate_risk"]>0 for r in matrix),
    counterexample=dict(t=t,rho=rho,error_probability=p,independent_mixture_risk=1.,
        true_selected_risk=1+carry,positive_missing_selection_term=carry,integration_check=integral,
        scope="Analytical one-probe/two-frozen-linear-program identity; not a family from the formal grid."),
    analysis_cpu_seconds=time.process_time()-START,inputs=inputs,
    no_observations_generated=True,no_bootstrap=True,no_new_grid=True,
    original_source_bytes_unchanged=sha(source_path)==inputs[str(source_path.relative_to(REPO))])
with (HERE/"analysis_summary.json").open("x",encoding="utf-8") as file:json.dump(summary,file,indent=2,default=lambda v:v.item())
print(json.dumps(summary,indent=2,default=lambda v:v.item()))
