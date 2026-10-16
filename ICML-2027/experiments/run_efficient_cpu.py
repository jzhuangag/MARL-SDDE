"""Prospective allocation test against equally efficient fixed-q critics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from efficient_critic import efficient_actions, efficient_forecast, efficient_select, efficient_true_risk
from learning_value import covers, fit_calibration, observe_calibration
from run_cpu import CONFIG, contexts, dump, sha256


HERE = Path(__file__).resolve().parent
PLAN = HERE/"efficient_preregistration.json"


def run(out: Path, lock_path: Path|None) -> None:
    plan, base = json.loads(PLAN.read_text()), json.loads(CONFIG.read_text())
    if sha256(CONFIG) != plan["base_config_sha256"]:
        raise ValueError("base population changed")
    seeds, phase = plan["development_seeds"], "development"
    if lock_path:
        lock = json.loads(lock_path.read_text())
        if not lock["development_all_pass"] or lock["plan_sha256"] != sha256(PLAN):
            raise ValueError("invalid confirmation lock")
        if lock["implementation_sha256"] != sha256(HERE/"efficient_critic.py"):
            raise ValueError("changed efficient critic")
        seeds, phase = lock["seeds"], "confirmation"
        if set(seeds)&set(plan["development_seeds"]):
            raise ValueError("seed overlap")
    if out.exists():
        raise FileExistsError("refusing to overwrite results")
    out.mkdir(parents=True)
    rows, certs = [], []
    for cell in contexts(base):
        paid = efficient_actions(base,cell,paid=True)
        unpaid = efficient_actions(base,cell,paid=False)
        best = min(paid,key=lambda a: efficient_true_risk(a,cell["rho"]))
        other = min(efficient_true_risk(a,cell["rho"]) for a in paid if a.q != best.q)
        interior = best.q in (2,4) and other/efficient_true_risk(best,cell["rho"]) >= 1.03
        fixed = {f"fixed_q{q}":min((a for a in unpaid if a.q==q),key=lambda a: efficient_true_risk(a,cell["rho"])) for q in base["q_catalogue"]}
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed,cell["cell"],0]))
            values = observe_calibration(rng,target=cell["target"],rho=cell["rho"],persistence=cell["persistence"],
                                         workers=base["calibration_workers"],observations=base["calibration_innovations"])
            cal = fit_calibration(values,persistence=cell["persistence"],delta=base["calibration_delta"])
            selected = efficient_select(paid,cal)
            corr = min(paid,key=lambda a:(cal.variance(a.q)[0]/a.updates,a.q,a.spacing)).q
            corr_action = efficient_select([a for a in paid if a.q==corr],cal)
            methods = {"efficient_value":selected,"correlation_only":corr_action,**fixed,"paid_oracle":best}
            valid = all(efficient_forecast(a,cal)[1]-1e-12 <= efficient_true_risk(a,cell["rho"]) <= efficient_forecast(a,cal)[2]+1e-12 for a in paid)
            bound = efficient_forecast(selected,cal)[2]-min(efficient_forecast(a,cal)[1] for a in paid)
            regret = efficient_true_risk(selected,cell["rho"])-efficient_true_risk(best,cell["rho"])
            certs.append({**cell,"seed":seed,"joint_covered":covers(cal,target=cell["target"],rho=cell["rho"]),
                          "all_intervals_valid":valid,"regret_bound_valid":regret<=bound+1e-12,
                          "regret":regret,"regret_upper":max(0.,bound),"interior_qualified":interior,
                          "selected_q":selected.q,"oracle_q":best.q})
            for name,action in methods.items():
                rows.append({**cell,"seed":seed,"method":name,"q":action.q,"spacing":action.spacing,
                             "updates":action.updates,"risk":efficient_true_risk(action,cell["rho"]),
                             "messages":action.messages,"actor_transitions":action.actor_transitions,
                             "complete_catalogue":{a.q for a in paid}==set(base["q_catalogue"])})
        if cell["cell"]%90==0:
            print(f"{phase}: {cell['cell']+1}/360",flush=True)
    frame, cert = pd.DataFrame(rows), pd.DataFrame(certs)
    frame.to_csv(out/"endpoints.csv",index=False,float_format="%.17g",lineterminator="\n")
    cert.to_csv(out/"certificates.csv",index=False,float_format="%.17g",lineterminator="\n")
    aggregate = {m:float(np.exp(np.log(frame.loc[frame.method.eq(m),"risk"]).mean())) for m in plan["methods"]}
    strongest = min(fixed,key=lambda m:aggregate[m])
    pivot = frame.pivot(index=["seed","cell"],columns="method",values="risk")
    comparisons = {}
    for opponent in (strongest,"paid_oracle","correlation_only"):
        logs = np.log(pivot.efficient_value/pivot[opponent]).groupby("seed").mean()
        radius = t.ppf(.975,len(logs)-1)*logs.std(ddof=1)/np.sqrt(len(logs))
        comparisons[opponent] = {"ratio":float(np.exp(logs.mean())),"ci95_low":float(np.exp(logs.mean()-radius)),"ci95_high":float(np.exp(logs.mean()+radius))}
    interior, covered = cert[cert.interior_qualified], cert[cert.joint_covered]
    cfg = plan["mandatory_gates"]
    gates = {
        "complete_finite_exact_accounting":bool(np.isfinite(frame.select_dtypes(include=[np.number])).all().all()
            and frame.complete_catalogue.all() and frame.messages.le(base["message_budget"]).all()
            and frame.actor_transitions.le(frame.actor_budget).all() and len(frame)==360*len(seeds)*len(plan["methods"])),
        "coverage_event_implies_all_risk_intervals":bool(covered.all_intervals_valid.all() and covered.regret_bound_valid.all()),
        "joint_coverage_fraction_min":cert.joint_covered.mean()>=cfg["joint_coverage_fraction_min"],
        "ratio_to_strongest_unpaid_efficient_fixed_q_max":comparisons[strongest]["ratio"]<=cfg["ratio_to_strongest_unpaid_efficient_fixed_q_max"],
        "ratio_to_paid_efficient_oracle_max":comparisons["paid_oracle"]["ratio"]<=cfg["ratio_to_paid_efficient_oracle_max"],
        "interior_qualified_cells_min":interior.cell.nunique()>=cfg["interior_qualified_cells_min"],
        "interior_selection_fraction_min":bool(len(interior) and interior.selected_q.eq(interior.oracle_q).mean()>=cfg["interior_selection_fraction_min"]),
    }
    summary = {"experiment_id":plan["experiment_id"],"phase":phase,"cells":360,"seeds":len(seeds),"rows":len(frame),
               "plan_sha256":sha256(PLAN),"implementation_sha256":sha256(HERE/"efficient_critic.py"),"runner_sha256":sha256(Path(__file__)),
               "geometric_risks":aggregate,"strongest_unpaid_efficient_fixed_q":strongest,"comparisons":comparisons,
               "joint_coverage_fraction":float(cert.joint_covered.mean()),"risk_interval_coverage_fraction":float(cert.all_intervals_valid.mean()),
               "interior_qualified_cells":int(interior.cell.nunique()),"interior_selection_fraction":float(interior.selected_q.eq(interior.oracle_q).mean()),
               "gates_without_replay":{k:bool(v) for k,v in gates.items()},"all_gates_without_replay_pass":bool(all(gates.values())),
               "scope":plan["scope"],"gpu_performance_authorized":False}
    dump(out/"summary.json",summary)
    (out/"SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in sorted(out.iterdir()) if p.is_file()),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--confirmation-lock",type=Path)
    args=parser.parse_args()
    run(args.out,args.confirmation_lock)
