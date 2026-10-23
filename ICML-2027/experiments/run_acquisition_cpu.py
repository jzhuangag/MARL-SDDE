"""Fully charged acquisition-aware allocation and budget-wise strong controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from acquisition_gate import gated_select, perfect_information_is_futile, public_fallback
from efficient_critic import efficient_actions, efficient_forecast, efficient_select, efficient_true_risk
from learning_value import calibration_cost, covers, fit_calibration, observe_calibration
from run_cpu import CONFIG, contexts, dump, sha256


HERE=Path(__file__).resolve().parent
PLAN=HERE/"acquisition_preregistration.json"


def run(out:Path, lock_path:Path|None) -> None:
    plan,base=json.loads(PLAN.read_text()),json.loads(CONFIG.read_text())
    if sha256(CONFIG)!=plan["base_config_sha256"]:
        raise ValueError("changed population")
    base["calibration_innovations"]=plan["calibration_innovations"]
    seeds,phase=plan["development_seeds"],"development"
    if lock_path:
        lock=json.loads(lock_path.read_text())
        if not lock["development_all_pass"] or lock["plan_sha256"]!=sha256(PLAN):
            raise ValueError("invalid confirmation lock")
        for filename,expected in lock["source_hashes"].items():
            if sha256(HERE/filename)!=expected:
                raise ValueError("changed frozen source "+filename)
        seeds,phase=lock["seeds"],"confirmation"
        if set(seeds)&set(plan["development_seeds"]):
            raise ValueError("seed overlap")
    if out.exists():
        raise FileExistsError("refusing to overwrite an experiment")
    out.mkdir(parents=True)
    rows,certificates=[],[]
    for cell in contexts(base):
        paid,unpaid=efficient_actions(base,cell,paid=True),efficient_actions(base,cell,paid=False)
        fallback=public_fallback(unpaid,plan["rho_prior"])
        futile=perfect_information_is_futile(paid,fallback)
        best_paid=min(paid,key=lambda a:efficient_true_risk(a,cell["rho"]))
        admissible=min([best_paid,fallback],key=lambda a:efficient_true_risk(a,cell["rho"]))
        next_q=min(efficient_true_risk(a,cell["rho"]) for a in paid if a.q!=best_paid.q)
        interior=(not futile and best_paid.q in (2,4) and next_q/efficient_true_risk(best_paid,cell["rho"])>=1.03)
        fixed={f"fixed_q{q}":max((a for a in unpaid if a.q==q),key=lambda a:(a.information_factor,-a.spacing)) for q in base["q_catalogue"]}
        for seed in seeds:
            # Always-probe is a distinct comparator. If gated selection skips,
            # its path does not call observe_calibration or receive these values.
            rng=np.random.default_rng(np.random.SeedSequence([seed,cell["cell"],0]))
            values=observe_calibration(rng,target=cell["target"],rho=cell["rho"],persistence=cell["persistence"],
                                       workers=base["calibration_workers"],observations=base["calibration_innovations"])
            cal=fit_calibration(values,persistence=cell["persistence"],delta=base["calibration_delta"])
            always=efficient_select(paid,cal)
            selected, skipped=gated_select(paid,unpaid,plan["rho_prior"],lambda:cal)
            if skipped!=futile:
                raise ArithmeticError("inconsistent public gate")
            valid=all(efficient_forecast(a,cal)[1]-1e-12<=efficient_true_risk(a,cell["rho"])<=efficient_forecast(a,cal)[2]+1e-12 for a in paid)
            covered=covers(cal,target=cell["target"],rho=cell["rho"])
            certificates.append({**cell,"seed":seed,"skip_probe":futile,"gate_parameter_inputs":0 if futile else values.size,
                                  "joint_covered":covered,"all_risk_intervals_valid":valid,
                                  "interior_qualified":interior,"selected_q":selected.q,"oracle_q":best_paid.q})
            methods={"learning_value_gated":selected,"always_probe_efficient":always,"budgetwise_fixed":fallback,
                     **fixed,"admissible_oracle":admissible}
            pm,pe,_=calibration_cost(base["calibration_workers"],base["calibration_innovations"],cell["overhead"])
            for method,a in methods.items():
                paid_method=method=="always_probe_efficient" or (method=="learning_value_gated" and not futile)
                rows.append({**cell,"seed":seed,"method":method,"q":a.q,"spacing":a.spacing,"updates":a.updates,
                             "risk":efficient_true_risk(a,cell["rho"]),"messages":a.messages,"actor_transitions":a.actor_transitions,
                             "probe_messages":pm if paid_method else 0,"probe_actor_transitions":pe if paid_method else 0,
                             "skip_probe":futile if method=="learning_value_gated" else False,
                             "complete_catalogue":{x.q for x in paid}==set(base["q_catalogue"])})
        if cell["cell"]%90==0:
            print(f"{phase}: {cell['cell']+1}/360",flush=True)
    frame,cert=pd.DataFrame(rows),pd.DataFrame(certificates)
    frame.to_csv(out/"endpoints.csv",index=False,float_format="%.17g",lineterminator="\n")
    cert.to_csv(out/"certificates.csv",index=False,float_format="%.17g",lineterminator="\n")
    pivot=frame.pivot(index=["seed","cell"],columns="method",values="risk")
    comparison={}
    for opponent in ("budgetwise_fixed","admissible_oracle","always_probe_efficient"):
        logs=np.log(pivot.learning_value_gated/pivot[opponent]).groupby("seed").mean()
        radius=t.ppf(.975,len(logs)-1)*logs.std(ddof=1)/np.sqrt(len(logs))
        comparison[opponent]={"ratio":float(np.exp(logs.mean())),"ci95_low":float(np.exp(logs.mean()-radius)),"ci95_high":float(np.exp(logs.mean()+radius))}
    probed=cert[~cert.skip_probe]
    covered=probed[probed.joint_covered]
    skipped=frame[frame.method.eq("learning_value_gated")&frame.skip_probe]
    interior=cert[cert.interior_qualified]
    cfg=plan["mandatory_gates"]
    gates={
        "complete_finite_exact_accounting":bool(len(frame)==360*len(seeds)*len(plan["methods"])
            and np.isfinite(frame.select_dtypes(include=[np.number])).all().all()
            and frame.messages.le(base["message_budget"]).all() and frame.actor_transitions.le(frame.actor_budget).all()),
        "full_catalogue":bool(frame.complete_catalogue.all()),
        "no_charge_or_observation_when_futile":bool(skipped.probe_messages.eq(0).all() and skipped.probe_actor_transitions.eq(0).all()
            and cert.loc[cert.skip_probe,"gate_parameter_inputs"].eq(0).all()
            and np.allclose(pivot.loc[cert.set_index(["seed","cell"]).skip_probe,"learning_value_gated"],pivot.loc[cert.set_index(["seed","cell"]).skip_probe,"budgetwise_fixed"],rtol=0,atol=0)),
        "coverage_event_implies_true_risk_intervals":bool(covered.all_risk_intervals_valid.all()),
        "joint_parameter_coverage_fraction_min":probed.joint_covered.mean()>=cfg["joint_parameter_coverage_fraction_min"],
        "ratio_to_budgetwise_efficient_fixed_max":comparison["budgetwise_fixed"]["ratio"]<=cfg["ratio_to_budgetwise_efficient_fixed_max"],
        "budgetwise_paired_ci95_upper_max":comparison["budgetwise_fixed"]["ci95_high"]<cfg["budgetwise_paired_ci95_upper_max"],
        "ratio_to_admissible_oracle_max":comparison["admissible_oracle"]["ratio"]<=cfg["ratio_to_admissible_oracle_max"],
        "interior_qualified_cells_min":interior.cell.nunique()>=cfg["interior_qualified_cells_min"],
        "interior_selection_fraction_min":bool(len(interior) and interior.selected_q.eq(interior.oracle_q).mean()>=cfg["interior_selection_fraction_min"]),
    }
    summary={"experiment_id":plan["experiment_id"],"phase":phase,"cells":360,"seeds":len(seeds),"rows":len(frame),
             "comparisons":comparison,"skip_probe_cells":int(cert[cert.skip_probe].cell.nunique()),
             "probed_joint_coverage_fraction":float(probed.joint_covered.mean()),"probed_all_action_risk_coverage_fraction":float(probed.all_risk_intervals_valid.mean()),
             "interior_qualified_cells":int(interior.cell.nunique()),"interior_selection_fraction":float(interior.selected_q.eq(interior.oracle_q).mean()),
             "gates_without_replay":{k:bool(v) for k,v in gates.items()},"all_gates_without_replay_pass":bool(all(gates.values())),
             "plan_sha256":sha256(PLAN),"source_hashes":{f:sha256(HERE/f) for f in ["learning_value.py","efficient_critic.py","acquisition_gate.py","run_acquisition_cpu.py"]},
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
