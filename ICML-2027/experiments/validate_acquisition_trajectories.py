"""Fresh stochastic training paths for the confirmed Gaussian critic.

These are value-estimation error curves, never team-return benchmark curves.
Evaluation truth is used only to calculate errors, not by the learner.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from efficient_critic import information_factor, innovation_gain
from run_cpu import sha256


ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"TSP-V3"/"manuscript"))
from figure_style import apply_publication_style, set_panel_label_below, style_axis


def run(source:Path,out:Path) -> None:
    if out.exists():
        raise FileExistsError("refusing to overwrite trajectory validation")
    frame=pd.read_csv(source/"endpoints.csv")
    summary=json.loads((source/"summary.json").read_text())
    if not summary["all_gates_without_replay_pass"]:
        raise ValueError("source has a failed mandatory gate")
    selected=frame[(frame.seed==frame.seed.min())&(frame.target==1)&(frame.persistence==.9)
                   &(frame.delay==2)&(frame.overhead==4)&(frame.actor_budget==32768)
                   &frame.rho.isin([0.,.5,1.])]
    methods=["learning_value_gated","budgetwise_fixed","fixed_q1","fixed_q2","fixed_q4","fixed_q8"]
    rows=[]
    for _,record in selected[selected.method.isin(methods)].iterrows():
        q,b,n=int(record.q),int(record.spacing),int(record.updates)
        r=.9**b
        variance=record.rho+(1-record.rho)/q
        method_index=methods.index(record.method)
        rng=np.random.default_rng(np.random.SeedSequence([62000001,int(record.cell),method_index]))
        replicas=2048
        noise=np.sqrt(variance)*rng.normal(size=replicas)
        estimate=np.zeros(replicas)
        history=np.zeros((3,replicas))
        previous_observation=None
        checkpoints=set(np.unique(np.rint(np.linspace(0,n,21)).astype(int)))
        rows.append({"rho":record.rho,"method":record.method,"updates":0,"budget_fraction":0.,
                     "empirical_mse":1.,"exact_mse":1.,"standard_error":0.,"within_5_se":True})
        for k in range(1,n+1):
            # Generate a stale TD packet and reconstruct its observation using
            # the known scalar Jacobian and recorded parameter version.
            stale_gradient=history[2]-(record.target+noise)
            observed=history[2]-stale_gradient
            if k==1:
                estimate=observed.copy()
            else:
                independent_observation=(observed-r*previous_observation)/(1-r)
                gain=innovation_gain(k,r)
                estimate=estimate+gain*(independent_observation-estimate)
            previous_observation=observed.copy()
            history[1:]=history[:-1]
            history[0]=estimate
            if k in checkpoints:
                squared=(estimate-record.target)**2
                empirical=float(squared.mean())
                se=float(squared.std(ddof=1)/np.sqrt(replicas))
                expected=variance/information_factor(k,r)
                mf=(record.probe_messages+k*(4+q))/65536
                ef=(record.probe_actor_transitions+k*q*b)/32768
                rows.append({"rho":record.rho,"method":record.method,"updates":k,
                             "budget_fraction":max(mf,ef),"empirical_mse":empirical,
                             "exact_mse":expected,"standard_error":se,
                             "within_5_se":abs(empirical-expected)<=5*se+1e-12})
            noise=r*noise+np.sqrt(variance*(1-r*r))*rng.normal(size=replicas)
    out.mkdir(parents=True)
    data=pd.DataFrame(rows)
    data.to_csv(out/"stochastic_curves.csv",index=False,float_format="%.17g",lineterminator="\n")
    apply_publication_style()
    styles={"learning_value_gated":("Learning-value control","#0072B2","-",2.),
            "budgetwise_fixed":("Budget-wise fixed","#222222","--",1.5),
            "fixed_q1":(r"Fixed $q=1$","#777777","--",1.),
            "fixed_q2":(r"Fixed $q=2$","#56B4E9",":",1.),
            "fixed_q4":(r"Fixed $q=4$","#009E73","-.",1.),
            "fixed_q8":(r"Fixed $q=8$","#E69F00","--",1.)}
    fig,axes=plt.subplots(1,3,figsize=(7.15,2.8),sharex=True)
    for ax,rho,letter in zip(axes,[0.,.5,1.],"abc"):
        for method,(label,color,line,width) in styles.items():
            curve=data[data.rho.eq(rho)&data.method.eq(method)].sort_values("budget_fraction")
            x=curve.budget_fraction.to_numpy()
            y=curve.empirical_mse.to_numpy()
            ax.plot(x,y,label=label,color=color,linestyle=line,linewidth=width)
            if method=="learning_value_gated":
                se=curve.standard_error.to_numpy()
                ax.fill_between(x,np.maximum(y-1.96*se,1e-12),y+1.96*se,color=color,alpha=.14,linewidth=0.)
        ax.set_yscale("log")
        ax.set_xlim(0,1)
        ax.set_xticks([0,.25,.5,.75,1],["0.00","0.25","0.50","0.75","1.00"])
        ax.set_xlabel("Charged dual-budget fraction")
        set_panel_label_below(ax,rf"({letter}) $\rho={rho:g}$",y=-.31)
        style_axis(ax)
    axes[0].set_ylabel("Value-estimation MSE")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3,frameon=False,fontsize=7.)
    fig.subplots_adjust(left=.085,right=.99,bottom=.26,top=.72,wspace=.30)
    fig.savefig(out/"stochastic_value_curves.png",dpi=240)
    plt.close(fig)
    report={"scope":"Fresh Gaussian critic learning paths; not MARL team return", "seed_base":62000001,
            "replicas_per_path":2048,"registered_contexts":3,"method_paths":18,"checkpoints":len(data),
            "all_exact_risk_checks_pass":bool(data.within_5_se.all()),"tolerance_standard_errors":5,
            "source_summary_sha256":sha256(source/"summary.json"),"runner_sha256":sha256(Path(__file__)),
            "confidence_band":"1.96 Monte Carlo standard errors across fresh learning replicas, conditional on a fixed confirmed calibration decision."}
    (out/"validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.out)
