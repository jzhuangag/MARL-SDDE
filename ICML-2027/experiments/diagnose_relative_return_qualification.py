"""Post-decision precision/cost diagnosis; never changes registered gates."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from relative_return_estimator import empirical_bernstein


def physics_return_bound(task,h):
    """Conservative native finite-horizon bound, not a fitted sample range.

    Initial coordinates in [-1,1], zero velocity, dt=.1, damping=.25,
    mass=1, discrete physical force norm<=5. Spread contact force magnitude
    per neighbor<=100*(.3+.001*log(2)); landmarks have no collision.
    Position updates precede velocity updates; H*dt*vmax overbounds motion.
    Rewards are nonpositive. Exact coincident collision states are singular
    in upstream real-arithmetic dynamics and outside its finite-state domain.
    """
    force=5.0
    if task=="spread":
        force+=2*100*(.3+.001*math.log(2))
    vmax=.1*force/.25
    displacement=h*.1*vmax
    if task=="spread":
        per_step=.5*3*(2*math.sqrt(2)+displacement)+.5*2
    else:
        per_step=2*math.sqrt(2)+displacement
    return h*per_step


def diagnose(root,out):
    root,out=Path(root),Path(out)
    if json.loads((root/"execution.json").read_text())["state"]!="complete":
        raise AssertionError("do not inspect incomplete outcomes")
    out.mkdir(parents=True,exist_ok=False)
    payloads=[json.loads(p.read_text()) for p in sorted(root.glob("context_*.json"))]
    records=[]; confidence=[]
    k=len(payloads)*3
    for p in payloads:
        b=physics_return_bound(p["task"]["name"],25)
        for row in p["candidates"]:
            truth=np.asarray(row["validation"])
            truth_mean=float(truth.mean())
            truth_se=float(truth.std(ddof=1)/math.sqrt(len(truth)))
            for method,xs in row["samples"].items():
                x=np.asarray(xs)
                mean,se=float(x.mean()),float(x.std(ddof=1)/math.sqrt(len(x)))
                variance=float(x.var(ddof=1))
                needed=max(2,math.ceil(1.645**2*variance/(truth_mean**2))) if truth_mean!=0 else None
                ledger=row["ledgers"][method]
                per_transition=ledger["joint_transitions"]/len(x)
                low,radius=empirical_bernstein(x,2*b,.05/k)
                if (np.abs(x)>b+1e-6).any():
                    raise AssertionError("native analytical support violated")
                records.append({"context":p["index"],"task":p["task"]["name"],"stage":p["stage"],"scale":row["scale"],
                                "method":method,"mean":mean,"se":se,"validation_mean":truth_mean,"validation_se":truth_se,
                                "mean_nonzero":bool(mean!=0),"score_correct_sign":bool(mean*truth_mean>0),
                                "zero_sample_fraction":float((x==0).mean()),"normal_precision_sample_projection":needed,
                                "normal_precision_transition_projection":needed*per_transition if needed else None,
                                "normal_projection_validated_effect_precise":bool(abs(truth_mean)>1.645*truth_se),
                                "support_width":2*b,"empirical_bernstein_lower":low,"empirical_bernstein_radius":radius})
                if method=="first_disagreement":
                    confidence.append(low)
    with (out/"precision_cost.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n");w.writeheader();w.writerows(records)
    result={"formal_interface":"Known-support simultaneous empirical Bernstein, delta=.05/24; no sample-range substitution",
            "confidence_positive_count":sum(x>0 for x in confidence),"confidence_candidates":len(confidence),
            "normal_projections_are_not_certificates":True,
            "score_nonzero_count_by_method":{m:sum(r["mean_nonzero"] for r in records if r["method"]==m) for m in ("paired_crn","paired_maximal","first_disagreement")},
            "native_bound":{"spread":physics_return_bound("spread",25),"reference":physics_return_bound("reference",25)},
            "scope":"Variance and empirical predictive utility do not establish an affordable certified online control rule."}
    with (out/"diagnosis.json").open("w",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    return records,result


def plot(records,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.findfont("Times New Roman",fallback_to_default=False)
    plt.rcParams.update({"font.family":"Times New Roman","font.size":10,"mathtext.fontset":"stix",
                         "pdf.fonttype":42,"axes.grid":True,"grid.color":".65","grid.alpha":.25,
                         "grid.linewidth":.6,"legend.frameon":False})
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    methods=["paired_crn","paired_maximal","first_disagreement"]
    labels=["Paired CRN","Maximal coupling","First-disagreement"]
    styles=[("#999999","o"),("#e69f00","s"),("#0072b2","^")]
    for ax,task,letter in zip(axes,["spread","reference"],["a","b"]):
        for method,label,(color,marker) in zip(methods,labels,styles):
            rr=[r for r in records if r["task"]==task and r["method"]==method]
            xx=np.array([r["validation_mean"] for r in rr]);yy=np.array([r["mean"] for r in rr])
            ax.errorbar(xx,yy,xerr=[1.96*r["validation_se"] for r in rr],
                        yerr=[1.96*r["se"] for r in rr],fmt=marker,color=color,label=label,alpha=.8,capsize=2,ms=4)
        lim=max(max(abs(r["mean"])+1.96*r["se"],abs(r["validation_mean"])+1.96*r["validation_se"])
                for r in records if r["task"]==task)*1.1
        ax.plot([-lim,lim],[-lim,lim],"k--",lw=.8)
        ax.axhline(0,color=".5",lw=.6);ax.axvline(0,color=".5",lw=.6)
        ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim)
        ax.set_xlabel("Independent paired return difference")
        ax.set_ylabel("Estimated return difference")
        ax.text(.5,-.32,f"({letter}) {task.capitalize()}",ha="center",transform=ax.transAxes)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3)
    fig.subplots_adjust(top=.85,bottom=.27,left=.08,right=.98,wspace=.30)
    fig.savefig(Path(out)/"relative_return_precision.png",dpi=180)
    fig.savefig(Path(out)/"relative_return_precision.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--output",required=True)
    args=ap.parse_args();records,result=diagnose(args.root,args.output);plot(records,args.output)
    print(json.dumps(result,indent=2))
