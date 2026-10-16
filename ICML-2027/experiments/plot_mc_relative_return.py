"""Estimator precision, not a learning-return convergence curve."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def run(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with (Path(root)/"metrics.csv").open() as f:rows=list(csv.DictReader(f))
    font_manager.findfont("Times New Roman",fallback_to_default=False)
    plt.rcParams.update({"font.family":"Times New Roman","font.size":10,"mathtext.fontset":"stix",
        "axes.grid":True,"grid.color":".65","grid.alpha":.25,"grid.linewidth":.6,"legend.frameon":False})
    lim=max(max(abs(float(r["mean"]))+1.96*float(r["se"]),abs(float(r["validation_mean"]))+1.96*float(r["validation_se"]),
                abs(float(r["branch_mean"]))+1.96*float(r["branch_se"])) for r in rows)*1.1
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for ax,task,letter in zip(axes,("spread","reference"),("a","b")):
        selected=[r for r in rows if r["task"]==task]
        for mean,se,label,color,marker in (("branch_mean","branch_se","First-disagreement","#e69f00","s"),
                                           ("mean","se","Finite-horizon MC control","#0072b2","^")):
            ax.errorbar([float(r["validation_mean"]) for r in selected],[float(r[mean]) for r in selected],
                xerr=[1.96*float(r["validation_se"]) for r in selected],yerr=[1.96*float(r[se]) for r in selected],
                fmt=marker,color=color,capsize=2,ms=4,label=label,alpha=.85)
        ax.plot([-lim,lim],[-lim,lim],"k--",lw=.8);ax.axhline(0,color=".5",lw=.6);ax.axvline(0,color=".5",lw=.6)
        ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim)
        ax.set_xlabel("Independent shared-bank return difference");ax.set_ylabel("Estimated return difference")
        ax.text(.5,-.32,f"({letter}) {task.capitalize()}",ha="center",transform=ax.transAxes)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2)
    fig.subplots_adjust(top=.85,bottom=.27,left=.08,right=.98,wspace=.30)
    fig.savefig(out/"mc_return_precision.png",dpi=180);plt.close(fig)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--output",required=True)
    args=ap.parse_args();run(args.root,args.output)
