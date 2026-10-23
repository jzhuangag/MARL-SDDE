"""Publication views of the audited direct-validation table; no science/reanalysis."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from figure_style import apply_publication_style, style_axis, set_panel_label_below, PDF_METADATA

source = HERE.parents[1] / "experiments/v3_risk_gate8h/results/formal_20261019/analysis_a/cell_results.csv"
rows = list(csv.DictReader(source.open(encoding="utf-8")))
assert len(rows) == 1536
policies = ["always_all", "fixed_q2_b1", "information_only", "known_hypothesis_oracle_evaluation_only", "no_carryover_gate", "risk_corrected_gate"]
labels = ["All agents", "Fixed q=2, b=1", "Information only", "Oracle (evaluation)", "Without carryover", "Corrected gate"]
colors = ["#777777", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2"]
styles = ["--", "-.", ":", (0,(5,1)), (0,(1,2)), "-"]
scales = [32,128,512,2048]
panels = [("running_mean","low","Running mean, low"), ("running_mean","high","Running mean, high"),
          ("averaged_delayed_affine","low","Delayed affine, low"), ("averaged_delayed_affine","high","Delayed affine, high")]
apply_publication_style()

def series(learner, regime, policy, field):
    values=[]
    for s in scales:
        selected=[float(r[field]) for r in rows if r["learner"]==learner and r["regime"]==regime and r["policy"]==policy and int(r["scale"])==s]
        assert len(selected)==16 and np.isfinite(selected).all(), (learner,regime,policy,s,field)
        values.append(np.mean(selected))
    return values

def axis(ax, x, y, label, policy_index):
    ax.plot(x,y,color=colors[policy_index],linestyle=styles[policy_index],marker="o" if policy_index==5 else None,
            markersize=3,linewidth=1.4 if policy_index==5 else 1.0,label=label,zorder=5 if policy_index==5 else 2)

outputs=[]
def save(fig, name):
    for ext in ("pdf","png"):
        path=HERE/"figures"/(name+"."+ext)
        if path.exists():
            raise RuntimeError("Preserve existing figure: "+str(path))
        fig.savefig(path, dpi=220, metadata=PDF_METADATA if ext=="pdf" else None)
        outputs.append(dict(path=str(path.relative_to(HERE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plt.close(fig)

fig, axes=plt.subplots(1,4,figsize=(7.15,2.65))
for n,(ax,(learner,regime,title)) in enumerate(zip(axes,panels)):
    for i,(policy,label) in enumerate(zip(policies,labels)):
        axis(ax,scales,series(learner,regime,policy,"mean_risk"),label,i)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(scales)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x",labelrotation=35)
    ax.set_xlabel("Budget scale")
    if n==0: ax.set_ylabel("Mean terminal squared risk")
    ax.set_box_aspect(1); style_axis(ax)
    set_panel_label_below(ax,"("+chr(97+n)+") "+title,y=-.65)
fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.01))
fig.subplots_adjust(left=.065,right=.99,bottom=.35,top=.78,wspace=.38)
save(fig,"direct_gate_budget_risk_20261020")

for learner,regime,title in panels:
    fig,axes=plt.subplots(1,3,figsize=(7.15,3.05))
    fields=[(None,"Budget scale"),("mean_messages","Paid messages"),("mean_environment","Paid environment units")]
    for n,(ax,(field,xlabel)) in enumerate(zip(axes,fields)):
        for i,(policy,label) in enumerate(zip(policies,labels)):
            x=scales if field is None else series(learner,regime,policy,field)
            axis(ax,x,series(learner,regime,policy,"mean_risk"),label,i)
        ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel(xlabel)
        if n==0:
            ax.set_ylabel("Mean terminal squared risk");ax.set_xticks(scales);ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_box_aspect(1);style_axis(ax);set_panel_label_below(ax,"("+chr(97+n)+") "+xlabel,y=-.38)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.00))
    fig.subplots_adjust(left=.08,right=.99,bottom=.26,top=.78,wspace=.30)
    save(fig,"direct_gate_"+learner+"_"+regime+"_20261020")
manifest=HERE/"internal/direct_validation_integration_20261020/figure_manifest.json"
with manifest.open("x",encoding="utf-8") as file:
    json.dump(dict(input=str(source),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   aggregation="Arithmetic mean over all 16 families at each fixed learner/regime/scale/policy; no resampling",outputs=outputs),file,indent=2)
