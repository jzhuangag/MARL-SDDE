"""Preserved-result visualization; never invokes environment or training."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"ICML-2027/results/amortized_packet_development_20261016"
OUT=ROOT/"ICML-2027/results/amortized_packet_development_audit_20261016"
METHODS=["fresh_happo_half","fresh_happo","fresh_happo_double","fresh_parallel","eligible_all","age_clip","online_kl","current_value","full_trace","no_price"]
LABELS=["HAPPO x0.5","HAPPO x1","HAPPO x2","Fresh parallel","All-ready","Age-scaled","KL-adaptive","Full method","Full-vector only","No resource prices"]
COLORS=["#999999","#D55E00","#009E73","#E69F00","#CC79A7","#7D5A50","#56B4E9","#0072B2","#6F4E9C","#222222"]
BYTE_KEYS=("public_bytes","advantage_bytes","program_bytes","probe_bytes","policy_bytes","diagnostic_probe_bytes")
MESSAGE_KEYS=("public_messages","advantage_messages","program_messages","probe_messages","policy_messages","diagnostic_probe_messages")


def main():
    validation=json.loads((OUT/"validation.json").read_text())
    rows=validation["exact_cost_ledger_checks"]
    if not (OUT/"cost_counts.csv").exists():
        with (OUT/"cost_counts.csv").open("x",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    else:
        with (OUT/"cost_counts.csv").open(encoding="utf-8",newline="") as f:
            assert len(list(csv.DictReader(f)))==66
    hashes=json.loads((DATA/"hashes.json").read_text());cells=[json.loads((DATA/name).read_text()) for name in hashes]
    plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":10,"grid.color":"#B0B0B0","grid.alpha":.25,"grid.linewidth":.6})
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    fig.subplots_adjust(left=.075,right=.96,bottom=.07,top=.865,hspace=.67,wspace=.23)
    for col,task in enumerate(("spread","reference")):
        selected=[r for r in cells if r["task"]==task]
        curves={}
        for row in selected:
            records=[]
            for ev in row["evaluation"]:
                steps=[x["step_charges"] for x in row["ledger"][:ev["frame"]]]
                records.append(dict(native=ev["frame"]*500,
                    messages=sum(sum(s[k] for k in MESSAGE_KEYS) for s in steps),
                    bytes=sum(sum(s[k] for k in BYTE_KEYS) for s in steps),value=float(np.mean(ev["trained_episode_returns"]))))
            curves[(row["seed"],row["method"])]=records
        for ri,key in enumerate(("native","messages","bytes")):
            ax=axes[ri,col]
            stop=min(c[-1][key] for c in curves.values());grid=np.linspace(0,stop,101)
            for mi,method in enumerate(METHODS):
                cc=[c for (seed,m),c in curves.items() if m==method]
                values=np.array([np.interp(grid,[p[key] for p in c],[p["value"] for p in c]) for c in cc])
                mean=values.mean(axis=0);se=values.std(axis=0,ddof=1)/np.sqrt(len(cc))
                # Draw the coincident no-price trace visibly dashed; never hide an ablation.
                style="-" if method=="current_value" else ":" if method=="no_price" else "--"
                ax.plot(grid/(1e6 if key=="bytes" else 1),mean,color=COLORS[mi],ls=style,
                    lw=2.1 if method=="current_value" else 1.1,label=LABELS[mi],zorder=4 if method=="current_value" else 5 if method=="no_price" else 2)
                if method=="current_value":ax.fill_between(grid/(1e6 if key=="bytes" else 1),mean-se,mean+se,color=COLORS[mi],alpha=.12,zorder=1)
            ax.grid(True);ax.set_ylabel("Team return")
            ax.set_xlabel({"native":"Native training transitions","messages":"Charged logical messages","bytes":"Charged logical payload (MB)"}[key])
            ax.set_xlim(0,stop/(1e6 if key=="bytes" else 1))
            ax.text(.5,-.34,f"({chr(97+ri*2+col)}) {task.capitalize()}",ha="center",transform=ax.transAxes)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.985),ncol=5,frameon=False,fontsize=9)
    for suffix in ("png","pdf"):
        target=OUT/f"verified_return_resource_presentation.{suffix}"
        if target.exists():raise FileExistsError(target)
        fig.savefig(target,dpi=180)
    plt.close(fig)
    caption=("Complete frozen development, all methods and all three seeds per task. "
             "Resource panels use a shared horizon within each task; payload MB denotes decimal logical array bytes, not measured wire traffic. "
             "Shading is one seed-level SEM for the full method, not a95% confidence interval. "
             "Full and no-price policy paths coincide in all six task/seed cells. "
             "No smoothing, new rollout, method removal or frozen-plot replacement was performed.\n")
    with (OUT/"figure_caption_presentation.txt").open("x",encoding="utf-8") as f:f.write(caption)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()}
    with (OUT/"PRESENTATION_SHA256SUMS.json").open("x",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":main()
