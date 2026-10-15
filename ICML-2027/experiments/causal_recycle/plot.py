"""All registered methods, native-data and charged-communication curves."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from run import HERE


def main(args):
    plan=json.loads((HERE/"plan.json").read_text());root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    assert json.loads((root/"execution.json").read_text())["state"]=="complete"
    assert not json.loads((root/"execution.json").read_text())["qualification"]
    plt.rcParams.update({"font.family":"Times New Roman","font.size":8,"axes.labelsize":8,"legend.fontsize":7,"pdf.fonttype":42})
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.8));styles=["--","-.",":","--","-.",":","--","-","-.",":"]
    labels=[]
    for ti,task in enumerate(plan["tasks"]):
        for col,key in enumerate(("training_transitions","logical_communication_bytes")):
            ax=axes[ti,col]
            cells={m:[json.loads((root/f'{task["name"]}_{s}_{m}.json').read_text()) for s in plan["seeds"]] for m in plan["methods"]}
            limit=min(r["curves"][-1][key] for rows in cells.values() for r in rows);grid=np.linspace(0,limit,101)
            for k,method in enumerate(plan["methods"]):
                values=np.stack([np.interp(grid,[x[key] for x in r["curves"]],[x["return"] for x in r["curves"]]) for r in cells[method]])
                color="#0072B2" if method=="recycle" else ("#D55E00" if method.startswith("recycle_") else plt.cm.Greys(.35+.06*k))
                line,=ax.plot(grid/limit,values.mean(axis=0),styles[k],color=color,lw=1.7 if method=="recycle" else 1.,label=method)
                if ti==0 and col==0:labels.append(line)
                if method=="recycle":
                    se=values.std(axis=0,ddof=1)/np.sqrt(len(values));ax.fill_between(grid/limit,values.mean(axis=0)-se,values.mean(axis=0)+se,color=color,alpha=.15,lw=0)
            ax.set_xlim(0,1);ax.set_xticks(np.linspace(0,1,5));ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.grid(color="grey",alpha=.22,lw=.5);ax.set_ylabel("Team return")
            ax.set_xlabel(("Native transition" if col==0 else "Charged communication")+" fraction\n"+f'({chr(97+ti*2+col)}) {task["name"]}')
    fig.legend(handles=labels,loc="upper center",ncol=5,frameon=False,bbox_to_anchor=(.5,1.01))
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/"complete_learning.pdf",bbox_inches="tight");fig.savefig(out/"complete_learning.png",dpi=220,bbox_inches="tight");plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True);main(p.parse_args())
