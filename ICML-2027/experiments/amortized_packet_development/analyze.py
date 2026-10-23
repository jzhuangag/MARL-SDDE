"""Locked whole-study learning/independent-score/cost decision, not confirmation."""
import csv
import json
from pathlib import Path
import numpy as np
from amortized_packet_development.audit import check

BYTE_KEYS = ("public_bytes","advantage_bytes","program_bytes","probe_bytes","policy_bytes","diagnostic_probe_bytes")
MESSAGE_KEYS = ("public_messages","advantage_messages","program_messages","probe_messages","policy_messages","diagnostic_probe_messages")
ROW_KEYS = ("sampling_rows","critic_value_rows","critic_backward_rows","actor_backward_rows","score_rows",
            "potential_fit_rows","potential_predict_rows","factor_forward_rows","baseline_score_rows",
            "diagnostic_score_rows","diagnostic_potential_rows")


def work(row,plan):
    c=row["charges"]
    return (sum(c[k] for k in ROW_KEYS)+(c["state_copy_bytes"]+c["proposal_copy_bytes"]+c["program_bytes"]
             +row["initial_state_copy_bytes"])/plan["copy_bytes_per_work_unit"]
            +(c["interpolation_entries"]+c["geometry_entries"])/plan["scalar_entries_per_work_unit"])


def curve(row,plan):
    result=[]
    for e in row["evaluation"]:
        steps=[x["step_charges"] for x in row["ledger"][:e["frame"]]]
        result.append(dict(frame=e["frame"],native=e["frame"]*plan["bank_episodes"]*plan["horizon"],
            bytes=sum(sum(s[k] for k in BYTE_KEYS) for s in steps),
            messages=sum(sum(s[k] for k in MESSAGE_KEYS) for s in steps),
            value=float(np.mean(e["trained_episode_returns"])),frozen=float(np.mean(e["frozen_episode_returns"]))))
    return result


def auc(c,key):
    x=np.array([v[key] for v in c],float); y=np.array([v["value"] for v in c],float)
    assert np.all(np.diff(x)>0)
    return float(np.trapezoid(y,x)/x[-1])


def analyze(root,replay,plan,output,dump,sha):
    root,replay,output=Path(root),Path(replay),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((root/"hashes.json").read_text()); rh=json.loads((replay/"hashes.json").read_text())
    expected={(t["name"],s,m) for t in plan["tasks"] for s in plan["seeds"] for m in plan["methods"]}
    rows=[]; verified=[]
    for name,value in hashes.items():
        assert sha(root/name)==value
        row=json.loads((root/name).read_text()); verified.append(check(row,plan)); rows.append(row)
    assert len(rows)==len(expected) and {(r["task"],r["seed"],r["method"]) for r in rows}==expected
    expected_replay={(t["name"],plan["seeds"][-1],m) for t in plan["tasks"] for m in plan["replay_methods"]}
    seen=set()
    for name,value in rh.items():
        assert name in hashes and sha(replay/name)==value
        other=json.loads((replay/name).read_text()); original=next(r for r in rows if (r["task"],r["seed"],r["method"])==(other["task"],other["seed"],other["method"]))
        check(other,plan)
        assert {k:v for k,v in other.items() if k!="cpu_seconds"}=={k:v for k,v in original.items() if k!="cpu_seconds"}
        seen.add((other["task"],other["seed"],other["method"]))
    assert seen==expected_replay
    for task in plan["tasks"]:
        for seed in plan["seeds"]:
            group=[r for r in rows if r["task"]==task["name"] and r["seed"]==seed]
            assert len({r["initial_hash"] for r in group})==1
            assert all(r["evaluation_episode_ids"]==group[0]["evaluation_episode_ids"] for r in group)
            streams=[[(e["agent"],e["ordinal"],e["due"]) for e in r["program_events"]] for r in group if r["program_events"]]
            assert all(v==streams[0] for v in streams)
    metrics=[]; curves={}
    for r in rows:
        c=curve(r,plan); curves[(r["task"],r["seed"],r["method"])]=c
        ds=[x["diagnostic"] for x in r["ledger"] if x["diagnostic"] is not None]
        heldout=float(sum(np.mean(x["selected_contrasts"]) for x in ds)) if ds else 0.
        predicted=float(sum(x["selected_training_mean"] for x in ds)) if ds else 0.
        metrics.append(dict(task=r["task"],seed=r["seed"],method=r["method"],native_auc=auc(c,"native"),
            message_auc_own_horizon=auc(c,"messages"),byte_auc_own_horizon=auc(c,"bytes"),
            final_return=c[-1]["value"],final_frozen=c[-1]["frozen"],heldout_selected_gain_sum=heldout,
            selected_training_gain_sum=predicted,selection_optimism_sum=predicted-heldout,
            nonidle_receipts=sum(any(a for a in x["action"]) for x in r["ledger"]),
            communication_bytes=c[-1]["bytes"],communication_messages=c[-1]["messages"],
            work_proxy=work(r,plan),training_transitions=r["charges"]["training_transitions"],
            evaluation_transitions=r["charges"]["evaluation_transitions"],cpu_seconds=r["cpu_seconds"]))
    summaries=[]
    for task in plan["tasks"]:
        name=task["name"]; task_metrics=[r for r in metrics if r["task"]==name]
        mean=lambda method,key:float(np.mean([r[key] for r in task_metrics if r["method"]==method]))
        strongest=max(plan["controls"],key=lambda m:mean(m,"native_auc"))
        ref=mean(strongest,"native_auc"); gain=100*(mean("current_value","native_auc")-ref)/max(abs(ref),1e-8)
        end=mean(strongest,"final_return"); final=100*(mean("current_value","final_return")-end)/max(abs(end),1e-8)
        frozen=mean(strongest,"final_frozen"); learning=100*(end-frozen)/max(abs(frozen),1e-8)
        positives=sum(next(r["native_auc"] for r in task_metrics if r["seed"]==s and r["method"]=="current_value")>
                      next(r["native_auc"] for r in task_metrics if r["seed"]==s and r["method"]==strongest) for s in plan["seeds"])
        precision_positive=sum(r["heldout_selected_gain_sum"]>0 for r in task_metrics if r["method"]=="current_value")
        min_updates=min(r["nonidle_receipts"] for r in task_metrics if r["method"]=="current_value")
        resource={}
        for resource_key in ("messages","bytes"):
            stop=min(curves[(name,s,m)][-1][resource_key] for s in plan["seeds"] for m in plan["methods"])
            grid=np.linspace(0,stop,101)
            def common_auc(method):
                values=[]
                for s in plan["seeds"]:
                    c=curves[(name,s,method)]; y=np.interp(grid,[r[resource_key] for r in c],[r["value"] for r in c])
                    values.append(float(np.trapezoid(y,grid)/stop))
                return float(np.mean(values))
            resource[resource_key+"_common_horizon"]=float(stop)
            resource[resource_key+"_common_auc_by_method"]={m:common_auc(m) for m in plan["methods"]}
        byte_ratio=mean("current_value","communication_bytes")/mean(strongest,"communication_bytes")
        work_ratio=mean("current_value","work_proxy")/mean(strongest,"work_proxy")
        gates=dict(native_auc=gain>=plan["gates"]["auc_gain_percent"],positive_seeds=positives>=2,
            final_return=final>=plan["gates"]["final_change_percent"],baseline_learning=learning>=plan["gates"]["baseline_learning_percent"],
            independent_precision_mean=mean("current_value","heldout_selected_gain_sum")>0,
            independent_precision_seeds=precision_positive>=2,nondegenerate_updates=min_updates>=plan["gates"]["minimum_nonidle_receipts"],
            communication_overhead=byte_ratio<=plan["gates"]["maximum_cost_ratio"],work_overhead=work_ratio<=plan["gates"]["maximum_cost_ratio"])
        summaries.append(dict(task=name,strongest_control=strongest,native_auc_gain_percent=gain,positive_seeds=positives,
            final_return_change_percent=final,strongest_baseline_learning_percent=learning,
            heldout_selected_gain_sum_mean=mean("current_value","heldout_selected_gain_sum"),precision_positive_seeds=precision_positive,
            selection_optimism_sum_mean=mean("current_value","selection_optimism_sum"),min_nonidle_receipts=min_updates,
            full_to_strongest_byte_ratio=byte_ratio,full_to_strongest_work_proxy_ratio=work_ratio,gates=gates,**resource))
    passed=all(all(s["gates"].values()) for s in summaries)
    dump(output/"integrity.json",dict(cells=verified,semantic_replays=len(seen),exact_mapping=True,evaluation_leakage=False))
    dump(output/"summary.json",dict(decision="GO_FRESH_CONFIRMATION_PLANNING" if passed else "NO_GO",tasks=summaries,
        development_only=True,icml_ready=False,manuscript_admission=False,all_frozen_gates_applied=True,
        independent_validation_not_used_by_controller=True,selected_training_score_unbiased=False,
        cross_family_validation=False,precision_is_expected_not_high_probability=True))
    with (output/"metrics.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator="\n");w.writeheader();w.writerows(metrics)
    return summaries


def plots(root,plan,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":9,"axes.grid":True,"grid.alpha":.25})
    root,output=Path(root),Path(output); hashes=json.loads((root/"hashes.json").read_text())
    rows=[json.loads((root/name).read_text()) for name in hashes]
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout="constrained")
    for ti,task in enumerate(plan["tasks"]):
        for ci,key in enumerate(("native","messages")):
            ax=axes[ti,ci]
            for method in plan["methods"]:
                cc=[curve(r,plan) for r in rows if r["task"]==task["name"] and r["method"]==method]
                stop=min(c[-1][key] for c in cc);grid=np.linspace(0,stop,101)
                ys=np.array([np.interp(grid,[v[key] for v in c],[v["value"] for v in c]) for c in cc])
                mean=ys.mean(axis=0);se=ys.std(axis=0,ddof=1)/np.sqrt(len(ys))
                ax.plot(grid,mean,label=method,lw=2 if method=="current_value" else 1,ls="-" if method=="current_value" else "--")
                if method=="current_value": ax.fill_between(grid,mean-se,mean+se,alpha=.15)
            ax.set_xlabel("Native training transitions" if key=="native" else "Charged logical messages")
            ax.set_ylabel("Team return");ax.text(.5,-.32,f"({chr(97+ti*2+ci)}) {task['name']}",ha="center",transform=ax.transAxes)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=5,fontsize=7)
    fig.savefig(output/"return_resource.png",dpi=200);fig.savefig(output/"return_resource.pdf");plt.close(fig)
