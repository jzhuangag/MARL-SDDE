"""Complete-learning decision. Qualification never ranks or reads return values."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from run import dump,HERE,base


def check(row,plan,qualification):
    n=len(row["actor_parameter_bytes"]);full=sum(row["actor_parameter_bytes"]);f=row["frames"];b=plan["bank_episodes"];h=plan["horizon"]
    fresh=row["method"].startswith("happo_");control=row["method"].startswith("recycle")
    assert row["evaluation_used_by_controller"] is False and row["native_environment"] is True
    assert len(row["ledger"])==f and row["charges"]["training_transitions"]==f*b*h
    assert row["charges"]["source_backward_rows"]==f*n*b*h and row["charges"]["source_actor_steps"]==f*n
    assert row["charges"]["critic_steps"]==f*plan["critic_steps"]
    assert row["charges"]["critic_backward_rows"]==f*plan["critic_steps"]*b*h
    assert row["charges"]["gradient_bytes"]==f*full
    assert row["charges"]["advantage_bytes"]==f*n*b*h*4
    assert row["charges"]["gradient_messages"]==f*n and row["charges"]["trajectory_messages"]==f*(n+1)
    assert row["charges"]["advantage_messages"]==f*n
    assert len(row["training_episode_ids"])==f*b and len(set(row["training_episode_ids"]))==f*b
    assert not set(row["training_episode_ids"])&set(row["evaluation_episode_ids"])
    eval_count=plan["qualification_eval_episodes"] if qualification else plan["evaluation_episodes"]
    eval_frames=[0]+[i for i in range(1,f+1) if i%plan["evaluation_every"]==0 or i==f]
    assert [x["frame"] for x in row["curves"]]==eval_frames
    assert row["charges"]["evaluation_transitions"]==len(eval_frames)*eval_count*h
    assert row["charges"]["evaluation_policy_rows"]==len(eval_frames)*eval_count*h*n
    events={x["id"]:x for x in row["packet_events"]};seen=set();broadcast=score=potential=proposals=commits=trajectory=0
    for t,entry in enumerate(row["ledger"]):
        assert entry["frame"]==t and len(entry["action"])==n
        assert entry["outbound_bytes"]==sum(v for v,a in zip(row["actor_parameter_bytes"],entry["action"]) if a>0)
        for jid in entry["arrived_packet_ids"]:
            assert jid not in seen and events[jid]["due"]<=t;seen.add(jid)
        assert set(entry["ready_packet_ids"])<=set(entry["arrived_packet_ids"])
        if control:
            assert entry["decision_before_bank"] and entry["current_bank_used_only_after_decision"]
            assert entry["probe_episodes"] in plan["probes"]
            d=entry["control"]
            assert np.isclose(d["outbound_fraction"],entry["outbound_bytes"]/full)
            assert np.isclose(d["forward_fraction"],(entry["hint_policy_rows"]+entry["score_policy_rows"])/(2*n*b*h))
            assert np.isclose(d["QC_after"],max(0,d["QC_before"]+d["outbound_fraction"]-plan["outbound_target"]))
            assert np.isclose(d["QF_after"],max(0,d["QF_before"]+d["forward_fraction"]-plan["forward_target"]))
            assert entry["score_policy_rows"]==2*len(entry["ready_packet_ids"])*entry["probe_episodes"]*h
        broadcast+=entry["outbound_bytes"];score+=entry["hint_policy_rows"]+entry["score_policy_rows"]
        if fresh:score+=2*n*b*h
        potential+=entry["potential_rows"];proposals+=entry["proposal_adam_steps"];commits+=sum(a>0 for a in entry["action"])
        trajectory+=entry["trajectory_bytes"]
        c=entry["charges"]
        assert c["policy_broadcast_bytes"]==broadcast and c["extra_score_policy_rows"]==score
        assert c["extra_potential_rows"]==potential and c["proposal_adam_steps"]==proposals and c["committed_adam_steps"]==commits
        assert c["trajectory_bytes"]==trajectory and c["training_transitions"]==(t+1)*b*h
    if not fresh:
        assert len(events)==f*n and row["packet_transmitted_bytes"]==f*full
        for i in range(n):
            actor_events=[e for e in events.values() if e["agent"]==i]
            assert [e["launch"] for e in actor_events]==list(range(f))
            for event in actor_events:assert event["source_versions"]==row["ledger"][event["launch"]]["source_versions"]
    assert row["charges"]["sampling_policy_rows"]==f*n*b*h
    assert row["charges"]["policy_messages"]==commits
    return {"task":row["task"],"seed":row["seed"],"method":row["method"],"payload_pass":True,
            "has_version_staleness":any(age>0 for e in row["ledger"] for age in e["arrival_version_ages"])}


def semantic(row):return {k:v for k,v in row.items() if k!="cpu_seconds"}


def main(args):
    root=Path(args.root);execution=json.loads((root/"execution.json").read_text());assert execution["state"]=="complete"
    plan=json.loads((HERE/"plan.json").read_text());qualification=execution["qualification"]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((root/"hashes.json").read_text());rows=[];integrity=[]
    for name,sha in hashes.items():
        assert base.sha(root/name)==sha;row=json.loads((root/name).read_text());integrity.append(check(row,plan,qualification));rows.append(row)
    expected=2*len(plan["methods"])*(1 if qualification else len(plan["seeds"]))
    assert len(rows)==expected and len({(r["task"],r["seed"],r["method"]) for r in rows})==expected
    for task in plan["tasks"]:
        for seed in ([plan["qualification_seed"]] if qualification else plan["seeds"]):
            group=[r for r in rows if r["task"]==task["name"] and r["seed"]==seed]
            assert len({r["initial_hash"] for r in group})==1
            service=[[(e["agent"],e["launch"],e["due"]) for e in r["packet_events"]] for r in group if not r["method"].startswith("happo_")]
            assert all(s==service[0] for s in service)
    replay_checked=0
    if args.replay_root:
        replay=Path(args.replay_root);assert json.loads((replay/"execution.json").read_text())["state"]=="complete"
        for row in rows:
            name=f'{row["task"]}_{row["seed"]}_{row["method"]}.json'
            if (replay/name).exists():
                assert semantic(row)==semantic(json.loads((replay/name).read_text()));replay_checked+=1
        assert replay_checked==(expected if qualification else 6)
    dump(out/"integrity.json",{"payloads":integrity,"replay_checked":replay_checked,"qualification":qualification})
    if qualification:
        # DO NOT inspect or aggregate any qualification return value.
        verdict={"decision":"QUALIFIED" if replay_checked==expected else "REPLAY_REQUIRED","performance_evidence":False,
            "cells":expected,"replay_cells":replay_checked,"all_accounting_causal_payload_checks":True}
        dump(out/"decision.json",verdict);return
    metrics=[]
    for row in rows:
        curve=row["curves"];x=np.array([p["training_transitions"] for p in curve]);y=np.array([p["return"] for p in curve])
        assert np.isfinite(y).all()
        metrics.append({"task":row["task"],"seed":row["seed"],"method":row["method"],
            "auc":float(np.trapz(y,x)/(x[-1]-x[0])),"final_return":float(y[-1]),"initial_return":float(y[0]),
            "logical_bytes":curve[-1]["logical_communication_bytes"],"extra_score_rows":row["charges"]["extra_score_policy_rows"],
            "proposal_adam_steps":row["charges"]["proposal_adam_steps"],"potential_rows":row["charges"]["extra_potential_rows"]})
    decisions=[]
    for task in plan["tasks"]:
        name=task["name"];controls=[m for m in plan["methods"] if not m.startswith("recycle")]
        best=max(controls,key=lambda m:np.mean([r["auc"] for r in metrics if r["task"]==name and r["method"]==m]))
        gains=[];final=[];communication=[]
        for seed in plan["seeds"]:
            a=next(r for r in metrics if r["task"]==name and r["seed"]==seed and r["method"]=="recycle")
            b=next(r for r in metrics if r["task"]==name and r["seed"]==seed and r["method"]==best)
            gains.append((a["auc"]-b["auc"])/max(abs(b["auc"]),1));final.append((a["final_return"]-b["final_return"])/max(abs(b["final_return"]),1))
            ac=next(r["curves"] for r in rows if r["task"]==name and r["seed"]==seed and r["method"]=="recycle")
            bc=next(r["curves"] for r in rows if r["task"]==name and r["seed"]==seed and r["method"]==best)
            end=min(ac[-1]["logical_communication_bytes"],bc[-1]["logical_communication_bytes"]);grid=np.linspace(0,end,201)
            aa=np.trapz(np.interp(grid,[c["logical_communication_bytes"] for c in ac],[c["return"] for c in ac]),grid)/end
            bb=np.trapz(np.interp(grid,[c["logical_communication_bytes"] for c in bc],[c["return"] for c in bc]),grid)/end
            communication.append(float((aa-bb)/max(abs(bb),1)))
        gates=plan["mandatory_gates"]
        passed=np.mean(gains)>=gates["both_tasks_mean_auc_gain_min"] and np.mean(np.array(gains)>0)>=gates["each_task_positive_seed_fraction_min"] and np.mean(final)>=gates["both_tasks_final_return_change_min"]
        staleness=all(r["has_version_staleness"] for r in integrity if r["task"]==name and r["method"]=="recycle")
        decisions.append({"task":name,"strongest_baseline":best,"seed_auc_gains":gains,"mean_auc_gain":float(np.mean(gains)),
            "positive_seed_fraction":float(np.mean(np.array(gains)>0)),"mean_final_return_change":float(np.mean(final)),
            "mean_common_range_communication_auc_gain":float(np.mean(communication)),"nonzero_version_staleness":staleness,"pass":bool(passed and staleness)})
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator="\n");w.writeheader();w.writerows(metrics)
    dump(out/"decision.json",{"decision":"DEVELOPMENT_PASS" if all(d["pass"] for d in decisions) and replay_checked==6 else "NO_GO",
        "tasks":decisions,"replay_checked":replay_checked,"confirmation_or_manuscript_evidence":False})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True);p.add_argument("--replay-root");main(p.parse_args())
