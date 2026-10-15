"""Post-decision, offline verification/diagnosis; no new native trajectories."""
import argparse
import csv
import hashlib
import json
import math
import statistics as st
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    with Path(p).open("w",encoding="utf-8",newline="\n") as f:f.write(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def finite(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for y in x.values():finite(y)
    elif isinstance(x,list):
        for y in x:finite(y)
def average(x):return st.mean(x) if x else 0.


def main(args):
    root=ROOT/"ICML-2027/results/causal_recycle_development_20261016";replay=Path(str(root)+"_replay")
    a=Path(str(root)+"_analysis_a");b=Path(str(root)+"_analysis_b");out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/"ICML-2027/experiments/causal_recycle_development/plan.json").read_text())
    lock=json.loads((ROOT/"ICML-2027/internal/causal_recycle_development_lock_20261016.json").read_text())
    for name,digest in {**lock["qualified"]["files"],**lock["stage_files"]}.items():assert sha(ROOT/name)==digest
    harl=ROOT/"tmp/HARL"
    assert subprocess.check_output(["git","-C",str(harl),"rev-parse","HEAD"],text=True).strip()==lock["qualified"]["harl_commit"]
    assert not subprocess.check_output(["git","-C",str(harl),"status","--porcelain"],text=True).strip()
    runtime=ROOT/"ICML-2027/runtime/pettingzoo_1_24_3"
    tree="".join(str(p.relative_to(runtime)).replace("\\","/")+":"+sha(p)+"\n" for p in sorted(runtime.rglob("*.py")))
    assert hashlib.sha256(tree.encode()).hexdigest()==lock["qualified"]["mpe_runtime_tree_sha256"]
    assert json.loads((a/"workflow_complete.json").read_text())["workflow_cpu_seconds"]<plan["cpu_seconds_limit"]
    analyzed={name:sha(a/name) for name in ("decision.json","metrics.csv","integrity.json")}
    for name in analyzed:assert (a/name).read_bytes()==(b/name).read_bytes()
    rows=[];replay_count=0
    for directory,count in ((root,60),(replay,6)):
        assert json.loads((directory/"execution.json").read_text())["state"]=="complete"
        hashes=json.loads((directory/"hashes.json").read_text());assert len(hashes)==count
        for name,digest in hashes.items():
            assert sha(directory/name)==digest;row=json.loads((directory/name).read_text());finite(row)
            if directory==replay:
                original=json.loads((root/name).read_text())
                assert {k:v for k,v in original.items() if k!="cpu_seconds"}=={k:v for k,v in row.items() if k!="cpu_seconds"};replay_count+=1
            else:rows.append(row)
    expected={(t["name"],s,m) for t in plan["tasks"] for s in plan["seeds"] for m in plan["methods"]}
    assert {(r["task"],r["seed"],r["method"]) for r in rows}==expected
    details=[]
    for row in rows:
        n=len(row["actor_parameter_bytes"]);f=plan["frames"];bh=plan["bank_episodes"]*plan["horizon"];c=row["charges"]
        fresh=row["method"].startswith("happo_");full=row["method"].startswith("recycle")
        assert len(row["ledger"])==f and c["training_transitions"]==f*bh and c["evaluation_transitions"]==4800
        assert c["sampling_policy_rows"]==n*f*bh and c["source_backward_rows"]==n*f*bh
        assert c["critic_backward_rows"]==plan["critic_steps"]*f*bh
        assert c["gradient_messages"]==n*f and c["trajectory_messages"]==(n+1)*f and c["advantage_messages"]==n*f
        assert c["gradient_bytes"]==sum(row["actor_parameter_bytes"])*f and c["advantage_bytes"]==n*f*bh*4
        assert len(set(row["training_episode_ids"]))==640 and not set(row["training_episode_ids"])&set(row["evaluation_episode_ids"])
        assert not row["evaluation_used_by_controller"]
        events={e["id"]:e for e in row["packet_events"]};seen=set();ready_count=arrived_count=0;ages=[];scores=[];prices=[];trace=[];probes=[];entropies=[]
        accepted=halves=0;hash_by_version={};broadcast=trajectory=score_rows=potential_rows=proposal=0
        for t,e in enumerate(row["ledger"]):
            action=e["action"];committed=sum(x>0 for x in action);accepted+=committed;halves+=sum(x==.5 for x in action)
            before=e["source_versions"] if full or fresh else [v-int(alpha>0) for v,alpha in zip(e["source_versions"],action)]
            after=[v+int(alpha>0) for v,alpha in zip(before,action)]
            assert after==e["server_versions_after"]
            v=tuple(e["source_versions"])
            if v in hash_by_version:assert hash_by_version[v]==e["source_policy_hash"]
            hash_by_version[v]=e["source_policy_hash"]
            if not fresh:
                arrivals=[j for j in events.values() if j["id"] not in seen and j["due"]<=t]
                assert {j["id"] for j in arrivals}==set(e["arrived_packet_ids"])
                latest={}
                for j in arrivals:
                    seen.add(j["id"]);ages.append(sum(x-y for x,y in zip(before,j["source_versions"])))
                    if j["agent"] not in latest or (j["launch"],j["id"])>(latest[j["agent"]]["launch"],latest[j["agent"]]["id"]):latest[j["agent"]]=j
                assert {j["id"] for j in latest.values()}==set(e["ready_packet_ids"])
                ready_count+=len(latest);arrived_count+=len(arrivals)
            broadcast+=sum(size for size,alpha in zip(row["actor_parameter_bytes"],action) if alpha>0)
            trajectory+=e["trajectory_bytes"];score_rows+=e["hint_policy_rows"]+e["score_policy_rows"]+(2*n*bh if fresh else 0)
            potential_rows+=e["potential_rows"];proposal+=e["proposal_adam_steps"]
            ec=e["charges"]
            assert ec["policy_broadcast_bytes"]==broadcast and ec["trajectory_bytes"]==trajectory
            assert ec["extra_score_policy_rows"]==score_rows and ec["extra_potential_rows"]==potential_rows
            assert ec["proposal_adam_steps"]==proposal and ec["committed_adam_steps"]==accepted
            if full:
                d=e["control"];scores.append(abs(d["selected_gain_estimate"]));prices.append(d["QC_price"]/math.sqrt(f))
                trace.append(d["sample_variance_trace"]);probes.append(d["probe_episodes"])
                entropies.append(-sum(p*math.log(p) for p in d["action_probability"] if p>0))
        if not fresh:
            assert len(events)==f*n and row["packet_transmitted_bytes"]==c["gradient_bytes"]
            for j in events.values():
                lo,hi=plan["delay_early"] if j["launch"]<f//2 else plan["delay_late"]
                word=hashlib.sha256(f'{row["seed"]}:{j["agent"]}:{j["launch"]}'.encode()).digest()
                assert j["due"]==j["launch"]+lo+int.from_bytes(word[:8],"big")%(hi-lo+1)
                assert j["source_versions"]==row["ledger"][j["launch"]]["source_versions"]
        total=c["gradient_bytes"]+broadcast+trajectory+c["advantage_bytes"]
        assert row["curves"][-1]["logical_communication_bytes"]==total
        details.append({"task":row["task"],"seed":row["seed"],"method":row["method"],"accepted_actor_updates":accepted,
            "half_steps":halves,"launched_gradients":f*n,"arrivals_before_terminal":arrived_count,"eligible_latest_packets":ready_count,
            "superseded_before_terminal":arrived_count-ready_count,"declined_latest_packets":ready_count-accepted if not fresh else 0,
            "mean_receipt_precommit_version_age":average(ages),"mean_abs_selected_normalized_score":average(scores),
            "mean_full_broadcast_price":average(prices),"mean_sample_variance_trace":average(trace),"mean_probe_episodes":average(probes),
            "mean_action_distribution_entropy":average(entropies),"score_forward_rows":score_rows,"potential_rows":potential_rows,
            "executed_actor_adam_calls":proposal+(f*n if fresh else 0),"committed_updates_not_extra_adam_calls":accepted,
            "logical_bytes":total,"final_QC":row["ledger"][-1]["control"].get("QC_after",0),
            "final_QF":row["ledger"][-1]["control"].get("QF_after",0)})
    grouped=[]
    for task in plan["tasks"]:
        for method in plan["methods"]:
            group=[x for x in details if x["task"]==task["name"] and x["method"]==method]
            grouped.append({"task":task["name"],"method":method,**{k:average([x[k] for x in group]) for k in details[0] if k not in ("task","method","seed")}})
    matched=[]
    for task in plan["tasks"]:
        for seed in plan["seeds"]:
            x=next(r for r in rows if r["task"]==task["name"] and r["seed"]==seed and r["method"]=="recycle")
            y=next(r for r in rows if r["task"]==task["name"] and r["seed"]==seed and r["method"]=="recycle_no_hint")
            matched.append({"task":task["name"],"seed":seed,"same_actions":all(a["action"]==b["action"] for a,b in zip(x["ledger"],y["ledger"])),"same_final_policy":x["final_hash"]==y["final_hash"]})
    for name,data in (("cost_and_control.csv",details),("grouped_cost_and_control.csv",grouped)):
        with (out/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n");w.writeheader();w.writerows(data)
    verification={"scientific_cells":60,"semantic_replay_cells":replay_count,"all_registered_gates_unchanged":True,
        "source_and_runtime_hashes_pass":True,"independent_service_and_exact_costs_pass":True,"finite_and_mapping_pass":True,
        "analyzer_byte_replay":analyzed,"training_transitions":60*16000,"evaluation_transitions":60*4800,
        "replay_training_transitions":6*16000,"replay_evaluation_transitions":6*4800,
        "hint_ablation_matched":matched,"native_performance_decision":json.loads((a/"decision.json").read_text())}
    dump(out/"verification.json",verification);dump(out/"hashes.json",{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({"verified":True,"decision":verification["native_performance_decision"]["decision"],"hint_ablation_matched":matched}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);main(p.parse_args())
