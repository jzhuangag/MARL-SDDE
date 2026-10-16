"""Post-completion audit/diagnosis only; no environment, rollout or retraining."""
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"amortized_packet_development"))
sys.path.insert(0,str(HERE))
from amortized_packet_development.audit import check
from amortized_packet_development.analyze import curve,auc,work,BYTE_KEYS,MESSAGE_KEYS
from amortized_packet_development.workflow import state

ROOT=HERE.parents[1]
DATA=ROOT/"ICML-2027/results/amortized_packet_development_20261016"
OUT=ROOT/"ICML-2027/results/amortized_packet_development_audit_20261016"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    with Path(path).open("w",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    plan=json.loads((HERE/"amortized_packet_development/plan.json").read_text())
    lockpath=ROOT/"ICML-2027/internal/amortized_packet_development_lock_20261016.json"
    lock=json.loads(lockpath.read_text());assert state()==lock
    complete=json.loads((DATA/"workflow_complete.json").read_text())
    execution=json.loads((DATA/"execution.json").read_text())
    assert execution["state"]=="complete" and complete["cells"]==60 and complete["replays"]==6
    assert execution["source_commit"]=="dd279792735ecf0c91b4bb3e86420c3c7a732736"
    assert sha(lockpath)==complete["source_lock_sha256"]
    for relative,value in lock["files"].items():
        original=subprocess.check_output(["git","show",f'{execution["source_commit"]}:{relative}'])
        assert hashlib.sha256(original).hexdigest()==value
    analyses=[Path(str(DATA)+"_analysis_"+s) for s in ("a","b")]
    analysis_hashes={}
    for name in ("metrics.csv","summary.json","integrity.json"):
        assert (analyses[0]/name).read_bytes()==(analyses[1]/name).read_bytes()
        analysis_hashes[name]=sha(analyses[0]/name)
    rows=[];verified=[];artifacts={}
    for directory,count in ((DATA,60),(Path(str(DATA)+"_replay"),6)):
        hashes=json.loads((directory/"hashes.json").read_text());assert len(hashes)==count
        assert {p.name for p in directory.glob("*.json") if p.name.startswith(("spread_","reference_"))}==set(hashes)
        phase=[]
        for name,value in hashes.items():
            assert sha(directory/name)==value
            row=json.loads((directory/name).read_text());assert row["source_lock_sha256"]==sha(lockpath)
            verified.append(check(row,plan));phase.append(row)
            artifacts[str((directory/name).relative_to(ROOT)).replace("\\","/")]=value
        if directory==DATA:rows=phase
        else:
            for row in phase:
                original=next(r for r in rows if (r["task"],r["seed"],r["method"])==(row["task"],row["seed"],row["method"]))
                assert {k:v for k,v in original.items() if k!="cpu_seconds"}=={k:v for k,v in row.items() if k!="cpu_seconds"}
    expected={(t["name"],s,m) for t in plan["tasks"] for s in plan["seeds"] for m in plan["methods"]}
    assert {(r["task"],r["seed"],r["method"]) for r in rows}==expected
    summary=json.loads((analyses[0]/"summary.json").read_text())
    metrics=list(csv.DictReader((analyses[0]/"metrics.csv").open(newline="",encoding="utf-8")))
    diagnostics=[];method_table=[]
    for task in plan["tasks"]:
        name=task["name"];group=[r for r in rows if r["task"]==name]
        for seed in plan["seeds"]:
            cells=[r for r in group if r["seed"]==seed]
            assert len({r["initial_hash"] for r in cells})==1
            assert all(r["evaluation_episode_ids"]==cells[0]["evaluation_episode_ids"] for r in cells)
            streams=[[(e["agent"],e["ordinal"],e["due"]) for e in r["program_events"]] for r in cells if r["program_events"]]
            assert all(x==streams[0] for x in streams)
        for method in plan["methods"]:
            cells=[r for r in group if r["method"]==method]
            cs=[curve(r,plan) for r in cells]
            method_table.append(dict(task=name,method=method,native_auc=float(np.mean([auc(c,"native") for c in cs])),
                final_return=float(np.mean([c[-1]["value"] for c in cs])),
                bytes_mean=float(np.mean([c[-1]["bytes"] for c in cs])),messages_mean=float(np.mean([c[-1]["messages"] for c in cs])),
                work_proxy_mean=float(np.mean([work(r,plan) for r in cells])),cpu_seconds_mean=float(np.mean([r["cpu_seconds"] for r in cells]))))
            for row,c in zip(cells,cs):
                old=next(x for x in metrics if (x["task"],int(x["seed"]),x["method"])==(name,row["seed"],method))
                assert abs(auc(c,"native")-float(old["native_auc"]))<1e-12
        for row in (r for r in group if r["method"] in ("current_value","full_trace","no_price")):
            eligible=[e for e in row["ledger"] if e["ready"]]
            for e in row["ledger"]:
                if not e["control"]:continue
                agents=sorted(p["agent"] for p in e["proposals"])
                assert e["coefficients"]==[[a[i] for i in agents] for a in e["candidate_actions"]]
                for i,index in zip(agents,e["basis_indices"]):
                    assert e["candidate_actions"][index]==[float(j==i) for j in range(len(e["action"]))]
            tail=[float(np.mean(e["diagnostic"]["selected_contrasts"])) for e in row["ledger"]]
            predicted=[e["diagnostic"]["selected_training_mean"] for e in row["ledger"]]
            risks=[e["control"]["expected_risk_feedback"][e["control"]["probe_index"]] for e in eligible]
            ages=[age for e in eligible for age in e["age"]]
            qs=np.array([e["control"]["queues_before"] for e in row["ledger"]])
            raw_max_changed=sum(e["control"]["candidate_index"]!=int(np.argmax(e["control"]["normalized_score_vector"])) for e in row["ledger"])
            no_price=next(r for r in group if r["seed"]==row["seed"] and r["method"]=="no_price")
            same_actions=all(e["action"]==p["action"] and e["committed_policy"]==p["committed_policy"] for e,p in zip(row["ledger"],no_price["ledger"]))
            diagnostics.append(dict(task=name,seed=row["seed"],method=row["method"],
                eligible_frames=len(eligible),nonidle_frames=sum(any(e["action"]) for e in eligible),
                selected_training_gain_sum=float(sum(predicted)),independent_gain_sum=float(sum(tail)),
                selection_optimism_sum=float(sum(predicted)-sum(tail)),positive_tail_frames=sum(x>0 for x in tail),
                selected_alpha_counts=dict(Counter(str(a) for e in eligible for a in e["action"])),
                probe_counts=dict(Counter(str(e["probe_episodes"]) for e in eligible)),
                geometry_counts=dict(Counter(e["control"]["geometry"] for e in eligible)),
                candidate_count_max=max(len(e["candidate_hashes"]) for e in eligible),
                mean_selected_expected_risk_feedback=float(np.mean(risks)),
                mean_joint_version_age=float(np.mean(ages)),max_joint_version_age=max(ages),
                maximum_queue=qs.max(axis=0).tolist(),frames_with_positive_queue=int((qs.max(axis=1)>0).sum()),
                priced_choice_differs_from_unpriced_argmax=raw_max_changed,
                policy_action_path_identical_to_no_price=same_actions,
                extra_probe_bytes=row["charges"]["probe_bytes"],tail_probe_bytes=row["charges"]["diagnostic_probe_bytes"],
                tensor_copy_bytes=row["charges"]["state_copy_bytes"],probe_forward_rows=row["charges"]["score_rows"],
                tail_forward_rows=row["charges"]["diagnostic_score_rows"]))
    for item in summary["tasks"]:
        table=[r for r in method_table if r["task"]==item["task"]]
        best=max((r for r in table if r["method"] in plan["controls"]),key=lambda r:r["native_auc"])
        full=next(r for r in table if r["method"]=="current_value")
        assert best["method"]==item["strongest_control"]
        assert abs(100*(full["native_auc"]-best["native_auc"])/abs(best["native_auc"])-item["native_auc_gain_percent"])<1e-12
        assert abs(100*(full["final_return"]-best["final_return"])/abs(best["final_return"])-item["final_return_change_percent"])<1e-12
        assert not all(item["gates"].values())
    assert summary["decision"]==complete["decision"]=="NO_GO"
    report=dict(decision="NO_GO",source_commit=execution["source_commit"],source_lock_sha256=sha(lockpath),
        verification_status="VERIFIED",native_experiment_restarted=False,cells_checked=60,registered_semantic_replays_checked=6,
        analysis_hashes=analysis_hashes,runtime_lock_verified=True,source_at_registered_commit_verified=True,
        exact_cost_ledger_checks=verified,mechanism_diagnostics=diagnostics,all_method_table=method_table,
        frozen_task_decisions=summary["tasks"],artifact_sha256=artifacts,cpu_seconds=complete["cpu_seconds"],
        native_training_transitions_science=sum(r["charges"]["training_transitions"] for r in rows),
        native_evaluation_transitions_science=sum(r["charges"]["evaluation_transitions"] for r in rows),
        native_training_transitions_replays=6*12000,native_evaluation_transitions_replays=6*3000,
        inferential_significance_claimed=False,seed_count_per_task=3,confirmation_permitted=False,
        manuscript_admission=False,automatic_successor_preregistration_permitted=False)
    dump(OUT/"validation.json",report)
    with (OUT/"all_method_table.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(method_table[0]),lineterminator="\n");writer.writeheader();writer.writerows(method_table)
    print(json.dumps({k:report[k] for k in ("decision","cells_checked","registered_semantic_replays_checked","cpu_seconds")}))


if __name__=="__main__":main()
