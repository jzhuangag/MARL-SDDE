"""Outcome-free payload audit: evaluation returns are checked only for finiteness."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def finite(x):
    if isinstance(x, float):
        assert math.isfinite(x)
    elif isinstance(x, dict):
        for value in x.values():
            finite(value)
    elif isinstance(x, list):
        for value in x:
            finite(value)


def check(row, plan):
    finite(row)
    n, b, h, f = len(row["actor_parameter_bytes"]), plan["bank_episodes"], plan["horizon"], plan["frames"]
    bh = b * h; full = row["method"] == "current_value"; fresh = row["method"] == "fresh_happo"
    assert row["evaluation_used_by_controller"] is False and row["gradient_payload_replaced_by_data_program"]
    assert row["frozen_final_hash"] == row["initial_hash"] and len(row["ledger"]) == f
    assert len(set(row["training_episode_ids"])) == f * b
    assert len(set(row["evaluation_episode_ids"])) == len(row["evaluation_episode_ids"])
    assert set(row["training_episode_ids"]).isdisjoint(row["evaluation_episode_ids"])
    eval_frames = [0] + [t for t in range(1, f+1) if t % plan["evaluation_every"] == 0 or t == f]
    assert [e["frame"] for e in row["evaluation"]] == eval_frames
    for e in row["evaluation"]:
        assert len(e["identities"]) == plan["evaluation_episodes"]
        assert e["frozen_policy_hash"] == row["initial_hash"] and e["shared_action_uniforms"]
        if e["frame"] == 0:
            assert e["trained_episode_returns"] == e["frozen_episode_returns"]
    events = {e["id"]: e for e in row["program_events"]}; seen = set(); staleness = False
    expected_totals = {k: 0 for k in row["charges"]}
    programs = (sum(d*4+16 for d in row["local_observation_dimensions"]) + n*4) * bh
    public = (row["state_dimension"] * 4 + 8) * bh; advantages = n * 4 * bh
    byte_scale = programs + public + advantages + sum(row["actor_parameter_bytes"])
    work_scale = (3*n + 1 + plan["actor_epochs"] * n + plan["critic_epochs"]) * bh
    assert byte_scale == row["communication_scale"] and work_scale == row["compute_scale"]
    q = np.zeros(2)
    for t, entry in enumerate(row["ledger"]):
        assert entry["frame"] == t and entry["receipt_after_bank"] and not entry["current_evaluation_used"]
        assert entry["versions_after"] == [v+int(a != 0) for v,a in zip(entry["source_versions"],entry["action"])]
        if t:
            assert entry["source_versions"] == row["ledger"][t-1]["versions_after"]
            assert entry["source_optimizer_hashes"] == row["ledger"][t-1]["committed_optimizer_hashes"]
        if not fresh:
            arrived = [e for e in events.values() if e["id"] not in seen and e["due"] <= t]
            assert {e["id"] for e in arrived} == set(entry["arrived"])
            latest = {}
            for event in arrived:
                seen.add(event["id"])
                assert all(v >= s for v,s in zip(entry["source_versions"],event["source_versions"]))
                age = sum(v-s for v,s in zip(entry["source_versions"],event["source_versions"]))
                staleness |= age > 0
                i = event["agent"]
                if i not in latest or latest[i]["ordinal"] < event["ordinal"]:
                    latest[i] = event
            assert set(entry["ready"]) == {e["id"] for e in latest.values()}
        proposals = entry["proposals"]
        assert len(proposals) == (n if fresh else len(entry["ready"]))
        for prop in proposals:
            assert prop["epochs"] == plan["actor_epochs"] and prop["rows"] == plan["actor_epochs"] * bh
            trace = prop["gradient_trace"]
            assert len(trace) == plan["actor_epochs"]
            assert len({s["policy_before"] for s in trace}) == plan["actor_epochs"]
            assert len({s["gradient_hash"] for s in trace}) >= 2
            assert trace[0]["policy_before"] == prop["source_server_policy"]
            assert trace[-1]["policy_after"] == prop["candidate_policy"]
            assert all(a["policy_after"] == z["policy_before"] for a,z in zip(trace,trace[1:]))
            assert prop["source_server_optimizer"] == entry["source_optimizer_hashes"][prop["agent"]]
            if not fresh:
                assert prop["packet_id"] in entry["ready"]
        for i, alpha in enumerate(entry["action"]):
            if not alpha:
                assert entry["committed_optimizer_hashes"][i] == entry["source_optimizer_hashes"][i]
            else:
                selected = next(p for p in proposals if p["agent"] == i and (full or p["alpha"] == alpha))
                assert entry["committed_optimizer_hashes"][i] == selected["candidate_optimizer"]
        s = entry["step_charges"]
        assert s["training_transitions"] == bh and s["sampling_rows"] == n*bh
        assert s["critic_value_rows"] == bh and s["critic_backward_rows"] == plan["critic_epochs"]*bh
        assert s["actor_backward_rows"] == len(proposals)*plan["actor_epochs"]*bh
        assert s["actor_adam_calls"] == len(proposals)*plan["actor_epochs"]
        assert s["source_backward_rows"] == s["gradient_bytes"] == s["gradient_messages"] == 0
        assert s["public_bytes"] == public and s["public_messages"] == 1
        assert s["advantage_bytes"] == advantages and s["advantage_messages"] == n
        assert s["program_bytes"] == programs and s["program_messages"] == n
        assert s["policy_bytes"] == sum(size for size,a in zip(row["actor_parameter_bytes"],entry["action"]) if a)
        assert s["policy_messages"] == sum(a != 0 for a in entry["action"])
        assert s["factor_forward_rows"] == (2*n*bh if fresh else 0)
        assert s["proposal_copy_bytes"] == sum(p["copied_program_bytes"] for p in proposals)
        assert s["state_copy_bytes"] == sum(p["amortized"]["modeled_state_tensor_copy_bytes"] for p in proposals)
        assert s["interpolation_entries"] == sum(p["amortized"]["interpolation_parameter_entries"] for p in proposals)
        if not fresh:
            for p in proposals:
                am = p["amortized"]
                assert am["actor_adam_calls"] == plan["actor_epochs"]
                assert am["actor_backward_rows"] == bh*plan["actor_epochs"]
                assert am["line_search_coefficients"] == (plan["alphas"] if full else [1.])
                assert not am["worker_optimizer_imported"] and am["positive_step_optimizer_follows_unit_block"]
        if full:
            m = entry["probe_episodes"]
            assert m in plan["probe_episodes"] and entry["probe_before_bank"] and entry["candidates_before_bank"]
            assert len(entry["candidate_hashes"]) == len(set(entry["candidate_hashes"]))
            d = entry["control"]
            assert d["committed_hash"] == entry["committed_policy"]
            assert entry["candidate_hashes"][d["candidate_index"]] == entry["committed_policy"]
            assert entry["candidate_actions"][d["candidate_index"]] == entry["action"]
            assert not d["selected_score_is_unbiased"] and not d["sample_variance_is_upper_certificate"]
            assert not d["hard_budget_guarantee"]
            assert d["representation_before_bank"] and entry["opened_meta"]["geometry"] == d["geometry"]
            from amortized_packet_value import expected_risk_feedback
            options = [(mm,ll,g) for mm in plan["probe_episodes"] for ll in plan["young_scales"] for g in plan["geometries"]]
            assert np.allclose(d["expected_risk_feedback"],[expected_risk_feedback(d["geometry_statistics"],g,mm,ll) for mm,ll,g in options],rtol=0,atol=1e-12)
            assert d["geometry"] == options[d["probe_index"]][2]
            assert s["geometry_entries"] == m*len(entry["candidate_hashes"])*(len(entry["ready"])+1)
            minimum_m = min(plan["probe_episodes"])
            min_bytes = minimum_m*h*sum(row["local_observation_dimensions"][events[j]["agent"]]*4+16 for j in entry["ready"])
            min_rows = (len(entry["ready"])*len(plan["alphas"])+int(s["potential_predict_rows"] > 0))*minimum_m*h
            min_rows += minimum_m*len(entry["candidate_hashes"])*(len(entry["ready"])+1)/plan["scalar_entries_per_work_unit"]
            assert d["sunk_cost"][0]+min_bytes/byte_scale < plan["communication_target"]
            assert d["sunk_cost"][1]+min_rows/work_scale < plan["compute_target"]
            eligible = [events[j]["agent"] for j in entry["ready"]]
            assert s["probe_bytes"] == m*h*sum(row["local_observation_dimensions"][i]*4+16 for i in eligible)
            assert s["probe_messages"] == len(eligible) and s["score_rows"] == len(entry["ready"])*len(plan["alphas"])*m*h
            assert s["potential_predict_rows"] in (0, m*h)
            communication = (s["public_bytes"]+s["program_bytes"]+s["advantage_bytes"]+s["probe_bytes"]+s["policy_bytes"])/byte_scale
            compute = (sum(s[k] for k in ("sampling_rows","critic_value_rows","critic_backward_rows","actor_backward_rows",
                                        "potential_fit_rows","score_rows","potential_predict_rows"))
                       +(s["state_copy_bytes"]+s["proposal_copy_bytes"]+s["program_bytes"])/plan["copy_bytes_per_work_unit"]
                       +(s["interpolation_entries"]+s["geometry_entries"])/plan["scalar_entries_per_work_unit"])/work_scale
            assert np.allclose(d["queues_before"], q, rtol=0, atol=1e-12)
            assert np.allclose(d["total_charge"], [communication, compute], rtol=0, atol=1e-12)
            q = np.maximum(0., q+np.array([communication,compute])-np.array([plan["communication_target"],plan["compute_target"]]))
            assert np.allclose(d["queues_after"],q,rtol=0,atol=1e-12)
        else:
            assert s["probe_bytes"] == s["probe_messages"] == s["score_rows"] == 0
        for k,v in s.items():
            expected_totals[k] += v
    expected_totals["evaluation_transitions"] = len(eval_frames)*plan["evaluation_episodes"]*h*2
    expected_totals["evaluation_rows"] = expected_totals["evaluation_transitions"]*n
    assert expected_totals == row["charges"]
    if not fresh:
        assert len(events) == n*f and row["reserved_program_bytes"] == row["transmitted_program_bytes"] == programs*f
        for e in events.values():
            assert e["ordinal"] == e["launch"]
            word = hashlib.sha256(f'{row["seed"]}:{e["agent"]}:{e["ordinal"]}'.encode()).digest()
            delay = plan["delay_min"] + int.from_bytes(word[:8],"big") % (plan["delay_max"]-plan["delay_min"]+1)
            assert e["due"] == e["launch"]+delay
            launch = row["ledger"][e["launch"]]
            assert e["source_versions"] == launch["source_versions"] and e["critic_version"] == e["launch"]
            assert e["data_hash"] == launch["program_hashes"][e["agent"]]
    if full:
        assert staleness
    return dict(task=row["task"],method=row["method"],seed=row["seed"],payload_pass=True,
                genuine_version_staleness=staleness, **row["charges"])


def analyze(root, replay, plan, output, dump, sha):
    root, replay, output = Path(root), Path(replay), Path(output)
    output.mkdir(parents=True,exist_ok=False)
    hashes = json.loads((root/"hashes.json").read_text())
    replay_hashes = json.loads((replay/"hashes.json").read_text())
    assert len(hashes) == plan["qualification_cells"] == len(replay_hashes)
    assert set(hashes) == set(replay_hashes)
    rows, checks = [], []
    for name, value in hashes.items():
        assert sha(root/name) == value and sha(replay/name) == replay_hashes[name]
        row = json.loads((root/name).read_text()); other = json.loads((replay/name).read_text())
        assert {k:v for k,v in row.items() if k != "cpu_seconds"} == {k:v for k,v in other.items() if k != "cpu_seconds"}
        rows.append(row); checks.append(check(row,plan)); check(other,plan)
    expected = {(t["name"], plan["seed"], m) for t in plan["tasks"] for m in plan["methods"]}
    assert {(r["task"],r["seed"],r["method"]) for r in rows} == expected
    for task in plan["tasks"]:
        group = [r for r in rows if r["task"] == task["name"]]
        assert len({r["initial_hash"] for r in group}) == 1
        # Full and eligible-all share service streams, not selection-clock ages.
        streams = [[(e["agent"],e["ordinal"],e["due"]) for e in r["program_events"]]
                   for r in group if r["method"] != "fresh_happo"]
        assert streams[0] == streams[1]
        assert all(r["evaluation_episode_ids"] == group[0]["evaluation_episode_ids"] for r in group)
    dump(output/"integrity.json",dict(cells=checks,semantic_replays=len(rows),evaluations_ranked=False))
    with (output/"resource_counts.csv").open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(checks[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(checks)
    dump(output/"decision.json",dict(decision="OPERATIONALLY_QUALIFIED",cells=len(rows),replays=len(rows),
        performance_evidence=False,precision_usefulness_established=False,
        confidence_certificate=False,manuscript_admission=False))
