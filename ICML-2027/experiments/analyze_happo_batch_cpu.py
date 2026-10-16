"""Frozen analyzer for the end-to-end HAPPO allocation feasibility screen."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLAN = HERE / "happo_batch_cpu_preregistration.json"


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def analyze(root, out, replay_root=None):
    p = json.loads(PLAN.read_text())
    source_sha = hashlib.sha256((HERE / "happo_batch_cpu.py").read_bytes()).hexdigest()
    plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    if out.exists():
        raise FileExistsError("fresh analysis output required")
    out.mkdir(parents=True)
    rows, payloads, payload_checks = [], {}, []
    for task in p["tasks"]:
        name = task["name"]
        for seed in p["development_seeds"] + p["holdout_seeds"]:
            for method in p["methods"]:
                if method == "outcome_aware_oracle" and seed in p["development_seeds"] and not p["oracle_development_runs"]:
                    continue
                folder = root / f"{name}_{seed}_{method}"
                meta = json.loads((folder / "scientific.json").read_text())
                curves = read_csv(folder / "curves.csv")
                decisions = read_csv(folder / "decisions.csv")
                for line in (folder / "SHA256SUMS").read_text().splitlines():
                    expected, path = line.split("  ", 1)
                    if hashlib.sha256((folder / path).read_bytes()).hexdigest() != expected:
                        raise AssertionError(f"checksum failed: {folder / path}")
                x = np.array([float(r["resource_fraction"]) for r in curves])
                y = np.array([float(r["mean_return"]) for r in curves])
                auc = float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))
                totals = meta["totals"]
                expected_joint = p["blocks"] * p["episodes_per_training_block"] * p["episode_length"]
                row_work = p["episodes_per_training_block"] * p["episode_length"] * p["actor_optimizer_steps_per_packet"]
                checks = dict(
                    mapping=meta["task"] == name and meta["seed"] == seed and meta["method"] == method,
                    finite=bool(np.isfinite(y).all() and np.isfinite(auc)),
                    curve_count=len(curves) == p["blocks"] // p["evaluation_every_blocks"] + 1,
                    decision_count=len(decisions) == p["blocks"],
                    no_evaluation_leakage=not meta["benchmark_evaluation_used_by_selection"],
                    oracle_role=meta["oracle_search_used_by_selection"] == (method == "outcome_aware_oracle"),
                    source_locked=meta["source_sha256"] == source_sha and meta["plan_sha256"] == plan_sha and not meta["smoke"],
                    exact_joint=totals["training_joint_transitions"] == expected_joint,
                    exact_actor=totals["training_actor_transitions"] == expected_joint * meta["agents"],
                    exact_packet_rows=totals["actor_optimizer_rows"] == totals["packet_messages"] * row_work,
                    packet_cap=totals["packet_messages"] <= meta["max_training_packet_messages"],
                    reconstructed_packets=sum(int(r["launched_packets"]) for r in decisions) == totals["packet_messages"],
                    exact_critic_rows=totals["critic_optimizer_rows"] == expected_joint * p["critic_optimizer_steps_per_block"],
                    exact_eval=totals["benchmark_evaluation_joint_transitions"] == len(curves) * p["evaluation_episodes"] * p["episode_length"],
                    exact_eval_actor=totals["benchmark_evaluation_actor_transitions"] == totals["benchmark_evaluation_joint_transitions"] * meta["agents"],
                )
                if method != "outcome_aware_oracle":
                    checks["no_oracle_extra_work"] = all(totals[key] == 0 for key in (
                        "oracle_search_joint_transitions", "oracle_counterfactual_optimizer_rows",
                        "oracle_diagnostic_joint_transitions"))
                else:
                    checked = sum(r["heldout_increment_measured"] == "True" for r in decisions)
                    menu_sizes = []
                    for decision in decisions:
                        available_credit = int(decision["remaining_credit"]) + int(decision["launched_packets"])
                        menu_sizes.append(sum(sum(x > 0 for x in action) <= available_credit
                                              for action in itertools.product(p["packet_step_multipliers"],
                                                                              repeat=meta["agents"])))
                    checks["diagnostic_count"] = checked == len(curves) - 1
                    checks["diagnostic_charge"] = (totals["oracle_diagnostic_joint_transitions"]
                                                    == checked * p["evaluation_episodes"] * p["episode_length"]
                                                    and totals["oracle_diagnostic_actor_transitions"]
                                                    == totals["oracle_diagnostic_joint_transitions"] * meta["agents"])
                    checks["oracle_search_charge"] = (totals["oracle_search_joint_transitions"]
                                                       == sum(menu_sizes) * p["oracle_search_episodes"]
                                                       * p["episode_length"]
                                                       and totals["oracle_search_actor_transitions"]
                                                       == totals["oracle_search_joint_transitions"] * meta["agents"])
                    checks["counterfactual_charge"] = (totals["oracle_counterfactual_packets"]
                                                         == sum(meta["agents"] - int(r["launched_packets"])
                                                                for r in decisions)
                                                         and totals["oracle_counterfactual_optimizer_rows"]
                                                         == totals["oracle_counterfactual_packets"] * row_work)
                if method in ("happo_cyclic", "happo_full_cadence", "parallel_cyclic", "stale_cyclic", "age_first"):
                    checks["fixed_budget_exhausted"] = totals["packet_messages"] == meta["max_training_packet_messages"]
                payload_checks.append(dict(task=name, seed=seed, method=method, checks=checks))
                record = dict(task=name, seed=seed, method=method, auc=auc, final_return=float(y[-1]),
                              initial_return=float(y[0]), final_gain_from_initial=float(y[-1] - y[0]),
                              noncyclic_fraction=float(np.mean([r["differs_from_cyclic"] == "True" for r in decisions])),
                              **totals)
                rows.append(record)
                payloads[name, seed, method] = record
    controls = [m for m in p["methods"] if m != "outcome_aware_oracle"]
    selected = {task["name"]: max(controls, key=lambda m: np.mean([
        payloads[task["name"], seed, m]["auc"] for seed in p["development_seeds"]])) for task in p["tasks"]}
    metrics, scientific_gates = {}, {}
    threshold = p["mandatory_gates"]
    for task in p["tasks"]:
        name = task["name"]
        task_metrics = {}
        for comparator in sorted(set((selected[name], "happo_cyclic"))):
            auc_gains, final_gains = [], []
            for seed in p["holdout_seeds"]:
                oracle = payloads[name, seed, "outcome_aware_oracle"]
                baseline = payloads[name, seed, comparator]
                auc_gains.append((oracle["auc"] - baseline["auc"]) / max(abs(baseline["auc"]), 1.0))
                final_gains.append((oracle["final_return"] - baseline["final_return"]) / max(abs(baseline["final_return"]), 1.0))
            task_metrics[comparator] = dict(auc_relative_gains=auc_gains, mean_auc_relative_gain=float(np.mean(auc_gains)),
                                            positive_seed_fraction=float(np.mean(np.array(auc_gains) > 0)),
                                            mean_final_return_relative_change=float(np.mean(final_gains)))
            prefix = name + ":" + comparator
            scientific_gates[prefix + ":auc"] = np.mean(auc_gains) >= threshold["oracle_holdout_task_mean_auc_relative_gain_min"]
            scientific_gates[prefix + ":positive_seeds"] = np.mean(np.array(auc_gains) > 0) >= threshold["oracle_holdout_task_positive_seed_fraction_min"]
            scientific_gates[prefix + ":final_return"] = np.mean(final_gains) >= threshold["oracle_holdout_task_mean_final_return_relative_change_min"]
        noncyclic = np.mean([payloads[name, seed, "outcome_aware_oracle"]["noncyclic_fraction"] for seed in p["holdout_seeds"]])
        scientific_gates[name + ":noncyclic"] = noncyclic >= threshold["oracle_noncyclic_action_fraction_min"]
        metrics[name] = dict(selected_development_comparator=selected[name], comparisons=task_metrics,
                             oracle_noncyclic_fraction=float(noncyclic))
        learning_gain = np.mean([(payloads[name, seed, "happo_cyclic"]["final_return"] - payloads[name, seed, "happo_cyclic"]["initial_return"])
                                 / max(abs(payloads[name, seed, "happo_cyclic"]["initial_return"]), 1.0) for seed in p["holdout_seeds"]])
        scientific_gates[name + ":baseline_learning"] = learning_gain >= threshold["happo_holdout_task_mean_final_gain_from_initial_min"]
        metrics[name]["happo_mean_final_gain_from_initial"] = float(learning_gain)
    replay_checks = []
    if replay_root:
        r = p["replay"]
        for task in r["tasks"]:
            for method in r["methods"]:
                name = f"{task}_{r['seed']}_{method}"
                for file in r["files"]:
                    replay_checks.append(dict(payload=name, file=file,
                                              equal=(root / name / file).read_bytes() == (replay_root / name / file).read_bytes()))
    integrity = all(all(x["checks"].values()) for x in payload_checks)
    initial = {}
    for (task, seed, method), record in payloads.items():
        meta = json.loads((root / f"{task}_{seed}_{method}" / "scientific.json").read_text())
        initial.setdefault((task, seed), set()).add(meta["initial_parameters_sha256"])
    integrity = integrity and all(len(v) == 1 for v in initial.values())
    replay_ok = bool(replay_checks) and all(r["equal"] for r in replay_checks)
    summary = dict(experiment_id=p["experiment_id"], payloads=len(rows), integrity_pass=integrity,
                   registered_payload_replay_pass=replay_ok, metrics=metrics,
                   scientific_gates={k: bool(v) for k, v in scientific_gates.items()},
                   stage_a_scientific_gates_pass=bool(all(scientific_gates.values())),
                   stage_b_authorized=bool(integrity and replay_ok and all(scientific_gates.values())),
                   interpretation="Outcome-aware selector is an optimistic empirical screen, not a mathematical oracle upper bound or deployable evidence. Stage B remains necessary after any Stage-A pass.")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    dump(out / "summary.json", summary)
    dump(out / "integrity.json", dict(payload_checks=payload_checks, registered_replay=replay_checks))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path)
    a = parser.parse_args()
    analyze(a.root, a.out, a.replay_root)
