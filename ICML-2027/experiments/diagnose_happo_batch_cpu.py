"""Supplementary ledger and descriptive diagnosis; does not change frozen gates."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values):
    return float(np.mean(values))


def diagnose(root, analysis):
    p = json.loads((HERE / "happo_batch_cpu_preregistration.json").read_text())
    lock = json.loads((ROOT / "ICML-2027/internal/happo_batch_cpu_source_lock_a1.json").read_text())
    checks = {"execution_lock": json.loads((root / "execution.json").read_text())["lock"] == lock,
              "source_lock": all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                                 for name, digest in lock["files"].items())}
    records = csv_rows(analysis / "metrics.csv")
    summary = json.loads((analysis / "summary.json").read_text())
    expected = set()
    payload_checks, diagnostics = [], {}
    for task in p["tasks"]:
        name, k = task["name"], task["launch_credit_per_block"]
        for seed in p["development_seeds"] + p["holdout_seeds"]:
            for method in p["methods"]:
                if method == "outcome_aware_oracle" and seed in p["development_seeds"]:
                    continue
                key = f"{name}_{seed}_{method}"
                expected.add(key)
                folder = root / key
                decisions = csv_rows(folder / "decisions.csv")
                curves = csv_rows(folder / "curves.csv")
                meta = json.loads((folder / "scientific.json").read_text())
                n = meta["agents"]
                credit, packets, overflow, proxy_rows, previous = 0, 0, 0, 0, set()
                good = True
                cumulative = {0: 0}
                for block, d in enumerate(decisions):
                    bank_cap = n + k - 1 if method == "happo_full_cadence" else n
                    overflow += max(0, credit + k - bank_cap)
                    credit = min(bank_cap, credit + k)
                    action = json.loads(d["action"])
                    chosen = {i for i, a in enumerate(action) if a > 0}
                    cyclic = [0.0] * n
                    for j in range(k):
                        cyclic[(block * k + j) % n] = 1.0
                    good &= (int(d["block"]) == block and len(action) == n
                             and all(a in p["packet_step_multipliers"] for a in action)
                             and len(chosen) <= credit and int(d["launched_packets"]) == len(chosen)
                             and (d["differs_from_cyclic"] == "True") == (action != cyclic))
                    if method == "history_surrogate" and len(previous) == n:
                        proxy_rows += n * (len(p["packet_step_multipliers"]) - 1) * p["episodes_per_training_block"] * p["episode_length"]
                    previous |= chosen
                    credit -= len(chosen)
                    packets += len(chosen)
                    cumulative[block + 1] = packets
                    good &= credit == int(d["remaining_credit"])
                    versions = json.loads(d["source_versions"])
                    if method in ("happo_cyclic", "happo_full_cadence", "parallel_cyclic"):
                        good &= versions == [block] * n
                    else:
                        ages = np.roll(p["ages"][name], block % n)
                        good &= versions == [max(0, block - int(age)) for age in ages]
                totals = meta["totals"]
                good &= totals["packet_messages"] == packets and totals["unused_credit_overflow"] == overflow
                good &= totals["proxy_forward_rows"] == proxy_rows
                curve_ok = all(int(c["block"]) == j * p["evaluation_every_blocks"]
                               and float(c["resource_fraction"]) == int(c["block"]) / p["blocks"]
                               and int(c["packet_messages"]) == cumulative[int(c["block"])]
                               and int(c["training_joint_transitions"]) == int(c["block"]) * p["episodes_per_training_block"] * p["episode_length"]
                               for j, c in enumerate(curves))
                payload_checks.append({"payload": key, "action_credit_source_proxy_recurrence": bool(good),
                                       "curve_resource_reconstruction": curve_ok,
                                       "unique_training_identities": len({d["train_episode_seed_hash"] for d in decisions}) == p["blocks"]})
        heldout = [r for r in records if r["task"] == name and int(r["seed"]) in p["holdout_seeds"]]
        methods = {method: {metric: mean([float(r[metric]) for r in heldout if r["method"] == method])
                            for metric in ("auc", "initial_return", "final_return", "packet_messages",
                                           "training_joint_transitions", "actor_optimizer_rows", "proxy_forward_rows",
                                           "oracle_search_joint_transitions", "oracle_counterfactual_optimizer_rows",
                                           "oracle_diagnostic_joint_transitions", "packet_bytes")}
                   for method in p["methods"]}
        stale_exposure, aliases = {}, []
        for method in p["methods"]:
            ages = []
            for seed in p["holdout_seeds"]:
                for decision in csv_rows(root / f"{name}_{seed}_{method}/decisions.csv"):
                    action, versions = json.loads(decision["action"]), json.loads(decision["source_versions"])
                    ages.extend(int(decision["block"]) - v for alpha, v in zip(action, versions) if alpha > 0)
            stale_exposure[method] = {"launched_packets": len(ages), "mean_source_age": mean(ages) if ages else 0.0,
                                      "strictly_stale_packet_fraction": mean(np.array(ages) > 0) if ages else 0.0}
        for left, right in itertools.combinations(p["methods"], 2):
            equal = all(np.array_equal(
                np.asarray([[float(r[k]) for k in ("resource_fraction", "mean_return", "std_return")]
                            for r in csv_rows(root / f"{name}_{seed}_{left}/curves.csv")]),
                np.asarray([[float(r[k]) for k in ("resource_fraction", "mean_return", "std_return")]
                            for r in csv_rows(root / f"{name}_{seed}_{right}/curves.csv")]))
                        for seed in p["holdout_seeds"])
            if equal:
                aliases.append([left, right])
        per_seed = []
        for seed in p["holdout_seeds"]:
            oracle = csv_rows(root / f"{name}_{seed}_outcome_aware_oracle/decisions.csv")
            evaluated = [d for d in oracle if d["heldout_increment_measured"] == "True"]
            increments = [float(d["heldout_oracle_increment"]) for d in evaluated]
            actions = [json.loads(d["action"]) for d in oracle]
            per_seed.append({"seed": seed, "mean_search_margin_all_blocks": mean([float(d["oracle_search_margin"]) for d in oracle]),
                             "mean_search_margin_at_evaluation": mean([float(d["oracle_search_margin"]) for d in evaluated]),
                             "mean_independent_increment_at_evaluation": mean(increments),
                             "positive_independent_increment_fraction": mean(np.array(increments) > 0),
                             "zero_independent_increment_fraction": mean(np.array(increments) == 0),
                             "packet_utilization": sum(sum(a > 0 for a in action) for action in actions) / (p["blocks"] * k),
                             "idle_block_fraction": mean([all(a == 0 for a in action) for action in actions]),
                             "step_multiplier_counts": {str(alpha): sum(a == alpha for action in actions for a in action)
                                                         for alpha in p["packet_step_multipliers"]},
                             "checkpoint_increments": increments})
        diagnostics[name] = {"holdout_method_means": methods, "oracle_search_transfer": per_seed,
                             "stale_exposure": stale_exposure, "identical_holdout_curve_pairs": aliases,
                             "interpretation": "Checkpoint increments use independent evaluation episodes but only six checkpoints per seed; diagnostic, not a significance test or proof about all candidates."}
    checks["exact_directory_mapping"] = {d.name for d in root.iterdir() if d.is_dir()} == expected
    checks["supplementary_ledger"] = all(all(v for k, v in d.items() if k != "payload") for d in payload_checks)
    return {"experiment_id": p["experiment_id"], "source_commit": "65bb5ca4a2895384c005e52f980da718edf2e214",
            "checks": checks, "payload_checks": payload_checks, "diagnostics": diagnostics,
            "frozen_analyzer_integrity_pass": summary["integrity_pass"],
            "frozen_scientific_gates_pass": summary["stage_a_scientific_gates_pass"],
            "scope": "Descriptive four-seed feasibility diagnosis. The adaptive outcome-aware menu is not an upper bound on possible allocation value."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh diagnosis file required")
    report = diagnose(args.root, args.analysis)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checks": report["checks"], "diagnostics": report["diagnostics"]}, indent=2))
