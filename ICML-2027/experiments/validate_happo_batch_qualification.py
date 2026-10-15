"""Integrity-only validator for the disjoint operational qualification run."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLAN = HERE / "happo_batch_cpu_preregistration.json"
RUNNER = HERE / "happo_batch_cpu.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate(root):
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    # These are the deliberately reduced operational settings in config(smoke=True).
    blocks, train_episodes, eval_every, eval_episodes = 12, 4, 3, 4
    search_episodes, actor_steps, critic_steps = 4, 1, 1
    horizon = plan["episode_length"]
    seed = plan["qualification_seed"]
    expected = {(task["name"], method) for task in plan["tasks"] for method in plan["methods"]}
    actual = set()
    checks = []
    initial_hashes = {}
    for task in plan["tasks"]:
        for method in plan["methods"]:
            folder = root / f"{task['name']}_{seed}_{method}"
            actual.add((task["name"], method))
            meta = json.loads((folder / "scientific.json").read_text(encoding="utf-8"))
            curves, decisions = read_csv(folder / "curves.csv"), read_csv(folder / "decisions.csv")
            totals, n = meta["totals"], meta["agents"]
            checksum_ok = True
            for line in (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", 1)
                checksum_ok &= sha(folder / name) == digest
            item = {
                "mapping": meta["task"] == task["name"] and meta["seed"] == seed and meta["method"] == method,
                "qualification_role": meta["smoke"] is True,
                "source_current": meta["source_sha256"] == sha(RUNNER) and meta["plan_sha256"] == sha(PLAN),
                "checksums": checksum_ok,
                "finite": all(np.isfinite(float(row[key])) for row in curves for key in ("mean_return", "std_return")),
                "counts": len(curves) == blocks // eval_every + 1 and len(decisions) == blocks,
                "training_charge": totals["training_joint_transitions"] == blocks * train_episodes * horizon
                                   and totals["training_actor_transitions"] == totals["training_joint_transitions"] * n,
                "optimizer_charge": totals["actor_optimizer_rows"] == totals["packet_messages"] * train_episodes * horizon * actor_steps
                                    and totals["critic_optimizer_rows"] == blocks * train_episodes * horizon * critic_steps,
                "evaluation_charge": totals["benchmark_evaluation_joint_transitions"] == len(curves) * eval_episodes * horizon
                                     and totals["benchmark_evaluation_actor_transitions"] == totals["benchmark_evaluation_joint_transitions"] * n,
                "packet_reconstruction": totals["packet_messages"] == sum(int(row["launched_packets"]) for row in decisions),
                "trajectory_identity": len({row["train_episode_seed_hash"] for row in decisions}) == blocks,
                "controller_separation": not meta["benchmark_evaluation_used_by_selection"],
            }
            if method == "outcome_aware_oracle":
                menu_sizes = []
                for row in decisions:
                    credit = int(row["remaining_credit"]) + int(row["launched_packets"])
                    menu_sizes.append(sum(sum(x > 0 for x in action) <= credit
                                          for action in itertools.product(plan["packet_step_multipliers"], repeat=n)))
                measured = sum(row["heldout_increment_measured"] == "True" for row in decisions)
                item["oracle_charges"] = (
                    meta["oracle_search_used_by_selection"]
                    and totals["oracle_search_joint_transitions"] == sum(menu_sizes) * search_episodes * horizon
                    and totals["oracle_search_actor_transitions"] == totals["oracle_search_joint_transitions"] * n
                    and totals["oracle_counterfactual_packets"] == sum(n - int(row["launched_packets"]) for row in decisions)
                    and totals["oracle_counterfactual_optimizer_rows"]
                        == totals["oracle_counterfactual_packets"] * train_episodes * horizon * actor_steps
                    and measured == len(curves) - 1
                    and totals["oracle_diagnostic_joint_transitions"] == measured * eval_episodes * horizon
                    and totals["oracle_diagnostic_actor_transitions"] == totals["oracle_diagnostic_joint_transitions"] * n)
            else:
                item["oracle_charges"] = (not meta["oracle_search_used_by_selection"]
                                           and totals["oracle_search_joint_transitions"] == 0
                                           and totals["oracle_counterfactual_optimizer_rows"] == 0
                                           and totals["oracle_diagnostic_joint_transitions"] == 0)
            checks.append({"task": task["name"], "method": method, "checks": item})
            initial_hashes.setdefault(task["name"], set()).add(meta["initial_parameters_sha256"])
    initial_match = all(len(values) == 1 for values in initial_hashes.values())
    passed = actual == expected and initial_match and all(all(item["checks"].values()) for item in checks)
    return {"experiment_id": plan["experiment_id"], "qualification_seed": seed,
            "payloads": len(checks), "complete_mapping": actual == expected,
            "initial_parameters_match": initial_match, "checks": checks, "passed": passed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    if args.out.exists():
        raise FileExistsError("fresh validation output required")
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("payloads", "complete_mapping", "initial_parameters_match", "passed")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
