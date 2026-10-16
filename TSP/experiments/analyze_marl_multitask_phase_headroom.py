"""Analyze the frozen multi-task endpoint participation-phase scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_mappo_probe_commit import collect_records


def relative_gain(value: float, reference: float) -> float:
    return float((value - reference) / max(abs(reference), 1e-12))


def _task_lookup(config: dict) -> tuple[dict[str, dict], dict[tuple[str, int], str]]:
    tasks = {task["task_id"]: task for task in config["tasks"]}
    rays = {}
    for task_id, task in tasks.items():
        for ray, budget in task["message_budgets"].items():
            rays[(task_id, int(budget))] = ray
    return tasks, rays


def evaluate(
    root: Path, config_path: Path, *, replay_identical: bool = False
) -> tuple[dict, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records, _ = collect_records(root, config, int(config["auc_grid_points"]))
    records = records[
        records.experiment_id.eq(config["experiment_id"])
        & records.method.isin(
            [f"fixed_q{q}" for q in config["fixed_q_endpoints"]]
        )
        & records.coupling.isin(config["coupling_regimes"])
        & records.training_seed.isin(config["development_seeds"])
    ].copy()

    tasks, rays = _task_lookup(config)
    task_name_to_id = {}
    for task_id, task in tasks.items():
        task_name = (
            f'{task["scenario"]}/{task["agent_conf"]}'
            if task["environment"] == "mamujoco"
            else task["scenario"]
        )
        task_name_to_id[task_name] = task_id
    records["task_id"] = records.task.map(task_name_to_id)
    records["ray"] = [
        rays.get((task_id, int(message_budget)))
        for task_id, message_budget in zip(records.task_id, records.message_budget)
    ]

    methods = [f"fixed_q{q}" for q in config["fixed_q_endpoints"]]
    expected = {
        (task["task_id"], ray, coupling, method, seed)
        for task in config["tasks"]
        for ray in task["message_budgets"]
        for coupling in config["coupling_regimes"]
        for method in methods
        for seed in config["development_seeds"]
    }
    observed = list(
        records[["task_id", "ray", "coupling", "method", "training_seed"]]
        .itertuples(index=False, name=None)
    )
    complete = len(observed) == len(set(observed)) and set(observed) == expected
    valid = bool(
        complete
        and np.isfinite(records.return_auc.to_numpy(float)).all()
        and records.accounting_ok.all()
        and (~records.upstream_modified).all()
        and records.harl_commit.eq(config["upstream"]["harl_commit"]).all()
    )

    cell_rows = []
    task_metrics = {}
    if complete:
        means = records.groupby(
            ["task_id", "ray", "coupling", "method"], as_index=False
        ).return_auc.mean()
        for task_id in tasks:
            task_frame = means[means.task_id == task_id]
            static = {
                method: float(
                    records[
                        (records.task_id == task_id) & (records.method == method)
                    ].return_auc.mean()
                )
                for method in methods
            }
            strong_method = max(methods, key=lambda method: (static[method], method))
            oracle_values = []
            oracle_actions = []
            for ray in tasks[task_id]["message_budgets"]:
                for coupling in config["coupling_regimes"]:
                    cell = task_frame[
                        (task_frame.ray == ray) & (task_frame.coupling == coupling)
                    ]
                    values = {
                        row.method: float(row.return_auc)
                        for row in cell.itertuples(index=False)
                    }
                    oracle_method = max(methods, key=lambda method: (values[method], method))
                    oracle_q = int(oracle_method.replace("fixed_q", ""))
                    oracle_value = values[oracle_method]
                    oracle_values.append(oracle_value)
                    oracle_actions.append(oracle_q)
                    cell_rows.append(
                        {
                            "task_id": task_id,
                            "ray": ray,
                            "coupling": coupling,
                            **{method: values[method] for method in methods},
                            "oracle_q": oracle_q,
                            "oracle_return_auc": oracle_value,
                        }
                    )
            oracle_mean = float(np.mean(oracle_values))
            strong_value = static[strong_method]
            task_metrics[task_id] = {
                "strong_static_method": strong_method,
                "strong_static_return_auc": strong_value,
                "oracle_mean_return_auc": oracle_mean,
                "oracle_relative_headroom": relative_gain(oracle_mean, strong_value),
                "distinct_oracle_actions": sorted(set(oracle_actions)),
            }

    cells = pd.DataFrame(cell_rows)
    thresholds = config["mandatory_stage_a_gates"]
    headroom_gate = bool(
        task_metrics
        and all(
            metric["oracle_relative_headroom"]
            >= float(thresholds["oracle_relative_headroom_over_task_static_min_each_task"])
            for metric in task_metrics.values()
        )
    )
    action_gate = bool(
        task_metrics
        and all(
            len(metric["distinct_oracle_actions"])
            >= int(thresholds["distinct_oracle_actions_min_per_task"])
            for metric in task_metrics.values()
        )
    )
    reversal_gate = False
    if not cells.empty:
        reversal_gate = all(
            int(
                cells[
                    (cells.task_id == task_id)
                    & (cells.ray == "message_binding")
                    & (cells.coupling == "independent")
                ].iloc[0].oracle_q
            )
            != int(
                cells[
                    (cells.task_id == task_id)
                    & (cells.ray == "message_binding")
                    & (cells.coupling == "shared")
                ].iloc[0].oracle_q
            )
            for task_id in tasks
        )
    gates = {
        "finite_complete_exact_accounting_and_clean_upstream": valid,
        "distinct_oracle_actions_min_per_task": action_gate,
        "oracle_relative_headroom_over_task_static_min_each_task": headroom_gate,
        "message_binding_regime_reversal_each_task": bool(reversal_gate),
        "analysis_replay_byte_identical": bool(replay_identical),
    }
    passed = all(gates.values())
    result = {
        "experiment_id": config["experiment_id"],
        "run_count": len(records),
        "expected_run_count": len(expected),
        "task_metrics": task_metrics,
        "gates": gates,
        "pass": passed,
        "decision": "authorize-full-q-and-controller-freeze" if passed else "stop",
    }
    return result, cells


def main() -> None:
    tsp = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=tsp / "experiments" / "marl_multitask_phase_headroom.json",
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--replay-identical", action="store_true")
    args = parser.parse_args()
    result, cells = evaluate(
        args.root, args.config, replay_identical=args.replay_identical
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.summary_csv, index=False)
    args.gate_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": result["decision"], "runs": result["run_count"]}))


if __name__ == "__main__":
    main()
