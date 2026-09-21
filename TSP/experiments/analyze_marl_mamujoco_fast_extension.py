"""Analyze the frozen MaMuJoCo fast-extension development experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_mappo_probe_commit import collect_records


def relative_gain(value: float, reference: float) -> float:
    return float((value - reference) / max(abs(reference), 1e-12))


def evaluate(
    root: Path, config_path: Path, *, replay_identical: bool = False
) -> tuple[dict, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records, _ = collect_records(root, config, int(config["auc_grid_points"]))
    methods = list(config["methods"])
    seeds = list(config["seed_registry"]["development_training"])
    task_name = f'{config["task"]["scenario"]}/{config["task"]["agent_conf"]}'
    records = records[
        records.experiment_id.eq(config["experiment_id"])
        & records.task.eq(task_name)
        & records.method.isin(methods)
        & records.coupling.isin(config["coupling_regimes"])
        & records.training_seed.isin(seeds)
    ].copy()

    expected = {
        (coupling, method, seed)
        for coupling in config["coupling_regimes"]
        for method in methods
        for seed in seeds
    }
    observed = list(
        records[["coupling", "method", "training_seed"]].itertuples(
            index=False, name=None
        )
    )
    complete = len(observed) == len(set(observed)) and set(observed) == expected
    valid = bool(
        complete
        and np.isfinite(records.return_auc.to_numpy(float)).all()
        and records.accounting_ok.all()
        and (~records.upstream_modified).all()
        and records.harl_commit.eq(config["upstream"]["harl_commit"]).all()
    )

    fixed_methods = [method for method in methods if method.startswith("fixed_q")]
    controller_name = "lyapunov_probe_commit"
    cell_rows = []
    near_envelope = False
    mixture_gain = float("nan")
    strong_method = None
    max_probe_fraction = float("inf")
    max_overhead = float("inf")
    if complete:
        means = records.groupby(["coupling", "method"], as_index=False).return_auc.mean()
        fixed_overall = {
            method: float(records[records.method.eq(method)].return_auc.mean())
            for method in fixed_methods
        }
        strong_method = max(fixed_methods, key=lambda method: (fixed_overall[method], method))
        controller_overall = float(
            records[records.method.eq(controller_name)].return_auc.mean()
        )
        mixture_gain = relative_gain(controller_overall, fixed_overall[strong_method])
        for coupling in config["coupling_regimes"]:
            frame = means[means.coupling.eq(coupling)]
            values = {
                row.method: float(row.return_auc) for row in frame.itertuples(index=False)
            }
            envelope_method = max(
                fixed_methods, key=lambda method: (values[method], method)
            )
            cell_rows.append(
                {
                    "coupling": coupling,
                    **{method: values[method] for method in methods},
                    "fixed_envelope_method": envelope_method,
                    "controller_relative_to_envelope": relative_gain(
                        values[controller_name], values[envelope_method]
                    ),
                }
            )
        threshold = -0.02
        near_envelope = all(
            row["controller_relative_to_envelope"] >= threshold for row in cell_rows
        )
        controller = records[records.method.eq(controller_name)]
        max_probe_fraction = float(controller.probe_message_fraction.max())
        max_overhead = float(controller.selection_overhead_fraction.max())

    thresholds = config["mandatory_development_gates"]
    gates = {
        "finite_complete_exact_accounting_and_clean_upstream": valid,
        "full_fixed_q_catalogue": bool(complete),
        "controller_within_two_percent_of_regime_fixed_q_envelope_each_regime": bool(
            near_envelope
        ),
        "controller_relative_auc_over_strong_single_fixed_q_min": bool(
            np.isfinite(mixture_gain)
            and mixture_gain
            >= float(thresholds["controller_relative_auc_over_strong_single_fixed_q_min"])
        ),
        "probe_message_fraction_max": bool(
            max_probe_fraction <= float(thresholds["probe_message_fraction_max"])
        ),
        "selection_overhead_fraction_max": bool(
            max_overhead <= float(thresholds["selection_overhead_fraction_max"])
        ),
        "analysis_replay_byte_identical": bool(replay_identical),
    }
    passed = all(gates.values())
    result = {
        "experiment_id": config["experiment_id"],
        "run_count": len(records),
        "expected_run_count": len(expected),
        "strong_single_fixed_q": strong_method,
        "controller_relative_auc_over_strong_single_fixed_q": mixture_gain,
        "max_probe_message_fraction": max_probe_fraction,
        "max_selection_overhead_fraction": max_overhead,
        "gates": gates,
        "pass": passed,
        "decision": "authorize-confirmation-preregistration" if passed else "stop",
    }
    return result, pd.DataFrame(cell_rows)


def main() -> None:
    tsp = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=tsp / "experiments" / "marl_mamujoco_fast_extension.json",
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
