"""Analyze the TSP-V3 calibrated-catalogue MaMuJoCo development crossover."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _tsp_experiments() -> Path:
    path = Path(__file__).resolve().parents[2] / "TSP" / "experiments"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


_tsp_experiments()
from analyze_mappo_probe_commit import collect_records  # noqa: E402


def relative_gain(value: float, reference: float) -> float:
    return float((value - reference) / max(abs(reference), 1e-12))


def evaluate_records(
    records: pd.DataFrame, config: dict, *, replay_identical: bool
) -> tuple[dict, pd.DataFrame]:
    fixed_methods = list(config["fixed_methods"])
    controller_method = str(config["controller_method"])
    seeds = list(config["seed_registry"]["development_training"])
    couplings = list(config["coupling_regimes"])
    task = f'{config["task"]["scenario"]}/{config["task"]["agent_conf"]}'
    expected = {
        (coupling, method, seed)
        for coupling in couplings
        for method in fixed_methods + [controller_method]
        for seed in seeds
    }
    records = records[
        records.task.eq(task)
        & records.method.isin(fixed_methods + [controller_method])
        & records.coupling.isin(couplings)
        & records.training_seed.isin(seeds)
    ].copy()
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

    cell_rows: list[dict] = []
    mixture_gain = float("nan")
    strong_method = None
    independent_selection = float("nan")
    shared_selection = float("nan")
    max_probe_fraction = float("inf")
    max_overhead = float("inf")
    near_envelope = False
    if complete:
        controller = records[records.method.eq(controller_method)]
        independent_selection = float(
            controller[controller.coupling.eq("independent")].selected_q.eq(8).mean()
        )
        shared_selection = float(
            controller[controller.coupling.eq("shared")].selected_q.eq(2).mean()
        )
        fixed_overall = {
            method: float(records[records.method.eq(method)].return_auc.mean())
            for method in fixed_methods
        }
        strong_method = max(
            fixed_methods, key=lambda method: (fixed_overall[method], method)
        )
        controller_overall = float(controller.return_auc.mean())
        mixture_gain = relative_gain(controller_overall, fixed_overall[strong_method])
        means = records.groupby(["coupling", "method"], as_index=False).return_auc.mean()
        for coupling in couplings:
            values = {
                row.method: float(row.return_auc)
                for row in means[means.coupling.eq(coupling)].itertuples(index=False)
            }
            envelope = max(
                fixed_methods, key=lambda method: (values[method], method)
            )
            cell_rows.append(
                {
                    "coupling": coupling,
                    **{method: values[method] for method in fixed_methods + [controller_method]},
                    "fixed_envelope_method": envelope,
                    "controller_relative_to_envelope": relative_gain(
                        values[controller_method], values[envelope]
                    ),
                }
            )
        near_envelope = all(
            row["controller_relative_to_envelope"] >= -0.02 for row in cell_rows
        )
        max_probe_fraction = float(controller.probe_message_fraction.max())
        max_overhead = float(controller.selection_overhead_fraction.max())

    thresholds = config["mandatory_development_gates"]
    gates = {
        "finite_complete_exact_accounting_and_clean_upstream": valid,
        "independent_selected_q8_fraction_min": bool(
            np.isfinite(independent_selection)
            and independent_selection
            >= float(thresholds["independent_selected_q8_fraction_min"])
        ),
        "shared_selected_q2_fraction_min": bool(
            np.isfinite(shared_selection)
            and shared_selection >= float(thresholds["shared_selected_q2_fraction_min"])
        ),
        "controller_within_two_percent_of_regime_fixed_q_envelope_each_regime": bool(
            near_envelope
        ),
        "controller_relative_auc_over_strong_single_fixed_q_min": bool(
            np.isfinite(mixture_gain)
            and mixture_gain
            >= float(
                thresholds["controller_relative_auc_over_strong_single_fixed_q_min"]
            )
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
        "baseline_source_experiment_id": config["baseline_source"]["experiment_id"],
        "development_data_reused_only_for_development": True,
        "run_count": len(records),
        "expected_run_count": len(expected),
        "strong_single_fixed_q": strong_method,
        "independent_selected_q8_fraction": independent_selection,
        "shared_selected_q2_fraction": shared_selection,
        "controller_relative_auc_over_strong_single_fixed_q": mixture_gain,
        "max_probe_message_fraction": max_probe_fraction,
        "max_selection_overhead_fraction": max_overhead,
        "gates": gates,
        "pass": passed,
        "decision": (
            "authorize-fresh-seed-confirmation-preregistration" if passed else "stop"
        ),
    }
    return result, pd.DataFrame(cell_rows)


def evaluate(
    fixed_root: Path,
    controller_root: Path,
    config_path: Path,
    *,
    replay_identical: bool = False,
) -> tuple[dict, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fixed_records, _ = collect_records(
        fixed_root, config, int(config["auc_grid_points"])
    )
    controller_records, _ = collect_records(
        controller_root, config, int(config["auc_grid_points"])
    )
    fixed_records = fixed_records[
        fixed_records.experiment_id.eq(config["baseline_source"]["experiment_id"])
        & fixed_records.method.isin(config["fixed_methods"])
    ]
    controller_records = controller_records[
        controller_records.experiment_id.eq(config["experiment_id"])
        & controller_records.method.eq(config["controller_method"])
    ]
    return evaluate_records(
        pd.concat([fixed_records, controller_records], ignore_index=True),
        config,
        replay_identical=replay_identical,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "experiments" / "marl_mamujoco_calibrated_development.json",
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--replay-identical", action="store_true")
    args = parser.parse_args()
    result, cells = evaluate(
        args.fixed_root,
        args.controller_root,
        args.config,
        replay_identical=args.replay_identical,
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.summary_csv, index=False)
    args.gate_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": result["decision"], "runs": result["run_count"]}))


if __name__ == "__main__":
    main()

