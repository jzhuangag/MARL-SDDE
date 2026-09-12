"""Analyze the frozen TSP-MARL-CONF-001 confirmation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_mappo_probe_commit as common


def one_sided_lower_t(values: np.ndarray, critical: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired estimand must be a finite vector with at least two values")
    if critical <= 0:
        raise ValueError("critical value must be positive")
    return float(values.mean() - critical * values.std(ddof=1) / np.sqrt(len(values)))


def evaluate_confirmation(root: Path, config_path: Path, audit_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records, _ = common.collect_records(
        root, config, checkpoints=int(config["evaluation"]["auc_grid_points"])
    )
    seeds = config["seed_registry"]["confirmation_training"]
    expected = {
        (coupling, method, seed)
        for coupling in config["coupling_regimes"]
        for method in config["methods"]
        for seed in seeds
    }
    observed = list(
        records[["coupling", "method", "training_seed"]].itertuples(
            index=False, name=None
        )
    )
    complete = len(observed) == len(set(observed)) and set(observed) == expected
    finite = bool(np.isfinite(records["return_auc"]).all())
    exact = bool(records["accounting_ok"].all())
    clean = bool((~records["upstream_modified"]).all())
    pinned = bool((records["harl_commit"] == config["upstream"]["commit"]).all())

    controller = records[records["method"] == "lyapunov_probe_commit"]
    metrics = {
        "independent_selected_q8_fraction": None,
        "shared_selected_q1_fraction": None,
        "independent_controller_vs_q8_mean": None,
        "independent_controller_vs_q8_lower": None,
        "shared_controller_vs_q8_mean": None,
        "shared_controller_vs_q8_lower": None,
        "shared_controller_vs_q1_mean": None,
        "shared_controller_vs_q1_lower": None,
        "mixture_controller_vs_q8_mean": None,
        "mixture_controller_vs_q8_lower": None,
        "maximum_probe_message_fraction": None,
        "maximum_selection_overhead_fraction": None,
    }
    if complete:
        metrics["independent_selected_q8_fraction"] = float(
            (
                controller[controller["coupling"] == "independent"]["selected_q"]
                == 8
            ).mean()
        )
        metrics["shared_selected_q1_fraction"] = float(
            (controller[controller["coupling"] == "shared"]["selected_q"] == 1).mean()
        )
        paired = records.pivot(
            index=["coupling", "training_seed"],
            columns="method",
            values="return_auc",
        )
        paired["relative_vs_q8"] = (
            paired["lyapunov_probe_commit"] - paired["fixed_q8"]
        ) / paired["fixed_q8"].abs().clip(lower=1e-12)
        paired["relative_vs_q1"] = (
            paired["lyapunov_probe_commit"] - paired["fixed_q1"]
        ) / paired["fixed_q1"].abs().clip(lower=1e-12)
        independent_q8 = paired.xs("independent")["relative_vs_q8"].sort_index()
        shared_q8 = paired.xs("shared")["relative_vs_q8"].sort_index()
        shared_q1 = paired.xs("shared")["relative_vs_q1"].sort_index()
        mixture_q8 = (independent_q8 + shared_q8) / 2.0
        critical = float(config["inference"]["student_t_one_sided_critical"])
        for name, values in (
            ("independent_controller_vs_q8", independent_q8),
            ("shared_controller_vs_q8", shared_q8),
            ("shared_controller_vs_q1", shared_q1),
            ("mixture_controller_vs_q8", mixture_q8),
        ):
            metrics[f"{name}_mean"] = float(values.mean())
            metrics[f"{name}_lower"] = one_sided_lower_t(
                values.to_numpy(float), critical
            )
        metrics["maximum_probe_message_fraction"] = float(
            controller["probe_message_fraction"].max()
        )
        metrics["maximum_selection_overhead_fraction"] = float(
            controller["selection_overhead_fraction"].max()
        )

    threshold = config["mandatory_confirmation_gates"]

    def at_least(metric: str, gate: str) -> bool:
        return metrics[metric] is not None and metrics[metric] >= float(threshold[gate])

    gates = {
        "finite_complete_exact_accounting_and_clean_upstream": bool(
            complete and finite and exact and clean and pinned
        ),
        "independent_selected_q8_fraction_min": at_least(
            "independent_selected_q8_fraction", "independent_selected_q8_fraction_min"
        ),
        "shared_selected_q1_fraction_min": at_least(
            "shared_selected_q1_fraction", "shared_selected_q1_fraction_min"
        ),
        "independent_controller_vs_q8_lower_confidence_bound_min": at_least(
            "independent_controller_vs_q8_lower",
            "independent_controller_vs_q8_lower_confidence_bound_min",
        ),
        "shared_controller_vs_q8_lower_confidence_bound_min": at_least(
            "shared_controller_vs_q8_lower",
            "shared_controller_vs_q8_lower_confidence_bound_min",
        ),
        "shared_controller_vs_q8_mean_min": at_least(
            "shared_controller_vs_q8_mean", "shared_controller_vs_q8_mean_min"
        ),
        "shared_controller_vs_q1_lower_confidence_bound_min": at_least(
            "shared_controller_vs_q1_lower",
            "shared_controller_vs_q1_lower_confidence_bound_min",
        ),
        "mixture_controller_vs_q8_lower_confidence_bound_min": at_least(
            "mixture_controller_vs_q8_lower",
            "mixture_controller_vs_q8_lower_confidence_bound_min",
        ),
        "mixture_controller_vs_q8_mean_min": at_least(
            "mixture_controller_vs_q8_mean", "mixture_controller_vs_q8_mean_min"
        ),
        "probe_message_fraction_max": metrics["maximum_probe_message_fraction"]
        is not None
        and metrics["maximum_probe_message_fraction"]
        <= float(threshold["probe_message_fraction_max"]),
        "controller_wall_clock_overhead_fraction_max": metrics[
            "maximum_selection_overhead_fraction"
        ]
        is not None
        and metrics["maximum_selection_overhead_fraction"]
        <= float(threshold["controller_wall_clock_overhead_fraction_max"]),
        "marginal_preserving_coupling_audit": bool(audit.get("pass")),
    }
    passed = all(gates.values())
    return {
        "experiment_id": config["experiment_id"],
        "run_count": len(records),
        "expected_run_count": len(expected),
        **metrics,
        "gates": gates,
        "all_mandatory_gates_pass": passed,
        "decision": "admit-confirmed-return-evidence" if passed else "stop",
        "records": records.to_dict(orient="records"),
    }


def main() -> None:
    tsp_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=tsp_root / "experiments" / "marl_probe_commit_confirmation.json",
    )
    parser.add_argument(
        "--audit", type=Path, default=tsp_root / "internal" / "marl_coupling_audit.json"
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, curves = common.collect_records(
        args.root, config, checkpoints=int(config["evaluation"]["auc_grid_points"])
    )
    summary = common.aggregate_curves(
        curves, checkpoints=int(config["evaluation"]["auc_grid_points"])
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    common.plot(summary, args.figure)
    gate = evaluate_confirmation(args.root, args.config, args.audit)
    args.gate_json.parent.mkdir(parents=True, exist_ok=True)
    args.gate_json.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": gate["decision"], "runs": gate["run_count"]}))


if __name__ == "__main__":
    main()
