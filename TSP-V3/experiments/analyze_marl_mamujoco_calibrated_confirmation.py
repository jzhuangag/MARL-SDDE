"""Analyze the fresh-seed TSP-V3 calibrated-catalogue confirmation."""

from __future__ import annotations

import argparse
import json
import math
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
import analyze_mappo_probe_commit as common  # noqa: E402


def relative_gain(value: pd.Series, reference: pd.Series) -> pd.Series:
    return (value - reference) / reference.abs().clip(lower=1e-12)


def one_sided_lower_t(values: pd.Series, critical: float) -> float:
    array = values.to_numpy(float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("paired estimand must be a finite vector")
    if critical <= 0:
        raise ValueError("critical value must be positive")
    return float(array.mean() - critical * array.std(ddof=1) / np.sqrt(len(array)))


def exact_one_sided_sign_p(successes: int, trials: int) -> float:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid sign-test counts")
    return float(
        sum(math.comb(trials, k) for k in range(successes, trials + 1))
        / (2**trials)
    )


def attach_final_returns(records: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, curve in curves.groupby(["coupling", "method", "training_seed"]):
        curve = curve.sort_values("budget_fraction")
        value = float(
            np.interp(
                1.0,
                curve["budget_fraction"].to_numpy(float),
                curve["team_return"].to_numpy(float),
            )
        )
        rows.append(
            {
                "coupling": keys[0],
                "method": keys[1],
                "training_seed": keys[2],
                "final_return": value,
            }
        )
    return records.merge(
        pd.DataFrame(rows),
        on=["coupling", "method", "training_seed"],
        how="left",
        validate="one_to_one",
    )


def evaluate_records(
    records: pd.DataFrame, config: dict, *, replay_identical: bool
) -> dict:
    methods = list(config["methods"])
    controller_method = str(config["controller_method"])
    seeds = list(config["seed_registry"]["confirmation_training"])
    couplings = list(config["coupling_regimes"])
    task = f'{config["task"]["scenario"]}/{config["task"]["agent_conf"]}'
    expected = {
        (coupling, method, seed)
        for coupling in couplings
        for method in methods
        for seed in seeds
    }
    records = records[
        records.experiment_id.eq(config["experiment_id"])
        & records.task.eq(task)
        & records.method.isin(methods)
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
        and np.isfinite(records[["return_auc", "final_return"]].to_numpy(float)).all()
        and records.accounting_ok.all()
        and (~records.upstream_modified).all()
        and records.harl_commit.eq(config["upstream"]["harl_commit"]).all()
    )

    metrics = {
        "independent_selected_q8_fraction": None,
        "shared_selected_q2_fraction": None,
        "independent_vs_q8_mean": None,
        "independent_vs_q8_lower": None,
        "shared_vs_q2_mean": None,
        "shared_vs_q2_lower": None,
        "mixture_vs_q8_mean": None,
        "mixture_vs_q8_lower": None,
        "mixture_vs_q8_positive_pairs": None,
        "mixture_vs_q8_sign_p": None,
        "mixture_final_return_vs_q8_mean": None,
        "maximum_probe_message_fraction": None,
        "maximum_selection_overhead_fraction": None,
    }
    paired_rows: list[dict] = []
    if complete:
        controller = records[records.method.eq(controller_method)]
        metrics["independent_selected_q8_fraction"] = float(
            controller[controller.coupling.eq("independent")].selected_q.eq(8).mean()
        )
        metrics["shared_selected_q2_fraction"] = float(
            controller[controller.coupling.eq("shared")].selected_q.eq(2).mean()
        )
        auc = records.pivot(
            index=["coupling", "training_seed"],
            columns="method",
            values="return_auc",
        )
        final = records.pivot(
            index=["coupling", "training_seed"],
            columns="method",
            values="final_return",
        )
        independent = relative_gain(
            auc.xs("independent")[controller_method],
            auc.xs("independent")["fixed_q8"],
        ).sort_index()
        shared = relative_gain(
            auc.xs("shared")[controller_method],
            auc.xs("shared")["fixed_q2"],
        ).sort_index()
        controller_mixture = (
            auc.xs("independent")[controller_method]
            + auc.xs("shared")[controller_method]
        ) / 2.0
        q8_mixture = (
            auc.xs("independent")["fixed_q8"]
            + auc.xs("shared")["fixed_q8"]
        ) / 2.0
        mixture = relative_gain(controller_mixture, q8_mixture).sort_index()
        controller_final_mixture = (
            final.xs("independent")[controller_method]
            + final.xs("shared")[controller_method]
        ) / 2.0
        q8_final_mixture = (
            final.xs("independent")["fixed_q8"]
            + final.xs("shared")["fixed_q8"]
        ) / 2.0
        mixture_final = relative_gain(
            controller_final_mixture, q8_final_mixture
        ).sort_index()
        critical = float(config["inference"]["student_t_one_sided_critical"])
        for name, values in (
            ("independent_vs_q8", independent),
            ("shared_vs_q2", shared),
            ("mixture_vs_q8", mixture),
        ):
            metrics[f"{name}_mean"] = float(values.mean())
            metrics[f"{name}_lower"] = one_sided_lower_t(values, critical)
        positive_pairs = int((mixture > 0).sum())
        metrics["mixture_vs_q8_positive_pairs"] = positive_pairs
        metrics["mixture_vs_q8_sign_p"] = exact_one_sided_sign_p(
            positive_pairs, len(mixture)
        )
        metrics["mixture_final_return_vs_q8_mean"] = float(mixture_final.mean())
        metrics["maximum_probe_message_fraction"] = float(
            controller.probe_message_fraction.max()
        )
        metrics["maximum_selection_overhead_fraction"] = float(
            controller.selection_overhead_fraction.max()
        )
        for seed in seeds:
            paired_rows.append(
                {
                    "training_seed": int(seed),
                    "independent_relative_auc_vs_q8": float(independent.loc[seed]),
                    "shared_relative_auc_vs_q2": float(shared.loc[seed]),
                    "mixture_relative_auc_vs_q8": float(mixture.loc[seed]),
                    "mixture_relative_final_return_vs_q8": float(
                        mixture_final.loc[seed]
                    ),
                }
            )

    threshold = config["mandatory_confirmation_gates"]

    def at_least(metric: str, gate: str) -> bool:
        value = metrics[metric]
        return value is not None and value >= float(threshold[gate])

    gates = {
        "finite_complete_exact_accounting_and_clean_upstream": valid,
        "independent_selected_q8_fraction_min": at_least(
            "independent_selected_q8_fraction", "independent_selected_q8_fraction_min"
        ),
        "shared_selected_q2_fraction_min": at_least(
            "shared_selected_q2_fraction", "shared_selected_q2_fraction_min"
        ),
        "independent_vs_q8_lower_min": at_least(
            "independent_vs_q8_lower", "independent_vs_q8_lower_min"
        ),
        "shared_vs_q2_lower_min": at_least(
            "shared_vs_q2_lower", "shared_vs_q2_lower_min"
        ),
        "mixture_vs_q8_lower_min": at_least(
            "mixture_vs_q8_lower", "mixture_vs_q8_lower_min"
        ),
        "mixture_vs_q8_mean_min": at_least(
            "mixture_vs_q8_mean", "mixture_vs_q8_mean_min"
        ),
        "mixture_vs_q8_positive_pairs_min": at_least(
            "mixture_vs_q8_positive_pairs", "mixture_vs_q8_positive_pairs_min"
        ),
        "mixture_vs_q8_sign_p_max": metrics["mixture_vs_q8_sign_p"] is not None
        and metrics["mixture_vs_q8_sign_p"]
        <= float(threshold["mixture_vs_q8_sign_p_max"]),
        "mixture_final_return_vs_q8_mean_min": at_least(
            "mixture_final_return_vs_q8_mean",
            "mixture_final_return_vs_q8_mean_min",
        ),
        "probe_message_fraction_max": metrics["maximum_probe_message_fraction"]
        is not None
        and metrics["maximum_probe_message_fraction"]
        <= float(threshold["probe_message_fraction_max"]),
        "selection_overhead_fraction_max": metrics[
            "maximum_selection_overhead_fraction"
        ]
        is not None
        and metrics["maximum_selection_overhead_fraction"]
        <= float(threshold["selection_overhead_fraction_max"]),
        "analysis_replay_byte_identical": bool(replay_identical),
    }
    passed = all(gates.values())
    return {
        "experiment_id": config["experiment_id"],
        "run_count": len(records),
        "expected_run_count": len(expected),
        **metrics,
        "paired_estimands": paired_rows,
        "gates": gates,
        "all_mandatory_gates_pass": passed,
        "decision": "admit-confirmed-return-evidence" if passed else "stop",
    }


def evaluate(
    root: Path, config_path: Path, *, replay_identical: bool = False
) -> tuple[dict, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records, curves = common.collect_records(
        root, config, checkpoints=int(config["evaluation"]["auc_grid_points"])
    )
    records = attach_final_returns(records, curves)
    gate = evaluate_records(records, config, replay_identical=replay_identical)
    summary = common.aggregate_curves(
        curves, checkpoints=int(config["evaluation"]["auc_grid_points"])
    )
    return gate, summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "experiments" / "marl_mamujoco_calibrated_confirmation.json",
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--replay-identical", action="store_true")
    args = parser.parse_args()
    gate, summary = evaluate(
        args.root, args.config, replay_identical=args.replay_identical
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    common.plot(summary, args.figure)
    args.gate_json.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": gate["decision"], "runs": gate["run_count"]}))


if __name__ == "__main__":
    main()
