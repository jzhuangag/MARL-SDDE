"""Analyze the full fixed-q landscape for the SMACv2 development task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_mappo_smacv2_headroom import aggregate_curves, collect, relative_gain
from lyapunov_probe_commit import message_limited_lyapunov_score


def collect_two(existing_root: Path, extension_root: Path, config: dict):
    records, curves = [], []
    for root in (existing_root, extension_root):
        frame, curve = collect(root, config)
        records.append(frame)
        curves.append(curve)
    records = pd.concat(records, ignore_index=True)
    curves = pd.concat(curves, ignore_index=True)
    methods = {f"fixed_q{q}" for q in config["fixed_q_grid"]}
    mask = (
        records.method.isin(methods)
        & records.coupling.isin(config["dependence_regimes"])
        & records.training_seed.isin(config["development_seeds"])
    )
    curve_mask = (
        curves.method.isin(methods)
        & curves.coupling.isin(config["dependence_regimes"])
        & curves.training_seed.isin(config["development_seeds"])
    )
    return records[mask].copy(), curves[curve_mask].copy()


def predicted_q(config: dict, regime: str) -> int:
    rho = float(np.clip(config["calibrated_rho"][regime], 0.0, 1.0))
    h = int(config["budgets"]["server_overhead"])
    length = int(config["budgets"]["rollout_length"])
    return min(
        config["fixed_q_grid"],
        key=lambda q: (message_limited_lyapunov_score(q, rho, h, length), q),
    )


def evaluate(existing_root: Path, extension_root: Path, config_path: Path, replay_identical: bool):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records, _ = collect_two(existing_root, extension_root, config)
    methods = [f"fixed_q{q}" for q in config["fixed_q_grid"]]
    expected = {
        (regime, method, seed)
        for regime in config["dependence_regimes"]
        for method in methods
        for seed in config["development_seeds"]
    }
    observed = list(records[["coupling", "method", "training_seed"]].itertuples(index=False, name=None))
    complete = len(observed) == len(set(observed)) and set(observed) == expected
    finite = bool(
        np.isfinite(records[["team_return_auc", "win_rate_auc", "terminal_team_return"]].to_numpy(float)).all()
    )
    valid = bool(
        complete
        and finite
        and records.accounting_ok.all()
        and (~records.upstream_modified).all()
        and (records.harl_commit == config["upstream"]["harl_commit"]).all()
    )
    metrics = {}
    if complete:
        means = records.groupby(["coupling", "method"])[
            ["team_return_auc", "win_rate_auc", "terminal_team_return"]
        ].mean()
        oracle_actions, oracle_values, predicted_actions, predicted_gaps = {}, {}, {}, {}
        for regime in config["dependence_regimes"]:
            values = {
                method: float(means.loc[(regime, method), "team_return_auc"])
                for method in methods
            }
            oracle_method = sorted(values, key=lambda m: (-values[m], m))[0]
            q_hat = predicted_q(config, regime)
            predicted_method = f"fixed_q{q_hat}"
            oracle_actions[regime] = int(oracle_method.replace("fixed_q", ""))
            oracle_values[regime] = values[oracle_method]
            predicted_actions[regime] = q_hat
            predicted_gaps[regime] = relative_gain(values[predicted_method], values[oracle_method])
        static = {
            method: float(records[records.method == method].team_return_auc.mean())
            for method in methods
        }
        strong_method = sorted(static, key=lambda m: (-static[m], m))[0]
        oracle_mixture = float(np.mean(list(oracle_values.values())))
        predicted_mixture = float(
            np.mean(
                [
                    means.loc[(regime, f"fixed_q{predicted_actions[regime]}"), "team_return_auc"]
                    for regime in config["dependence_regimes"]
                ]
            )
        )
        metrics = {
            "cell_means": {
                f"{regime}/{method}": {key: float(value) for key, value in row.items()}
                for (regime, method), row in means.iterrows()
            },
            "oracle_action": oracle_actions,
            "oracle_return_auc": oracle_values,
            "lyapunov_predicted_action": predicted_actions,
            "lyapunov_predicted_relative_gap_to_oracle": predicted_gaps,
            "strong_static_method": strong_method,
            "strong_static_return_auc": static[strong_method],
            "regime_oracle_mixture_return_auc": oracle_mixture,
            "regime_oracle_relative_gain_over_strong_static": relative_gain(
                oracle_mixture, static[strong_method]
            ),
            "lyapunov_predicted_mixture_return_auc": predicted_mixture,
            "lyapunov_predicted_relative_gap_to_oracle_mixture": relative_gain(
                predicted_mixture, oracle_mixture
            ),
        }
    thresholds = config["admission_gates_for_a_future_controller_study"]
    gates = {
        "finite_complete_exact_accounting_and_clean_upstream": valid,
        "regime_oracle_relative_gain_over_strong_static_min": bool(
            metrics
            and metrics["regime_oracle_relative_gain_over_strong_static"]
            >= float(thresholds["regime_oracle_relative_gain_over_strong_static_min"])
        ),
        "lyapunov_predicted_action_relative_gap_to_regime_oracle_min_each_regime": bool(
            metrics
            and all(
                gap >= float(
                    thresholds[
                        "lyapunov_predicted_action_relative_gap_to_regime_oracle_min_each_regime"
                    ]
                )
                for gap in metrics["lyapunov_predicted_relative_gap_to_oracle"].values()
            )
        ),
        "all_registered_fixed_q_present": bool(
            complete and set(records.selected_q) == set(config["fixed_q_grid"])
        ),
        "analysis_replay_byte_identical": bool(replay_identical),
    }
    passed = all(gates.values())
    return {
        "experiment_id": config["experiment_id"],
        "run_count": len(records),
        "expected_run_count": len(expected),
        **metrics,
        "gates": gates,
        "pass": passed,
        "decision": "authorize-fresh-controller-preregistration" if passed else "stop",
        "records": records.to_dict(orient="records"),
    }


def plot(summary: pd.DataFrame, output: Path) -> None:
    colors = {1: "#7f7f7f", 2: "#56B4E9", 4: "#009E73", 8: "#E69F00"}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35), sharey=True)
    for ax, regime in zip(axes, ("independent", "shared")):
        for q, color in colors.items():
            method = f"fixed_q{q}"
            frame = summary[(summary.coupling == regime) & (summary.method == method)]
            stats = frame.groupby("budget_fraction").team_return.agg(["mean", "std", "count"])
            x = stats.index.to_numpy(float)
            mean = stats["mean"].to_numpy(float)
            ci = 1.96 * stats["std"].fillna(0).to_numpy(float) / np.sqrt(stats["count"])
            ax.plot(x, mean, label=f"q={q}", color=color)
            ax.fill_between(x, mean-ci, mean+ci, color=color, alpha=0.10)
        ax.set_title(regime.capitalize())
        ax.set_xlabel("Charged budget fraction")
        ax.grid(color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Evaluation team return")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    tsp = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=tsp / "experiments" / "marl_smacv2_qgrid_extension.json"
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--replay-identical", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, curves = collect_two(args.existing_root, args.extension_root, config)
    summary = aggregate_curves(curves, int(config["evaluation"]["auc_grid_points"]))
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    plot(summary, args.figure)
    result = evaluate(args.existing_root, args.extension_root, args.config, args.replay_identical)
    args.gate_json.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "runs": result["run_count"]}))


if __name__ == "__main__":
    main()
