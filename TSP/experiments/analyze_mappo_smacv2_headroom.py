"""Analyze TSP-MARL-SMACV2-HEADROOM-001 against its frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_mappo_smacv2_development import auc_on_budget_fraction, _fixed_curve


def collect(root: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, curves = [], []
    for metadata_path in root.rglob("tsp_bridge_metadata.json"):
        curve, record = _fixed_curve(metadata_path)
        for metric in ("team_return", "win_rate"):
            record[f"{metric}_auc"] = auc_on_budget_fraction(
                curve, metric, int(config["evaluation"]["auc_grid_points"])
            )
            ordered = curve.sort_values("budget_fraction")
            record[f"initial_{metric}"] = float(ordered[metric].iloc[0])
            record[f"terminal_{metric}"] = float(ordered[metric].iloc[-1])
        records.append(record)
        tagged = curve.copy()
        tagged["method"] = record["method"]
        tagged["coupling"] = record["coupling"]
        tagged["training_seed"] = record["training_seed"]
        curves.append(tagged)
    if not records:
        raise FileNotFoundError(f"no fixed-q metadata below {root}")
    return pd.DataFrame(records), pd.concat(curves, ignore_index=True)


def relative_gain(preferred: float, comparator: float) -> float:
    return float((preferred - comparator) / max(abs(comparator), 1e-12))


def evaluate(root: Path, config_path: Path, replay_identical: bool) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records, _ = collect(root, config)
    expected = {
        (regime, method, seed)
        for regime in config["dependence_regimes"]
        for method in config["fixed_methods"]
        for seed in config["development_seeds"]
    }
    observed = list(
        records[["coupling", "method", "training_seed"]].itertuples(
            index=False, name=None
        )
    )
    complete = len(observed) == len(set(observed)) and set(observed) == expected
    finite = bool(
        np.isfinite(
            records[
                ["team_return_auc", "win_rate_auc", "terminal_team_return"]
            ].to_numpy(float)
        ).all()
    )
    valid = bool(
        complete
        and finite
        and records["accounting_ok"].all()
        and (~records["upstream_modified"]).all()
        and (
            records["harl_commit"] == config["upstream"]["harl_commit"]
        ).all()
    )
    means = (
        records.groupby(["coupling", "method"])[
            ["team_return_auc", "win_rate_auc", "terminal_team_return"]
        ]
        .mean()
        .sort_index()
    )
    metrics = {}
    if complete:
        independent_q1 = means.loc[("independent", "fixed_q1")]
        independent_q8 = means.loc[("independent", "fixed_q8")]
        shared_q1 = means.loc[("shared", "fixed_q1")]
        shared_q8 = means.loc[("shared", "fixed_q8")]
        independent_gain = relative_gain(
            independent_q8.team_return_auc, independent_q1.team_return_auc
        )
        shared_gain = relative_gain(shared_q1.team_return_auc, shared_q8.team_return_auc)
        oracle = 0.5 * (
            max(independent_q1.team_return_auc, independent_q8.team_return_auc)
            + max(shared_q1.team_return_auc, shared_q8.team_return_auc)
        )
        static = {
            method: float(
                records[records["method"] == method]["team_return_auc"].mean()
            )
            for method in config["fixed_methods"]
        }
        strong_method = sorted(static, key=lambda method: (-static[method], method))[0]
        strong = static[strong_method]
        metrics = {
            "independent_q8_return_auc_relative_gain": independent_gain,
            "shared_q1_return_auc_relative_gain": shared_gain,
            "regime_oracle_return_auc": float(oracle),
            "strong_static_method": strong_method,
            "strong_static_return_auc": strong,
            "regime_oracle_return_auc_relative_gain_over_strong_static": relative_gain(
                oracle, strong
            ),
            "independent_terminal_direction_agrees": bool(
                independent_q8.terminal_team_return
                >= independent_q1.terminal_team_return
            ),
            "shared_terminal_direction_agrees": bool(
                shared_q1.terminal_team_return >= shared_q8.terminal_team_return
            ),
            "cell_means": {
                f"{regime}/{method}": {
                    key: float(value) for key, value in row.items()
                }
                for (regime, method), row in means.iterrows()
            },
        }
    thresholds = config["mandatory_gates"]
    gates = {
        "finite_complete_exact_accounting_and_clean_upstream": valid,
        "independent_q8_return_auc_relative_gain_min": bool(
            metrics
            and metrics["independent_q8_return_auc_relative_gain"]
            >= thresholds["independent_q8_return_auc_relative_gain_min"]
        ),
        "shared_q1_return_auc_relative_gain_min": bool(
            metrics
            and metrics["shared_q1_return_auc_relative_gain"]
            >= thresholds["shared_q1_return_auc_relative_gain_min"]
        ),
        "regime_oracle_return_auc_relative_gain_over_strong_static_min": bool(
            metrics
            and metrics[
                "regime_oracle_return_auc_relative_gain_over_strong_static"
            ]
            >= thresholds[
                "regime_oracle_return_auc_relative_gain_over_strong_static_min"
            ]
        ),
        "terminal_return_directions_agree_with_auc": bool(
            metrics
            and metrics["independent_terminal_direction_agrees"]
            and metrics["shared_terminal_direction_agrees"]
        ),
        "analysis_replay_byte_identical": bool(replay_identical),
    }
    return {
        "experiment_id": config["experiment_id"],
        "run_count": len(records),
        "expected_run_count": len(expected),
        **metrics,
        "gates": gates,
        "pass": all(gates.values()),
        "decision": "authorize-controller-development-preregistration"
        if all(gates.values())
        else "stop",
        "records": records.to_dict(orient="records"),
    }


def aggregate_curves(curves: pd.DataFrame, points: int) -> pd.DataFrame:
    grid = np.linspace(0, 1, points)
    rows = []
    for (regime, method, seed), frame in curves.groupby(
        ["coupling", "method", "training_seed"]
    ):
        frame = frame.sort_values("budget_fraction")
        for fraction in grid:
            rows.append(
                {
                    "coupling": regime,
                    "method": method,
                    "training_seed": int(seed),
                    "budget_fraction": float(fraction),
                    "team_return": float(
                        np.interp(
                            fraction,
                            frame.budget_fraction,
                            frame.team_return,
                            left=frame.team_return.iloc[0],
                            right=frame.team_return.iloc[-1],
                        )
                    ),
                    "win_rate": float(
                        np.interp(
                            fraction,
                            frame.budget_fraction,
                            frame.win_rate,
                            left=frame.win_rate.iloc[0],
                            right=frame.win_rate.iloc[-1],
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot(samples: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.25), sharey=True)
    styles = {"fixed_q1": ("Fixed q=1", "#2474B5"), "fixed_q8": ("Fixed q=8", "#D95F02")}
    for ax, regime in zip(axes, ("independent", "shared")):
        for method, frame in samples[samples.coupling == regime].groupby("method"):
            stats = frame.groupby("budget_fraction").team_return.agg(["mean", "std", "count"])
            x = stats.index.to_numpy(float)
            mean = stats["mean"].to_numpy(float)
            ci = 1.96 * stats["std"].fillna(0).to_numpy(float) / np.sqrt(stats["count"])
            label, color = styles[method]
            ax.plot(x, mean, color=color, label=label)
            ax.fill_between(x, mean-ci, mean+ci, color=color, alpha=0.14)
        ax.set_title(regime.capitalize())
        ax.set_xlabel("Charged budget fraction")
        ax.grid(color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Evaluation team return")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    tsp = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=tsp / "experiments" / "marl_smacv2_headroom.json"
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--replay-identical", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, curves = collect(args.root, config)
    samples = aggregate_curves(curves, int(config["evaluation"]["auc_grid_points"]))
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.summary_csv, index=False)
    plot(samples, args.figure)
    result = evaluate(args.root, args.config, args.replay_identical)
    args.gate_json.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "runs": result["run_count"]}))


if __name__ == "__main__":
    main()
