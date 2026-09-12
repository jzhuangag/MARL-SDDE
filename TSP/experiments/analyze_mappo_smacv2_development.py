"""Analyze the frozen TSP-MARL-SMACV2-DEV-001 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def auc_on_budget_fraction(
    frame: pd.DataFrame, metric: str, checkpoints: int = 21
) -> float:
    frame = frame.sort_values("budget_fraction")
    x = frame["budget_fraction"].to_numpy(float)
    y = frame[metric].to_numpy(float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{metric} curve must be nonempty and finite")
    if np.any(np.diff(x) < 0) or x[0] < 0 or x[-1] > 1.0 + 1e-12:
        raise ValueError("invalid budget-fraction axis")
    if metric == "win_rate" and (np.any(y < 0) or np.any(y > 1)):
        raise ValueError("win rate must lie in [0,1]")
    grid = np.linspace(0.0, 1.0, checkpoints)
    values = np.interp(grid, x, y, left=y[0], right=y[-1])
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(values, grid))


def _fixed_curve(metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    q = int(metadata["q_rollout_workers"])
    rollout_length = int(metadata["rollout_length"])
    rows = []
    for line in metadata_path.with_name("progress.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 3:
            raise ValueError("SMACv2 progress must contain steps, return, win rate")
        actor_transitions = int(fields[0])
        denominator = q * rollout_length
        if actor_transitions % denominator:
            raise ValueError("fixed progress has fractional learner update")
        updates = actor_transitions // denominator
        messages = updates * int(metadata["message_cost_per_update"])
        environment = updates * int(metadata["environment_cost_per_update"])
        rows.append(
            {
                "budget_fraction": max(
                    messages / int(metadata["message_budget"]),
                    environment / int(metadata["environment_budget"]),
                ),
                "team_return": float(fields[1]),
                "win_rate": float(fields[2]),
            }
        )
    accounting_ok = all(
        (
            int(metadata["charged_training_messages"])
            <= int(metadata["message_budget"]),
            int(metadata["charged_training_environment_ticks"])
            <= int(metadata["environment_budget"]),
            int(metadata["charged_training_messages"])
            == int(metadata["usable_updates"])
            * int(metadata["message_cost_per_update"]),
            int(metadata["charged_training_environment_ticks"])
            == int(metadata["usable_updates"])
            * int(metadata["environment_cost_per_update"]),
        )
    )
    record = {
        "method": f"fixed_q{q}",
        "coupling": metadata["coupling"],
        "training_seed": int(metadata["seed"]),
        "selected_q": q,
        "accounting_ok": accounting_ok,
        "upstream_modified": bool(metadata["upstream_modified"]),
        "harl_commit": metadata["harl_commit"],
        "selection_overhead_fraction": 0.0,
        "probe_message_fraction": 0.0,
    }
    return pd.DataFrame(rows), record


def _controller_curve(metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    curve = pd.read_csv(metadata_path.with_name("charged_progress.csv"))[
        ["budget_fraction", "team_return", "win_rate"]
    ]
    accounting = metadata["accounting"]
    accounting_ok = all(
        (
            int(accounting["total_messages"]) <= int(metadata["message_budget"]),
            int(accounting["total_environment_ticks"])
            <= int(metadata["environment_budget"]),
            int(accounting["total_messages"])
            == int(accounting["probe_messages"])
            + int(accounting["training_messages"]),
            int(accounting["total_environment_ticks"])
            == int(accounting["probe_environment_ticks"])
            + int(accounting["training_environment_ticks"]),
        )
    )
    record = {
        "method": "lyapunov_probe_commit",
        "coupling": metadata["coupling"],
        "training_seed": int(metadata["training_seed"]),
        "selected_q": int(metadata["decision"]["selected_q"]),
        "rho_estimate": float(metadata["decision"]["certificate"]["estimate"]),
        "rho_upper": float(metadata["decision"]["certificate"]["upper"]),
        "accounting_ok": accounting_ok,
        "upstream_modified": bool(metadata["upstream_modified"]),
        "harl_commit": metadata["harl_commit"],
        "selection_overhead_fraction": float(
            metadata["selection_overhead_fraction"]
        ),
        "probe_message_fraction": int(accounting["probe_messages"])
        / int(metadata["message_budget"]),
    }
    return curve, record


def collect_records(
    root: Path, checkpoints: int = 21
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    curves = []
    for metadata_path in root.rglob("tsp_bridge_metadata.json"):
        if any(parent.name == "training" for parent in metadata_path.parents):
            continue
        curve, record = _fixed_curve(metadata_path)
        for metric in ("win_rate", "team_return"):
            record[f"{metric}_auc"] = auc_on_budget_fraction(
                curve, metric, checkpoints
            )
            record[f"terminal_{metric}"] = float(
                curve.sort_values("budget_fraction")[metric].iloc[-1]
            )
        records.append(record)
        cell = curve.copy()
        for key in ("method", "coupling", "training_seed"):
            cell[key] = record[key]
        curves.append(cell)
    for metadata_path in root.rglob("tsp_probe_commit_metadata.json"):
        curve, record = _controller_curve(metadata_path)
        for metric in ("win_rate", "team_return"):
            record[f"{metric}_auc"] = auc_on_budget_fraction(
                curve, metric, checkpoints
            )
            record[f"terminal_{metric}"] = float(
                curve.sort_values("budget_fraction")[metric].iloc[-1]
            )
        records.append(record)
        cell = curve.copy()
        for key in ("method", "coupling", "training_seed"):
            cell[key] = record[key]
        curves.append(cell)
    if not records:
        raise FileNotFoundError(f"no registered result metadata under {root}")
    return pd.DataFrame(records), pd.concat(curves, ignore_index=True)


def _fixed_q(method: str) -> int:
    if not method.startswith("fixed_q"):
        raise ValueError(f"not a fixed method: {method}")
    return int(method.removeprefix("fixed_q"))


def evaluate_gates(root: Path, config_path: Path, audit_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records, _ = collect_records(
        root, checkpoints=int(config["evaluation"]["auc_grid_points"])
    )
    seeds = config["seed_registry"]["development_training"]
    expected = {
        (coupling, method, seed)
        for coupling in config["controlled_dependence"]["regimes"]
        for method in config["methods"]
        for seed in seeds
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
                ["win_rate_auc", "team_return_auc", "terminal_win_rate"]
            ].to_numpy(float)
        ).all()
    )
    exact = bool(records["accounting_ok"].all())
    clean = bool((~records["upstream_modified"]).all())
    pinned = bool(
        (records["harl_commit"] == config["upstream"]["harl_commit"]).all()
    )

    metrics = {
        "independent_selected_q8_fraction": None,
        "shared_selected_q1_fraction": None,
        "strong_static_method": None,
        "strong_static_win_rate_auc": None,
        "regime_oracle_win_rate_auc": None,
        "static_oracle_win_rate_auc_headroom_absolute": None,
        "controller_equal_mixture_win_rate_auc_gain_over_strong_static_absolute": None,
        "controller_min_regime_win_rate_auc_gap_from_matching_fixed": None,
        "shared_win_rate_auc_gain_over_strong_static": None,
        "shared_team_return_auc_gain_over_strong_static": None,
        "maximum_probe_message_fraction": None,
        "maximum_selection_overhead_fraction": None,
    }
    if complete:
        controller = records[records["method"] == "lyapunov_probe_commit"]
        metrics["independent_selected_q8_fraction"] = float(
            (
                controller[controller["coupling"] == "independent"]["selected_q"]
                == 8
            ).mean()
        )
        metrics["shared_selected_q1_fraction"] = float(
            (
                controller[controller["coupling"] == "shared"]["selected_q"]
                == 1
            ).mean()
        )
        fixed = records[records["method"].str.startswith("fixed_q")]
        static_means = fixed.groupby("method")["win_rate_auc"].mean()
        strong_method = sorted(
            static_means.index,
            key=lambda method: (-float(static_means[method]), _fixed_q(method)),
        )[0]
        strong_auc = float(static_means[strong_method])
        oracle_rows = []
        matching_gaps = []
        for coupling in config["controlled_dependence"]["regimes"]:
            cell_means = (
                fixed[fixed["coupling"] == coupling]
                .groupby("method")[["win_rate_auc", "team_return_auc"]]
                .mean()
            )
            oracle_method = sorted(
                cell_means.index,
                key=lambda method: (
                    -float(cell_means.loc[method, "win_rate_auc"]),
                    _fixed_q(method),
                ),
            )[0]
            controller_cell = controller[controller["coupling"] == coupling]
            matching_gaps.append(
                float(controller_cell["win_rate_auc"].mean())
                - float(cell_means.loc[oracle_method, "win_rate_auc"])
            )
            oracle_rows.append(
                {
                    "coupling": coupling,
                    "method": oracle_method,
                    "win_rate_auc": float(
                        cell_means.loc[oracle_method, "win_rate_auc"]
                    ),
                }
            )
        oracle_auc = float(np.mean([row["win_rate_auc"] for row in oracle_rows]))
        controller_auc = float(controller["win_rate_auc"].mean())
        shared_controller = controller[controller["coupling"] == "shared"]
        shared_strong = fixed[
            (fixed["coupling"] == "shared") & (fixed["method"] == strong_method)
        ]
        metrics.update(
            {
                "strong_static_method": strong_method,
                "strong_static_win_rate_auc": strong_auc,
                "regime_oracle_win_rate_auc": oracle_rows,
                "static_oracle_win_rate_auc_headroom_absolute": oracle_auc
                - strong_auc,
                "controller_equal_mixture_win_rate_auc_gain_over_strong_static_absolute": controller_auc
                - strong_auc,
                "controller_min_regime_win_rate_auc_gap_from_matching_fixed": min(
                    matching_gaps
                ),
                "shared_win_rate_auc_gain_over_strong_static": float(
                    shared_controller["win_rate_auc"].mean()
                    - shared_strong["win_rate_auc"].mean()
                ),
                "shared_team_return_auc_gain_over_strong_static": float(
                    shared_controller["team_return_auc"].mean()
                    - shared_strong["team_return_auc"].mean()
                ),
                "maximum_probe_message_fraction": float(
                    controller["probe_message_fraction"].max()
                ),
                "maximum_selection_overhead_fraction": float(
                    controller["selection_overhead_fraction"].max()
                ),
            }
        )

    thresholds = config["mandatory_development_gates"]

    def at_least(metric: str, threshold: str) -> bool:
        return metrics[metric] is not None and float(metrics[metric]) >= float(
            thresholds[threshold]
        )

    direction_agrees = (
        metrics["shared_win_rate_auc_gain_over_strong_static"] is not None
        and metrics["shared_win_rate_auc_gain_over_strong_static"] >= 0
        and metrics["shared_team_return_auc_gain_over_strong_static"] >= 0
    )
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
        "static_oracle_win_rate_auc_headroom_absolute_min": at_least(
            "static_oracle_win_rate_auc_headroom_absolute",
            "static_oracle_win_rate_auc_headroom_absolute_min",
        ),
        "controller_equal_mixture_win_rate_auc_gain_over_strong_static_absolute_min": at_least(
            "controller_equal_mixture_win_rate_auc_gain_over_strong_static_absolute",
            "controller_equal_mixture_win_rate_auc_gain_over_strong_static_absolute_min",
        ),
        "controller_per_regime_win_rate_auc_gap_from_matching_fixed_min": at_least(
            "controller_min_regime_win_rate_auc_gap_from_matching_fixed",
            "controller_per_regime_win_rate_auc_gap_from_matching_fixed_min",
        ),
        "team_return_auc_direction_agrees_with_win_rate_in_shared_regime": bool(
            direction_agrees
        ),
        "probe_message_fraction_max": metrics["maximum_probe_message_fraction"]
        is not None
        and metrics["maximum_probe_message_fraction"]
        <= float(thresholds["probe_message_fraction_max"]),
        "controller_wall_clock_overhead_fraction_max": metrics[
            "maximum_selection_overhead_fraction"
        ]
        is not None
        and metrics["maximum_selection_overhead_fraction"]
        <= float(thresholds["controller_wall_clock_overhead_fraction_max"]),
        "marginal_preserving_coupling_audit": bool(audit.get("pass")),
        "analysis_replay_byte_identical": bool(
            audit.get("analysis_replay_byte_identical")
        ),
    }
    passed = all(gates.values())
    return {
        "experiment_id": config["experiment_id"],
        "run_count": len(records),
        "expected_run_count": len(expected),
        **metrics,
        "gates": gates,
        "all_mandatory_gates_pass": passed,
        "decision": "authorize-confirmation-preregistration" if passed else "stop",
        "records": records.to_dict(orient="records"),
    }


def aggregate_curves(curves: pd.DataFrame, checkpoints: int = 21) -> pd.DataFrame:
    grid = np.linspace(0.0, 1.0, checkpoints)
    rows = []
    for (coupling, method, seed), curve in curves.groupby(
        ["coupling", "method", "training_seed"]
    ):
        curve = curve.sort_values("budget_fraction")
        for fraction in grid:
            row = {
                "coupling": coupling,
                "method": method,
                "training_seed": int(seed),
                "budget_fraction": float(fraction),
            }
            for metric in ("team_return", "win_rate"):
                row[metric] = float(
                    np.interp(
                        fraction,
                        curve["budget_fraction"].to_numpy(float),
                        curve[metric].to_numpy(float),
                        left=float(curve[metric].iloc[0]),
                        right=float(curve[metric].iloc[-1]),
                    )
                )
            rows.append(row)
    samples = pd.DataFrame(rows)
    return (
        samples.groupby(["coupling", "method", "budget_fraction"], as_index=False)
        .agg(
            team_return_mean=("team_return", "mean"),
            team_return_std=("team_return", "std"),
            win_rate_mean=("win_rate", "mean"),
            win_rate_std=("win_rate", "std"),
            seeds=("training_seed", "nunique"),
        )
        .sort_values(["coupling", "method", "budget_fraction"])
    )


def plot(summary: pd.DataFrame, output: Path) -> None:
    methods = {
        "lyapunov_probe_commit": ("Lyapunov controller", "#111111", "-"),
        "fixed_q1": ("Fixed q=1", "#2474B5", "--"),
        "fixed_q8": ("Fixed q=8", "#D95F02", ":"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.5), sharex=True)
    for column, coupling in enumerate(("independent", "shared")):
        for row, metric in enumerate(("win_rate", "team_return")):
            ax = axes[row, column]
            for method, curve in summary[summary["coupling"] == coupling].groupby(
                "method"
            ):
                label, color, style = methods[method]
                x = curve["budget_fraction"].to_numpy(float)
                mean = curve[f"{metric}_mean"].to_numpy(float)
                std = curve[f"{metric}_std"].fillna(0.0).to_numpy(float)
                n = curve["seeds"].to_numpy(float)
                ci = 1.96 * std / np.sqrt(n)
                ax.plot(x, mean, color=color, linestyle=style, label=label)
                ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.12)
            ax.grid(color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(f"{coupling.capitalize()} rollout streams")
            if column == 0:
                ax.set_ylabel("Evaluation win rate" if row == 0 else "Team return")
            if row == 1:
                ax.set_xlabel("Charged budget fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    tsp_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=tsp_root / "experiments" / "marl_smacv2_development.json",
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, curves = collect_records(
        args.root, checkpoints=int(config["evaluation"]["auc_grid_points"])
    )
    summary = aggregate_curves(
        curves, checkpoints=int(config["evaluation"]["auc_grid_points"])
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    plot(summary, args.figure)
    gate = evaluate_gates(args.root, args.config, args.audit)
    args.gate_json.parent.mkdir(parents=True, exist_ok=True)
    args.gate_json.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": gate["decision"], "runs": gate["run_count"]}))


if __name__ == "__main__":
    main()
