"""Analyze the frozen TSP-MARL-DEV-002 development experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def auc_on_budget_fraction(frame: pd.DataFrame, checkpoints: int = 21) -> float:
    frame = frame.sort_values("budget_fraction")
    x = frame["budget_fraction"].to_numpy(float)
    y = frame["team_return"].to_numpy(float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("curve must be nonempty and finite")
    if np.any(np.diff(x) < 0) or x[0] < 0 or x[-1] > 1.0 + 1e-12:
        raise ValueError("invalid budget-fraction axis")
    grid = np.linspace(0.0, 1.0, checkpoints)
    values = np.interp(grid, x, y, left=y[0], right=y[-1])
    # NumPy 1.23 in the pinned HPC4 environment exposes the identical
    # trapezoidal rule under the historical name ``trapz``.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(values, grid))


def fixed_curve(metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    q = int(metadata["q_rollout_workers"])
    rollout_length = int(metadata["rollout_length"])
    rows = []
    for line in metadata_path.with_name("progress.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        actor_text, return_text = line.split(",")
        actor_transitions = int(actor_text)
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
                "team_return": float(return_text),
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


def controller_curve(metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    curve = pd.read_csv(metadata_path.with_name("charged_progress.csv"))[
        ["budget_fraction", "team_return"]
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


def collect_records(root: Path, config: dict, checkpoints: int = 21) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    curves = []
    for metadata_path in root.rglob("tsp_bridge_metadata.json"):
        # Controller training directories also contain fixed-runner metadata;
        # their parent tree is owned by the controller record and must not be
        # double counted as a standalone baseline.
        if any(parent.name == "training" for parent in metadata_path.parents):
            continue
        curve, record = fixed_curve(metadata_path)
        record["return_auc"] = auc_on_budget_fraction(curve, checkpoints)
        records.append(record)
        curve = curve.copy()
        for key in ("method", "coupling", "training_seed"):
            curve[key] = record[key]
        curves.append(curve)
    for metadata_path in root.rglob("tsp_probe_commit_metadata.json"):
        curve, record = controller_curve(metadata_path)
        record["return_auc"] = auc_on_budget_fraction(curve, checkpoints)
        records.append(record)
        curve = curve.copy()
        for key in ("method", "coupling", "training_seed"):
            curve[key] = record[key]
        curves.append(curve)
    if not records:
        raise FileNotFoundError(f"no registered result metadata under {root}")
    return pd.DataFrame(records), pd.concat(curves, ignore_index=True)


def evaluate_gates(root: Path, config_path: Path, audit_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records, _ = collect_records(root, config)
    training_seeds = config["seed_registry"]["development_training"]
    expected = {
        (coupling, method, seed)
        for coupling in config["coupling_regimes"]
        for method in config["methods"]
        for seed in training_seeds
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
    if complete:
        independent_selection = float(
            (
                controller[controller["coupling"] == "independent"]["selected_q"]
                == 8
            ).mean()
        )
        shared_selection = float(
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
        independent_gain = float(
            paired.xs("independent")["relative_vs_q8"].mean()
        )
        shared_gain = float(paired.xs("shared")["relative_vs_q8"].mean())
        mixture_gain = float(paired["relative_vs_q8"].mean())
        probe_fraction = float(controller["probe_message_fraction"].max())
        overhead = float(controller["selection_overhead_fraction"].max())
    else:
        # An invalid result lattice is itself a stop condition.  Do not hide a
        # duplicate or missing cell through aggregation merely to compute the
        # downstream scientific metrics.
        independent_selection = None
        shared_selection = None
        independent_gain = None
        shared_gain = None
        mixture_gain = None
        probe_fraction = None
        overhead = None

    thresholds = config["mandatory_development_gates"]
    gates = {
        "finite_complete_exact_accounting_and_clean_upstream": bool(
            complete and finite and exact and clean and pinned
        ),
        "independent_selected_q8_fraction_min": independent_selection is not None
        and independent_selection
        >= float(thresholds["independent_selected_q8_fraction_min"]),
        "shared_selected_q1_fraction_min": shared_selection is not None
        and shared_selection >= float(thresholds["shared_selected_q1_fraction_min"]),
        "independent_controller_relative_auc_vs_fixed_q8_min": independent_gain
        is not None
        and independent_gain
        >= float(
            thresholds["independent_controller_relative_auc_vs_fixed_q8_min"]
        ),
        "shared_controller_relative_auc_vs_fixed_q8_min": shared_gain is not None
        and shared_gain
        >= float(thresholds["shared_controller_relative_auc_vs_fixed_q8_min"]),
        "mixture_controller_relative_auc_vs_fixed_q8_min": mixture_gain is not None
        and mixture_gain
        >= float(thresholds["mixture_controller_relative_auc_vs_fixed_q8_min"]),
        "probe_message_fraction_max": probe_fraction is not None
        and probe_fraction <= float(thresholds["probe_message_fraction_max"]),
        "controller_wall_clock_overhead_fraction_max": overhead is not None
        and overhead
        <= float(thresholds["controller_wall_clock_overhead_fraction_max"]),
        "marginal_preserving_coupling_audit": bool(audit.get("pass")),
    }
    return {
        "experiment_id": config["experiment_id"],
        "run_count": len(records),
        "expected_run_count": len(expected),
        "independent_selected_q8_fraction": independent_selection,
        "shared_selected_q1_fraction": shared_selection,
        "independent_relative_auc_vs_fixed_q8": independent_gain,
        "shared_relative_auc_vs_fixed_q8": shared_gain,
        "mixture_relative_auc_vs_fixed_q8": mixture_gain,
        "maximum_probe_message_fraction": probe_fraction,
        "maximum_selection_overhead_fraction": overhead,
        "gates": gates,
        "all_mandatory_gates_pass": all(gates.values()),
        "decision": "authorize-confirmation-preregistration"
        if all(gates.values())
        else "stop",
        "records": records.to_dict(orient="records"),
    }


def aggregate_curves(curves: pd.DataFrame, checkpoints: int = 21) -> pd.DataFrame:
    grid = np.linspace(0.0, 1.0, checkpoints)
    rows = []
    for (coupling, method, seed), curve in curves.groupby(
        ["coupling", "method", "training_seed"]
    ):
        curve = curve.sort_values("budget_fraction")
        values = np.interp(
            grid,
            curve["budget_fraction"].to_numpy(float),
            curve["team_return"].to_numpy(float),
            left=float(curve["team_return"].iloc[0]),
            right=float(curve["team_return"].iloc[-1]),
        )
        rows.extend(
            {
                "coupling": coupling,
                "method": method,
                "training_seed": int(seed),
                "budget_fraction": float(fraction),
                "team_return": float(value),
            }
            for fraction, value in zip(grid, values)
        )
    samples = pd.DataFrame(rows)
    return (
        samples.groupby(["coupling", "method", "budget_fraction"], as_index=False)
        .agg(
            team_return_mean=("team_return", "mean"),
            team_return_std=("team_return", "std"),
            seeds=("training_seed", "nunique"),
        )
        .sort_values(["coupling", "method", "budget_fraction"])
    )


def plot(summary: pd.DataFrame, output: Path) -> None:
    regimes = list(summary["coupling"].drop_duplicates())
    fig, axes = plt.subplots(1, len(regimes), figsize=(7.15, 2.45), squeeze=False)
    for ax, coupling in zip(axes[0], regimes):
        cell = summary[summary["coupling"] == coupling]
        for method, curve in cell.groupby("method"):
            x = curve["budget_fraction"].to_numpy(float)
            mean = curve["team_return_mean"].to_numpy(float)
            std = curve["team_return_std"].fillna(0.0).to_numpy(float)
            n = curve["seeds"].to_numpy(float)
            ci = 1.96 * std / np.sqrt(n)
            ax.plot(x, mean, linewidth=1.8, label=method)
            ax.fill_between(x, mean - ci, mean + ci, alpha=0.14, linewidth=0)
        ax.set_title(f"{coupling.capitalize()} rollout streams")
        ax.set_xlabel("Charged budget fraction")
        ax.grid(color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Deterministic team return")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    tsp_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=tsp_root / "experiments" / "marl_probe_commit_development.json",
    )
    parser.add_argument(
        "--audit", type=Path, default=tsp_root / "internal" / "marl_coupling_audit.json"
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records, curves = collect_records(args.root, config)
    summary = aggregate_curves(curves)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    plot(summary, args.figure)
    gate = evaluate_gates(args.root, args.config, args.audit)
    args.gate_json.parent.mkdir(parents=True, exist_ok=True)
    args.gate_json.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": gate["decision"], "runs": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
