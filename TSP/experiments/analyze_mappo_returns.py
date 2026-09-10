"""Aggregate deterministic MAPPO team-return curves on a charged budget axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _usable_updates(metadata: dict) -> int:
    return min(
        int(metadata["message_budget"])
        // int(metadata["message_cost_per_update"]),
        int(metadata["environment_budget"])
        // int(metadata["environment_cost_per_update"]),
    )


def read_run(metadata_path: Path) -> pd.DataFrame:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    progress_path = metadata_path.with_name("progress.txt")
    rows = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        actor_transitions, team_return = line.split(",")
        updates = float(actor_transitions) / (
            metadata["q_rollout_workers"] * metadata["rollout_length"]
        )
        rows.append(
            {
                "resource_fraction": updates / metadata["usable_updates"],
                "team_return": float(team_return),
            }
        )
    if not rows:
        raise ValueError(f"no evaluation rows in {progress_path}")
    frame = pd.DataFrame(rows)
    frame["seed"] = metadata["seed"]
    frame["coupling"] = metadata["coupling"]
    frame["q"] = metadata["q_rollout_workers"]
    frame["critic_lr"] = metadata["critic_lr"]
    frame["method"] = metadata.get(
        "method", f"q={metadata['q_rollout_workers']}, eta={metadata['critic_lr']:g}"
    )
    return frame


def run_auc(metadata_path: Path, checkpoints: int = 21) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run = read_run(metadata_path).sort_values("resource_fraction")
    grid = np.linspace(0.0, 1.0, checkpoints)
    x = run["resource_fraction"].to_numpy(float)
    y = run["team_return"].to_numpy(float)
    values = np.interp(grid, x, y, left=y[0], right=y[-1])
    expected_updates = _usable_updates(metadata)
    accounting_ok = all(
        (
            int(metadata["usable_updates"]) == expected_updates,
            int(metadata["charged_training_messages"])
            == expected_updates * int(metadata["message_cost_per_update"]),
            int(metadata["charged_training_environment_ticks"])
            == expected_updates * int(metadata["environment_cost_per_update"]),
            int(metadata["charged_training_messages"])
            <= int(metadata["message_budget"]),
            int(metadata["charged_training_environment_ticks"])
            <= int(metadata["environment_budget"]),
        )
    )
    return {
        "metadata_path": str(metadata_path.resolve()),
        "coupling": metadata["coupling"],
        "q": int(metadata["q_rollout_workers"]),
        "critic_lr": float(metadata["critic_lr"]),
        "seed": int(metadata["seed"]),
        "return_auc": float(
            np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid))
        ),
        "terminal_return": float(values[-1]),
        "finite": bool(np.isfinite(values).all()),
        "accounting_ok": accounting_ok,
        "upstream_modified": bool(metadata["upstream_modified"]),
        "harl_commit": metadata["harl_commit"],
    }


def evaluate_development_gate(
    root: Path, config_path: Path, audit_path: Path, checkpoints: int = 21
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records = [run_auc(path, checkpoints) for path in root.rglob("tsp_bridge_metadata.json")]

    actions = config["fixed_action_grid"]
    expected = {
        (coupling, int(q), float(lr), int(seed))
        for coupling in config["coupling_regimes"]
        for q in actions["q_rollout_workers"]
        for lr in actions["critic_lr"]
        for seed in config["development_seed_registry"]["stage_d0_seeds"]
    }
    observed = [
        (row["coupling"], row["q"], row["critic_lr"], row["seed"])
        for row in records
    ]
    complete = len(observed) == len(set(observed)) and set(observed) == expected
    finite = bool(records) and all(row["finite"] for row in records)
    accounting = bool(records) and all(row["accounting_ok"] for row in records)
    clean = bool(records) and all(not row["upstream_modified"] for row in records)
    pinned = bool(records) and all(
        row["harl_commit"] == config["upstream"]["commit"] for row in records
    )

    frame = pd.DataFrame(records)
    action_table = (
        frame.groupby(["q", "critic_lr"], as_index=False)["return_auc"].mean()
        if complete and finite
        else pd.DataFrame()
    )
    if action_table.empty:
        strong = None
        oracle_rows = []
        baseline_auc = oracle_auc = headroom = float("nan")
        regimes_improve = False
    else:
        strong_row = action_table.loc[action_table["return_auc"].idxmax()]
        strong = {
            "q": int(strong_row["q"]),
            "critic_lr": float(strong_row["critic_lr"]),
            "mean_return_auc": float(strong_row["return_auc"]),
        }
        oracle_rows = []
        baseline_values = []
        oracle_values = []
        for coupling in config["coupling_regimes"]:
            cell = frame[frame["coupling"] == coupling]
            oracle = cell.loc[cell["return_auc"].idxmax()]
            baseline = cell[
                (cell["q"] == strong["q"])
                & np.isclose(cell["critic_lr"], strong["critic_lr"])
            ]["return_auc"].mean()
            oracle_rows.append(
                {
                    "coupling": coupling,
                    "q": int(oracle["q"]),
                    "critic_lr": float(oracle["critic_lr"]),
                    "return_auc": float(oracle["return_auc"]),
                    "strong_fixed_return_auc": float(baseline),
                    "strict_improvement": bool(oracle["return_auc"] > baseline),
                }
            )
            baseline_values.append(float(baseline))
            oracle_values.append(float(oracle["return_auc"]))
        baseline_auc = float(np.mean(baseline_values))
        oracle_auc = float(np.mean(oracle_values))
        headroom = (oracle_auc - baseline_auc) / max(abs(baseline_auc), 1e-12)
        regimes_improve = all(row["strict_improvement"] for row in oracle_rows)

    threshold = float(
        config["mandatory_stage_d0_gates"][
            "oracle_auc_headroom_over_global_strong_fixed_min"
        ]
    )
    gates = {
        "finite_and_complete": complete and finite,
        "exact_dual_budget_accounting": accounting,
        "upstream_worktree_clean": clean and pinned,
        "oracle_auc_headroom_over_global_strong_fixed_min": bool(
            np.isfinite(headroom) and headroom >= threshold
        ),
        "oracle_improves_each_coupling_regime": regimes_improve,
        "shared_and_independent_worker_marginals_match": bool(audit.get("pass")),
    }
    return {
        "experiment_id": config["experiment_id"],
        "analysis_role": config["role"],
        "run_count": len(records),
        "expected_run_count": len(expected),
        "auc_definition": (
            f"trapezoid over {checkpoints} equally spaced feasible-horizon fractions; "
            "relative headroom divides by absolute strong-fixed AUC"
        ),
        "strong_global_fixed": strong,
        "per_regime_oracle": oracle_rows,
        "strong_fixed_mean_auc": baseline_auc,
        "oracle_mean_auc": oracle_auc,
        "oracle_relative_headroom": headroom,
        "headroom_threshold": threshold,
        "gates": gates,
        "all_mandatory_gates_pass": all(gates.values()),
        "decision": "authorize-controller-design" if all(gates.values()) else "stop",
        "run_records": records,
    }


def aggregate(root: Path, checkpoints: int = 21) -> pd.DataFrame:
    runs = [read_run(path) for path in root.rglob("tsp_bridge_metadata.json")]
    if not runs:
        raise FileNotFoundError(f"no TSP bridge runs under {root}")
    grid = np.linspace(0.0, 1.0, checkpoints)
    interpolated = []
    for run in runs:
        run = run.sort_values("resource_fraction")
        x = run["resource_fraction"].to_numpy(float)
        y = run["team_return"].to_numpy(float)
        # Hold the first/last deterministic evaluation outside the observed grid.
        values = np.interp(grid, x, y, left=y[0], right=y[-1])
        for fraction, value in zip(grid, values):
            interpolated.append(
                {
                    "resource_fraction": fraction,
                    "team_return": value,
                    "seed": int(run["seed"].iloc[0]),
                    "coupling": run["coupling"].iloc[0],
                    "method": run["method"].iloc[0],
                }
            )
    samples = pd.DataFrame(interpolated)
    result = (
        samples.groupby(["coupling", "method", "resource_fraction"], as_index=False)
        .agg(team_return_mean=("team_return", "mean"), team_return_std=("team_return", "std"), seeds=("seed", "nunique"))
        .sort_values(["coupling", "method", "resource_fraction"])
    )
    result["team_return_ci95"] = 1.96 * result["team_return_std"].fillna(0.0) / np.sqrt(result["seeds"])
    return result


def plot(summary: pd.DataFrame, output: Path) -> None:
    regimes = list(summary["coupling"].drop_duplicates())
    fig, axes = plt.subplots(1, len(regimes), figsize=(7.15, 2.45), squeeze=False)
    for ax, coupling in zip(axes[0], regimes):
        cell = summary[summary["coupling"] == coupling]
        for method, curve in cell.groupby("method"):
            x = curve["resource_fraction"].to_numpy(float)
            mean = curve["team_return_mean"].to_numpy(float)
            ci = curve["team_return_ci95"].to_numpy(float)
            ax.plot(x, mean, linewidth=1.8, label=method)
            ax.fill_between(x, mean - ci, mean + ci, alpha=0.14, linewidth=0)
        ax.set_title(f"{coupling.capitalize()} rollout streams")
        ax.set_xlabel("Charged budget fraction")
        ax.grid(color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Deterministic team return")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=min(4, len(labels)), loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    tsp_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=tsp_root / "tmp" / "mr")
    parser.add_argument("--csv", type=Path, default=tsp_root / "tmp" / "marl_return_summary.csv")
    parser.add_argument("--figure", type=Path, default=tsp_root / "tmp" / "marl_return_curves.pdf")
    parser.add_argument(
        "--config",
        type=Path,
        default=tsp_root / "experiments" / "marl_return_development_grid.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=tsp_root / "internal" / "marl_coupling_audit.json",
    )
    parser.add_argument(
        "--gate-json", type=Path, default=tsp_root / "tmp" / "marl_return_gate.json"
    )
    args = parser.parse_args()
    result = aggregate(args.root)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.csv, index=False)
    plot(result, args.figure)
    gate = evaluate_development_gate(args.root, args.config, args.audit)
    args.gate_json.parent.mkdir(parents=True, exist_ok=True)
    args.gate_json.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(result),
                "csv": str(args.csv),
                "figure": str(args.figure),
                "gate_json": str(args.gate_json),
                "decision": gate["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
