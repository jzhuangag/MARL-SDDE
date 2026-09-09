"""Aggregate deterministic MAPPO team-return curves on a charged budget axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    args = parser.parse_args()
    result = aggregate(args.root)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.csv, index=False)
    plot(result, args.figure)
    print(json.dumps({"rows": len(result), "csv": str(args.csv), "figure": str(args.figure)}, sort_keys=True))


if __name__ == "__main__":
    main()
