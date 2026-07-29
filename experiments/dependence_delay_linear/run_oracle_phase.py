"""Run EXP-006A, the finite-budget oracle participation phase diagram."""

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oracle_phase import (
    BUDGETS,
    DEPENDENCE_PATHS,
    ERROR_AMPLITUDES,
    MAX_DELAYS,
    OVERHEADS,
    STRENGTHS,
    build_surface,
    evaluate_gates,
    find_actionable_rectangles,
    summarize_delay,
    summarize_tracks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "oracle_phase",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def plot_outputs(surface: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 8,
            "axes.grid": False,
        }
    )
    subset = surface[
        (surface["overhead"] == 4)
        & (surface["max_delay"] == 4)
        & (surface["path"].isin(["global", "clustered", "balanced"]))
        & (surface["strength"].isin([0.0, 0.8]))
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 9.0), sharex=True)
    image = None
    for row, path in enumerate(("global", "clustered", "balanced")):
        for column, strength in enumerate((0.0, 0.8)):
            cell = subset[
                (subset["path"] == path)
                & (subset["strength"] == strength)
            ]
            pivot = cell.pivot(
                index="error_amplitude",
                columns="budget",
                values="selected_q",
            ).reindex(
                index=list(ERROR_AMPLITUDES),
                columns=list(BUDGETS),
            )
            image = axes[row, column].imshow(
                np.log2(pivot.to_numpy(dtype=float)),
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=5,
                cmap="viridis",
            )
            axes[row, column].set_title(
                "{0}, strength={1:.1f}".format(path, strength)
            )
            axes[row, column].set_yticks(range(len(ERROR_AMPLITUDES)))
            axes[row, column].set_yticklabels(ERROR_AMPLITUDES)
            axes[row, column].set_xticks(range(len(BUDGETS)))
            axes[row, column].set_xticklabels(BUDGETS, rotation=35)
            axes[row, column].set_ylabel("Current error amplitude")
            axes[row, column].set_xlabel("Decision budget")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    colorbar.set_ticks(range(6))
    colorbar.set_ticklabels([1, 2, 4, 8, 16, 32])
    colorbar.set_label("Oracle-selected agents")
    fig.suptitle("Oracle participation phase diagram (D=4, overhead=4)")
    fig.savefig(output_dir / "fig1_oracle_phase.png", bbox_inches="tight")
    plt.close(fig)

    frequency = (
        surface.groupby(["path", "selected_q"])
        .size()
        .groupby(level=0)
        .apply(lambda values: values / values.sum())
        .rename("fraction")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.24
    q_values = np.asarray([1, 2, 4, 8, 16, 32], dtype=float)
    x = np.arange(len(q_values))
    for index, path in enumerate(("global", "clustered", "balanced")):
        values = (
            frequency[frequency["path"] == path]
            .set_index("selected_q")["fraction"]
            .reindex(q_values, fill_value=0.0)
        )
        ax.bar(x + (index - 1) * width, values, width, label=path)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(q)) for q in q_values])
    ax.set_xlabel("Oracle-selected agents")
    ax.set_ylabel("Fraction of cells")
    ax.set_title("Oracle action occupancy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_q_frequency.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        paths = ("global", "clustered")
        strengths = (0.0, 0.8)
        budgets = (500, 2000)
        errors = (0.03, 0.3)
        max_delays = (4, 16)
        overheads = (4,)
    else:
        paths = tuple(DEPENDENCE_PATHS)
        strengths = STRENGTHS
        budgets = BUDGETS
        errors = ERROR_AMPLITUDES
        max_delays = MAX_DELAYS
        overheads = OVERHEADS
    expected_rows = (
        len(paths)
        * len(strengths)
        * len(budgets)
        * len(errors)
        * len(max_delays)
        * len(overheads)
    )
    print(
        "Starting EXP-006A with {0} oracle cells".format(expected_rows),
        flush=True,
    )
    surface = build_surface(
        paths=paths,
        strengths=strengths,
        budgets=budgets,
        errors=errors,
        max_delays=max_delays,
        overheads=overheads,
    )
    tracks = summarize_tracks(surface)
    delay_summary = summarize_delay(surface)
    if args.smoke:
        rectangles = pd.DataFrame(
            columns=[
                "path",
                "budget_low",
                "budget_high",
                "error_low",
                "error_high",
                "pass",
            ]
        )
        gates = {"status": "SMOKE_NOT_SCIENTIFIC_EVIDENCE"}
    else:
        rectangles = find_actionable_rectangles(surface)
        gates = evaluate_gates(
            surface,
            tracks,
            delay_summary,
            rectangles,
            expected_rows=9720,
        )
    surface.to_csv(args.output_dir / "oracle_surface.csv", index=False)
    tracks.to_csv(args.output_dir / "track_summary.csv", index=False)
    delay_summary.to_csv(
        args.output_dir / "delay_summary.csv", index=False
    )
    rectangles.to_csv(
        args.output_dir / "actionable_rectangles.csv", index=False
    )
    plot_outputs(surface, args.output_dir)
    summary = {
        "experiment_id": "EXP-006A-oracle-participation-phase",
        "status": (
            "SMOKE_NOT_SCIENTIFIC_EVIDENCE"
            if args.smoke
            else "COMPLETED_PENDING_REPRODUCTION"
        ),
        "expected_rows": expected_rows,
        "grid": {
            "paths": list(paths),
            "strengths": list(strengths),
            "budgets": list(budgets),
            "error_amplitudes": list(errors),
            "max_delays": list(max_delays),
            "overheads": list(overheads),
        },
        "gates": gates,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(gates, indent=2), flush=True)
    print("Outputs written to {0}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
