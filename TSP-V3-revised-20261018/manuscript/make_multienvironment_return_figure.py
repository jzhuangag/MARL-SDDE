"""Build the two-environment MAPPO return figure from versioned summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import (
    CONFIDENCE_ALPHA,
    DOUBLE_FIGURE_HEIGHT_IN,
    LEGEND_FONT_SIZE,
    PDF_METADATA,
    PROPOSED_COLOR,
    Q_COLORS,
    Q_LINESTYLES,
    TEXT_WIDTH_IN,
    apply_publication_style,
    set_fraction_axis,
    set_panel_label_below,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[2]
MPE = ROOT / "TSP" / "internal" / "marl_mpe_qgrid_audit_curves.csv"
MAMUJOCO = (
    ROOT
    / "TSP-V3"
    / "results"
    / "marl_mamujoco_calibrated_confirmation_20260930"
    / "summary.csv"
)
OUTPUT = Path(__file__).resolve().parent / "figures" / "marl_multienvironment_returns.pdf"

STYLES = {
    "fixed_q1": (r"Fixed $q=1$", Q_COLORS[1], Q_LINESTYLES[1], 1.25),
    "fixed_q2": (r"Fixed $q=2$", Q_COLORS[2], Q_LINESTYLES[2], 1.25),
    "fixed_q4": (r"Fixed $q=4$", Q_COLORS[4], Q_LINESTYLES[4], 1.25),
    "fixed_q8": (r"Fixed $q=8$", Q_COLORS[8], Q_LINESTYLES[8], 1.25),
    "lyapunov_probe_commit": ("Lyapunov controller", PROPOSED_COLOR, "-", 2.0),
}


def draw(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    for method, (label, color, linestyle, linewidth) in STYLES.items():
        part = frame[frame.method.eq(method)].sort_values("budget_fraction")
        x = part.budget_fraction.to_numpy(float)
        mean = part.team_return_mean.to_numpy(float)
        ci = 1.96 * part.team_return_std.to_numpy(float) / np.sqrt(
            part.seeds.to_numpy(float)
        )
        ax.plot(
            x,
            mean,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        if method == "lyapunov_probe_commit":
            ax.fill_between(
                x,
                mean - ci,
                mean + ci,
                color=color,
                alpha=CONFIDENCE_ALPHA,
                linewidth=0.0,
            )
    set_panel_label_below(ax, title)
    set_fraction_axis(ax, label="Charged budget fraction")
    style_axis(ax)


def main() -> None:
    mpe = pd.read_csv(MPE)
    mamujoco = pd.read_csv(MAMUJOCO)
    apply_publication_style()
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(TEXT_WIDTH_IN, DOUBLE_FIGURE_HEIGHT_IN),
        sharex=True,
    )
    draw(
        axes[0],
        mpe[mpe.coupling.eq("independent")],
        "(a) MPE, independent",
    )
    draw(
        axes[1],
        mpe[mpe.coupling.eq("shared")],
        "(b) MPE, shared",
    )
    draw(
        axes[2],
        mamujoco[mamujoco.coupling.eq("independent")],
        "(c) MaMuJoCo, independent",
    )
    draw(
        axes[3],
        mamujoco[mamujoco.coupling.eq("shared")],
        "(d) MaMuJoCo, shared",
    )
    axes[0].set_ylabel("MPE return")
    axes[2].set_ylabel("MaMuJoCo return")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.25,
        handlelength=2.5,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.31, top=0.69, wspace=0.55)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT,
        metadata=PDF_METADATA,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
