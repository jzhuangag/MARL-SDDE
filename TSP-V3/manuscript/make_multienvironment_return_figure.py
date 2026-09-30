"""Build the two-environment MAPPO return figure from versioned summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    "fixed_q1": (r"fixed $q=1$", "#7f7f7f", "--", 1.0),
    "fixed_q2": (r"fixed $q=2$", "#56B4E9", "--", 1.0),
    "fixed_q4": (r"fixed $q=4$", "#009E73", "--", 1.0),
    "fixed_q8": (r"fixed $q=8$", "#E69F00", "--", 1.0),
    "lyapunov_probe_commit": ("Lyapunov controller", "#0072B2", "-", 2.1),
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
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)
    ax.set_title(title, fontsize=7.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Charged budget fraction")
    ax.grid(color="#dddddd", linewidth=0.5)


def main() -> None:
    mpe = pd.read_csv(MPE)
    mamujoco = pd.read_csv(MAMUJOCO)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 1.95))
    draw(
        axes[0],
        mpe[mpe.coupling.eq("independent")],
        "(a) MPE, independent streams",
    )
    draw(
        axes[1],
        mpe[mpe.coupling.eq("shared")],
        "(b) MPE, shared streams",
    )
    draw(
        axes[2],
        mamujoco[mamujoco.coupling.eq("independent")],
        "(c) MaMuJoCo, independent streams",
    )
    draw(
        axes[3],
        mamujoco[mamujoco.coupling.eq("shared")],
        "(d) MaMuJoCo, shared streams",
    )
    axes[0].set_ylabel("MPE return")
    axes[2].set_ylabel("MaMuJoCo return")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=0.55)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
