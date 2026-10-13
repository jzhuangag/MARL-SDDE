"""Plot controlled CPU evidence without altering outcomes or smoothening data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"TSP-V3"/"manuscript"))
from figure_style import apply_publication_style, style_axis, set_panel_label_below


def plot(source: Path) -> None:
    apply_publication_style()
    frame = pd.read_csv(source/"curves.csv")
    styles = {
        "finite_horizon": ("Observable finite-horizon", "#0072B2", "-", 2.),
        "correlation_only": ("Correlation-only", "#D55E00", "--", 1.3),
        "fixed_q1": (r"Oracle fixed $q=1$", "#666666", "--", 1.1),
        "fixed_q2": (r"Oracle fixed $q=2$", "#009E73", "-.", 1.1),
        "fixed_q4": (r"Oracle fixed $q=4$", "#CC79A7", ":", 1.1),
        "fixed_q8": (r"Oracle fixed $q=8$", "#E69F00", "--", 1.1),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.8), sharex=True)
    grid = np.linspace(0, 1, 101)
    for ax, rho, letter in zip(axes, (0., .5, 1.), "abc"):
        cell = frame[frame.rho.eq(rho)]
        for method, (label, color, line, width) in styles.items():
            curves = []
            for _, part in cell[cell.method.eq(method)].groupby("seed"):
                part = part.sort_values("budget_fraction")
                curves.append(np.interp(grid, part.budget_fraction, part.risk, left=float(part.target.iloc[0])**2))
            values = np.stack(curves)
            mean = values.mean(axis=0)
            ax.plot(grid, mean, label=label, color=color, linestyle=line, linewidth=width)
            if method == "finite_horizon":
                radius = 1.96*values.std(axis=0, ddof=1)/np.sqrt(len(values))
                ax.fill_between(grid, np.maximum(mean-radius, 1e-12), mean+radius, color=color, alpha=.14, linewidth=0.)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xticks([0,.25,.5,.75,1], ["0.00","0.25","0.50","0.75","1.00"])
        ax.set_xlabel("Charged dual-budget fraction")
        set_panel_label_below(ax, rf"({letter}) Spatial correlation $\rho={rho:g}$", y=-.31)
        style_axis(ax)
    axes[0].set_ylabel("Conditional terminal MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7.0)
    fig.subplots_adjust(left=.085, right=.99, bottom=.26, top=.72, wspace=.30)
    fig.savefig(source/"conditional_risk_curves.png", dpi=240)
    fig.savefig(source/"conditional_risk_curves.svg", metadata={"Date": None})
    plt.close(fig)

    certificates = pd.read_csv(source/"certificates.csv")
    population = certificates.groupby(["rho", "persistence"]).agg(
        selected_q=("selected_q", "mean"), oracle_q=("oracle_q", "mean"),
        parameter_coverage=("joint_covered", "mean"), regret=("regret", "mean"),
        regret_upper=("regret_upper", "mean"),
    ).reset_index()
    population.to_csv(source/"mechanism_summary.csv", index=False, float_format="%.17g", lineterminator="\n")
    print(json.dumps({"figure": str(source/"conditional_risk_curves.png"),
                      "caption": "Exact conditional population MSE after random charged calibration. Bands describe calibration-seed variation, not deep-MARL return uncertainty. Fixed-q comparators have free true-context step/spacing and full budgets."}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    plot(parser.parse_args().source)
