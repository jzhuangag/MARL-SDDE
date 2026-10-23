"""Regenerate the three controlled-study figures with the TSP-V3 style."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import (
    COLUMN_WIDTH_IN,
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
OUT = Path(__file__).resolve().parent / "figures"

INDEPENDENT_COLOR = "#0072B2"
INDEPENDENT_DELAY_COLOR = "#56B4E9"
SHARED_COLOR = "#D55E00"
SHARED_DELAY_COLOR = "#E69F00"


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def learning_value_figure() -> None:
    source = (
        ROOT
        / "experiments"
        / "dependence_delay_linear"
        / "results"
        / "exp016b_formal_20260801"
        / "analysis"
        / "core_results.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    names = [
        "Layer A",
        "Affine TD",
        "Delay\nactive",
        "Message\nbinding",
        "Environment\nbinding",
    ]
    values = np.asarray(
        [
            payload["primary_layer_A"]["relative_difference"],
            payload["layer_B"]["relative_difference"],
            payload["subset_results"]["delay_active"]["relative_difference"],
            payload["subset_results"]["message_binding"]["relative_difference"],
            payload["subset_results"]["environment_binding"]["relative_difference"],
        ]
    )

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.02))
    x = np.arange(len(values))
    bars = ax.bar(
        x,
        100.0 * values,
        width=0.67,
        color=[PROPOSED_COLOR, Q_COLORS[4], PROPOSED_COLOR, Q_COLORS[32], Q_COLORS[4]],
    )
    ax.axhline(
        3.0,
        color="#4D4D4D",
        linestyle="--",
        linewidth=1.0,
        label="Practical-effect threshold",
    )
    ax.set_ylabel("Risk reduction (%)")
    ax.set_xticks(x, names)
    ax.set_ylim(0.0, max(85.0, 100.0 * values.max() * 1.16))
    style_axis(ax, grid_axis="y")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            100.0 * value + 1.25,
            f"{100.0 * value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    ax.legend(frameon=False, loc="upper left", handlelength=2.5)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.25, top=0.96)
    save(fig, "exp016b_risk_reduction")


def joint_action_figure() -> None:
    persistence = np.asarray([0.0, 0.9, 0.98])
    q_independent = np.asarray([[16, 4], [32, 4], [16, 4]])
    q_shared = np.asarray([[1, 1], [1, 1], [1, 1]])
    gaps_independent = np.asarray([[64, 56], [250, 205], [1185, 1036]])
    gaps_shared = np.asarray([[50, 50], [183, 183], [924, 924]])

    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, 2.12))
    x = np.arange(len(persistence))
    width = 0.18
    labels = (
        r"Independent, $D=0$",
        r"Independent, $D=8$",
        r"Shared, $D=0$",
        r"Shared, $D=8$",
    )
    colors = (
        INDEPENDENT_COLOR,
        INDEPENDENT_DELAY_COLOR,
        SHARED_COLOR,
        SHARED_DELAY_COLOR,
    )
    axes[0].bar(x - 1.5 * width, q_independent[:, 0], width, color=colors[0], label=labels[0])
    axes[0].bar(x - 0.5 * width, q_independent[:, 1], width, color=colors[1], label=labels[1])
    axes[0].bar(x + 0.5 * width, q_shared[:, 0], width, color=colors[2], label=labels[2])
    axes[0].bar(x + 1.5 * width, q_shared[:, 1], width, color=colors[3], label=labels[3])
    set_panel_label_below(axes[0], "(a) Participation", y=-0.43)
    axes[0].set_ylabel(r"Selected $q$")
    axes[0].set_xticks(x, ["0", "0.9", "0.98"])
    axes[0].set_xlabel("Temporal persistence")
    style_axis(axes[0], grid_axis="y")

    axes[1].plot(persistence, gaps_independent[:, 0], "o-", color=colors[0], label=labels[0], markersize=3.1)
    axes[1].plot(persistence, gaps_independent[:, 1], "s--", color=colors[1], label=labels[1], markersize=3.1)
    axes[1].plot(persistence, gaps_shared[:, 0], "o-", color=colors[2], label=labels[2], markersize=3.1)
    axes[1].plot(persistence, gaps_shared[:, 1], "s--", color=colors[3], label=labels[3], markersize=3.1)
    set_panel_label_below(axes[1], "(b) Spacing", y=-0.43)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"Selected $b$")
    axes[1].set_xlabel("Temporal persistence")
    axes[1].set_xticks(persistence, ["0", "0.9", "0.98"])
    style_axis(axes[1])

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.0),
        columnspacing=0.9,
        handlelength=2.0,
        fontsize=6.6,
    )
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.34, top=0.70, wspace=0.43)
    save(fig, "exp010b_joint_actions")


def convergence_curve_figure() -> None:
    source = ROOT / "TSP" / "data" / "convergence_curve_summary.csv"
    data = pd.read_csv(source)
    data = data[
        (np.isclose(data["persistence"], 0.9))
        & (data["maximum_delay"] == 0)
        & data["policy"].isin(["joint", "q4", "q1", "q32"])
    ]
    styles = {
        "joint": ("Joint certificate", PROPOSED_COLOR, "-", 2.0),
        "q4": (r"Strong fixed $q=4$", Q_COLORS[4], Q_LINESTYLES[4], 1.45),
        "q1": (r"Fixed $q=1$", Q_COLORS[1], Q_LINESTYLES[1], 1.35),
        "q32": (r"Fixed $q=32$", Q_COLORS[32], Q_LINESTYLES[32], 1.45),
    }
    titles = (
        r"(a) Parameter, $\rho=0$",
        r"(b) Parameter, $\rho=0.9$",
        r"(c) Return estimate, $\rho=0$",
        r"(d) Return estimate, $\rho=0.9$",
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(TEXT_WIDTH_IN, DOUBLE_FIGURE_HEIGHT_IN),
        sharex=True,
    )
    for column, rho in enumerate((0.0, 0.9)):
        cell = data[np.isclose(data["rho"], rho)]
        for policy, (label, color, linestyle, linewidth) in styles.items():
            curve = cell[cell["policy"] == policy].sort_values("resource_fraction")
            x = curve["resource_fraction"].to_numpy(dtype=float)
            for row, (mean_name, ci_name) in enumerate(
                (("parameter_mean", "parameter_ci95"), ("return_mean", "return_ci95"))
            ):
                ax = axes[2 * row + column]
                mean = curve[mean_name].to_numpy(dtype=float)
                ci = curve[ci_name].fillna(0.0).to_numpy(dtype=float)
                ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
                ax.fill_between(
                    x,
                    np.maximum(mean - ci, 1e-8),
                    mean + ci,
                    color=color,
                    alpha=CONFIDENCE_ALPHA,
                    linewidth=0.0,
                )
    for ax, title in zip(axes, titles):
        set_panel_label_below(ax, title)
        ax.set_yscale("log")
        set_fraction_axis(ax, label="Charged budget fraction")
        style_axis(ax)
    axes[0].set_ylabel("Normalized error")
    axes[2].set_ylabel("Normalized error")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.5,
        handlelength=2.5,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.31, top=0.69, wspace=0.55)
    save(fig, "convergence_curves")


def main() -> None:
    apply_publication_style()
    learning_value_figure()
    joint_action_figure()
    convergence_curve_figure()


if __name__ == "__main__":
    main()
