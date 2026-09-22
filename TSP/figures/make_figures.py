from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"
PURPLE = "#CC79A7"


def save(fig: plt.Figure, name: str) -> None:
    pdf_metadata = {
        "Creator": "MARL-SDDE TSP figure pipeline",
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    png_metadata = {"Software": "MARL-SDDE TSP figure pipeline"}
    fig.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight",
        metadata=pdf_metadata,
    )
    fig.savefig(
        OUT / f"{name}.png",
        dpi=240,
        bbox_inches="tight",
        metadata=png_metadata,
    )
    plt.close(fig)


def learning_value_figure() -> None:
    source = ROOT / "experiments" / "dependence_delay_linear" / "results" / "exp016b_formal_20260801" / "analysis" / "core_results.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    names = ["Layer A", "Affine TD", "Delay active", "Message\nbinding", "Environment\nbinding"]
    values = [
        payload["primary_layer_A"]["relative_difference"],
        payload["layer_B"]["relative_difference"],
        payload["subset_results"]["delay_active"]["relative_difference"],
        payload["subset_results"]["message_binding"]["relative_difference"],
        payload["subset_results"]["environment_binding"]["relative_difference"],
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    bars = ax.bar(np.arange(len(values)), 100 * np.asarray(values), color=[BLUE, GREEN, BLUE, ORANGE, GREEN])
    ax.axhline(3.0, color=GRAY, linestyle="--", linewidth=1.0, label="registered practical threshold")
    ax.set_ylabel("Risk reduction (percent)")
    ax.set_xticks(np.arange(len(values)), names)
    ax.set_ylim(0, max(85, 100 * max(values) * 1.16))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, 100 * value + 1.3, f"{100 * value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "exp016b_risk_reduction")


def joint_action_figure() -> None:
    persistence = np.asarray([0.0, 0.9, 0.98])
    q_independent = np.asarray([[16, 4], [32, 4], [16, 4]])
    q_shared = np.asarray([[1, 1], [1, 1], [1, 1]])
    gaps_independent = np.asarray([[64, 56], [250, 205], [1185, 1036]])
    gaps_shared = np.asarray([[50, 50], [183, 183], [924, 924]])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    x = np.arange(len(persistence))
    width = 0.18
    axes[0].bar(x - 1.5 * width, q_independent[:, 0], width, color=BLUE, label=r"independent, $D=0$")
    axes[0].bar(x - 0.5 * width, q_independent[:, 1], width, color="#56B4E9", label=r"independent, $D=8$")
    axes[0].bar(x + 0.5 * width, q_shared[:, 0], width, color=ORANGE, label=r"shared, $D=0$")
    axes[0].bar(x + 1.5 * width, q_shared[:, 1], width, color="#E69F00", label=r"shared, $D=8$")
    axes[0].set_ylabel("Selected participation $q$")
    axes[0].set_xticks(x, ["0", "0.9", "0.98"])
    axes[0].set_xlabel("Temporal persistence")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_axisbelow(True)
    axes[1].plot(persistence, gaps_independent[:, 0], "o-", color=BLUE, label=r"independent, $D=0$")
    axes[1].plot(persistence, gaps_independent[:, 1], "s--", color="#56B4E9", label=r"independent, $D=8$")
    axes[1].plot(persistence, gaps_shared[:, 0], "o-", color=ORANGE, label=r"shared, $D=0$")
    axes[1].plot(persistence, gaps_shared[:, 1], "s--", color="#E69F00", label=r"shared, $D=8$")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Selected spacing $b$")
    axes[1].set_xlabel("Temporal persistence")
    axes[1].grid(color="#dddddd", linewidth=0.7)
    axes[1].set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
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
        "joint": ("Joint certificate", "#000000", "-", 2.0),
        "q4": (r"Strong fixed $q=4$", ORANGE, "--", 1.6),
        "q1": (r"Fixed $q=1$", GREEN, "-.", 1.3),
        "q32": (r"Fixed $q=32$", PURPLE, ":", 1.6),
    }
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.35), sharex=True)
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
                ax.plot(
                    x,
                    mean,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
                ax.fill_between(
                    x,
                    np.maximum(mean - ci, 1e-8),
                    mean + ci,
                    color=color,
                    alpha=0.10,
                    linewidth=0.0,
                )
        axes[column].set_title(rf"Param., $\rho={rho:g}$", fontsize=8.5)
        axes[2 + column].set_title(rf"Ret. est., $\rho={rho:g}$", fontsize=8.5)
    for ax in axes:
        ax.set_xlabel("Resource fraction")
        ax.set_yscale("log")
        ax.set_xlim(0.0, 1.0)
        ax.grid(color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Normalized error")
    axes[2].set_ylabel("Normalized error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.6)
    save(fig, "convergence_curves")


def marl_return_figure() -> None:
    """Render the frozen MAPPO confirmation curves in a compact two-panel form."""

    source = ROOT / "TSP" / "internal" / "marl_probe_commit_conf1_curve_summary.csv"
    data = pd.read_csv(source)
    styles = {
        "lyapunov_probe_commit": ("Lyapunov controller", "#000000", "-", 1.8),
        "fixed_q1": (r"Fixed $q=1$", GREEN, "-.", 1.35),
        "fixed_q8": (r"Fixed $q=8$", ORANGE, "--", 1.45),
    }
    regimes = ("independent", "shared")
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 1.78), sharex=True)
    for ax, coupling in zip(axes, regimes):
        cell = data[data["coupling"] == coupling]
        for method, (label, color, linestyle, linewidth) in styles.items():
            curve = cell[cell["method"] == method].sort_values("budget_fraction")
            x = curve["budget_fraction"].to_numpy(dtype=float)
            mean = curve["team_return_mean"].to_numpy(dtype=float)
            std = curve["team_return_std"].fillna(0.0).to_numpy(dtype=float)
            count = curve["seeds"].to_numpy(dtype=float)
            ci = 1.96 * std / np.sqrt(count)
            ax.plot(
                x,
                mean,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            ax.fill_between(
                x,
                mean - ci,
                mean + ci,
                color=color,
                alpha=0.11,
                linewidth=0.0,
            )
        ax.set_title(f"{coupling.capitalize()} streams", fontsize=8.5)
        ax.set_xlabel("Charged budget fraction")
        ax.set_xlim(0.0, 1.0)
        ax.grid(color="#dddddd", linewidth=0.65)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Team return")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=7.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82), w_pad=0.9)
    save(fig, "marl_probe_commit_return_curves_compact")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    learning_value_figure()
    joint_action_figure()
    convergence_curve_figure()
    marl_return_figure()
