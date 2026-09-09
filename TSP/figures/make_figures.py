from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
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
