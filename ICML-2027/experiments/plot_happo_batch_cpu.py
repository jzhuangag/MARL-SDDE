"""Plot every registered holdout method and the separately charged search work."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot(root, out):
    if out.exists():
        raise FileExistsError("fresh figure directory required")
    out.mkdir(parents=True)
    plan = json.loads((Path(__file__).parent / "happo_batch_cpu_preregistration.json").read_text())
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.linewidth": .7,
                         "axes.grid": True, "grid.color": "0.65", "grid.alpha": .25,
                         "grid.linewidth": .6, "legend.frameon": False})
    labels = {"happo_cyclic": "HAPPO cyclic", "happo_full_cadence": "HAPPO full cadence",
              "parallel_cyclic": "Fresh parallel", "stale_cyclic": "Stale cyclic",
              "age_first": "Age first", "history_surrogate": "History surrogate",
              "outcome_aware_oracle": "Outcome-aware menu (extra search)"}
    styles = [("#4d4d4d", "--"), ("#e69f00", "-."), ("#009e73", ":"),
              ("#cc79a7", "--"), ("#56b4e9", "-."), ("#7b3294", ":"), ("#0072b2", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.55))
    table = []
    for ax, task in zip(axes, plan["tasks"]):
        for method, (color, style) in zip(plan["methods"], styles):
            curves = [read_csv(root / f"{task['name']}_{seed}_{method}/curves.csv")
                      for seed in plan["holdout_seeds"]]
            x = np.array([float(r["resource_fraction"]) for r in curves[0]])
            ys = np.array([[float(r["mean_return"]) for r in curve] for curve in curves])
            ax.plot(x, ys.mean(axis=0), color=color, linestyle=style, label=labels[method],
                    linewidth=1.9 if method == "outcome_aware_oracle" else 1.2)
            for index, fraction in enumerate(x):
                table.append({"task": task["name"], "method": method, "training_fraction": fraction,
                              "mean_return": float(ys[:, index].mean()),
                              "seed_min": float(ys[:, index].min()), "seed_max": float(ys[:, index].max())})
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlabel("Training joint-transition fraction")
        ax.set_ylabel("Native team episodic return")
        ax.text(.5, -.30, "(a) Spread" if task["name"] == "spread" else "(b) Reference",
                ha="center", transform=ax.transAxes)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, fontsize=8)
    fig.subplots_adjust(top=.73, bottom=.27, left=.075, right=.985, wspace=.28)
    fig.savefig(out / "holdout_return_curves.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / "holdout_return_curves.png", dpi=220)
    plt.close(fig)
    with (out / "holdout_curve_data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    keys = ["training_joint_transitions", "benchmark_evaluation_joint_transitions",
            "oracle_search_joint_transitions", "oracle_diagnostic_joint_transitions"]
    cost_labels = ["Training", "Benchmark evaluation", "Search", "Independent diagnostic"]
    colors = ["#009e73", "#999999", "#e69f00", "#56b4e9"]
    for ax, task in zip(axes, plan["tasks"]):
        totals = [np.mean([[json.loads((root / f"{task['name']}_{seed}_{method}/scientific.json").read_text())["totals"][key]
                            for key in keys] for seed in plan["holdout_seeds"]], axis=0)
                  for method in plan["methods"]]
        bottom = np.zeros(len(totals))
        for col, label, color in zip(range(len(keys)), cost_labels, colors):
            values = np.array(totals)[:, col]
            ax.bar(range(len(totals)), values, bottom=bottom, color=color, label=label, width=.72)
            bottom += values
        ax.set_ylim(0, 5e5)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_xticks(range(len(totals)), ["Cyclic", "Cadence", "Parallel", "Stale", "Age", "History", "Menu"], rotation=30, ha="right")
        ax.set_ylabel("Charged joint transitions")
        ax.text(.5, -.34, "(a) Spread" if task["name"] == "spread" else "(b) Reference",
                ha="center", transform=ax.transAxes)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=4, fontsize=8)
    fig.subplots_adjust(top=.86, bottom=.29, left=.08, right=.985, wspace=.28)
    fig.savefig(out / "charged_search_cost.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / "charged_search_cost.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    plot(args.root, args.out)
