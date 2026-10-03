"""Audit the confirmed MPE controller against the complete fixed-q grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_mappo_probe_commit import aggregate_curves, collect_records


def relative_gap(value: float, reference: float) -> float:
    return float((value - reference) / max(abs(reference), 1e-12))


def collect(existing_root: Path, extension_root: Path, config: dict):
    records = []
    curves = []
    for root in (existing_root, extension_root):
        frame, curve = collect_records(
            root, config, int(config["evaluation"]["auc_grid_points"])
        )
        records.append(frame)
        curves.append(curve)
    records = pd.concat(records, ignore_index=True)
    curves = pd.concat(curves, ignore_index=True)
    allowed_methods = set(config["existing_methods"] + config["new_methods"])
    records = records[
        records.method.isin(allowed_methods)
        & records.coupling.isin(config["coupling_regimes"])
        & records.training_seed.isin(config["training_seeds"])
    ].copy()
    curves = curves[
        curves.method.isin(allowed_methods)
        & curves.coupling.isin(config["coupling_regimes"])
        & curves.training_seed.isin(config["training_seeds"])
    ].copy()
    return records, curves


def evaluate(
    existing_root: Path,
    extension_root: Path,
    config_path: Path,
    replay_identical: bool,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records, _ = collect(existing_root, extension_root, config)
    methods = config["existing_methods"] + config["new_methods"]
    expected = {
        (regime, method, seed)
        for regime in config["coupling_regimes"]
        for method in methods
        for seed in config["training_seeds"]
    }
    observed = list(
        records[["coupling", "method", "training_seed"]].itertuples(
            index=False, name=None
        )
    )
    complete = len(observed) == len(set(observed)) and set(observed) == expected
    finite = bool(np.isfinite(records.return_auc.to_numpy(float)).all())
    pinned = bool(
        (records.harl_commit == config["upstream"]["commit"]).all()
        and (~records.upstream_modified).all()
        and records.accounting_ok.all()
    )

    metrics = {}
    if complete:
        means = records.groupby(["coupling", "method"]).return_auc.mean()
        fixed = [f"fixed_q{q}" for q in config["fixed_q_grid"]]
        envelopes = {}
        controller_gaps = {}
        oracle_actions = {}
        for regime in config["coupling_regimes"]:
            candidates = {method: float(means.loc[(regime, method)]) for method in fixed}
            oracle_method = sorted(candidates, key=lambda m: (-candidates[m], m))[0]
            envelope = candidates[oracle_method]
            controller = float(means.loc[(regime, "lyapunov_probe_commit")])
            envelopes[regime] = envelope
            oracle_actions[regime] = int(oracle_method.replace("fixed_q", ""))
            controller_gaps[regime] = relative_gap(controller, envelope)
        static_means = {
            method: float(
                records[records.method == method].return_auc.mean()
            )
            for method in fixed
        }
        strong_method = sorted(
            static_means, key=lambda m: (-static_means[m], m)
        )[0]
        controller_mixture = float(
            records[records.method == "lyapunov_probe_commit"].return_auc.mean()
        )
        metrics = {
            "cell_means": {
                f"{regime}/{method}": float(value)
                for (regime, method), value in means.items()
            },
            "fixed_q_envelope": envelopes,
            "fixed_q_oracle_action": oracle_actions,
            "controller_relative_gap_to_fixed_q_envelope": controller_gaps,
            "strong_static_method": strong_method,
            "strong_static_return_auc": static_means[strong_method],
            "controller_mixture_return_auc": controller_mixture,
            "controller_relative_gain_over_descriptive_strong_static": relative_gap(
                controller_mixture, static_means[strong_method]
            ),
        }

    threshold = float(
        config["audit_gates"][
            "controller_relative_gap_to_fixed_q_envelope_min_each_regime"
        ]
    )
    gates = {
        "finite_complete_exact_accounting_and_clean_upstream": bool(
            complete and finite and pinned
        ),
        "controller_relative_gap_to_fixed_q_envelope_min_each_regime": bool(
            metrics
            and all(
                value >= threshold
                for value in metrics[
                    "controller_relative_gap_to_fixed_q_envelope"
                ].values()
            )
        ),
        "all_registered_fixed_q_present": bool(
            complete
            and set(records[records.method.str.startswith("fixed_q")].selected_q)
            == set(config["fixed_q_grid"])
        ),
        "analysis_replay_byte_identical": bool(replay_identical),
    }
    passed = all(gates.values())
    return {
        "experiment_id": config["experiment_id"],
        "run_count": len(records),
        "expected_run_count": len(expected),
        **metrics,
        "gates": gates,
        "pass": passed,
        "decision": "retain-baseline-complete-mpe-transfer"
        if passed
        else "withdraw-strong-mpe-baseline-claim",
        "records": records.to_dict(orient="records"),
    }


def plot(summary: pd.DataFrame, output: Path) -> None:
    styles = {
        "fixed_q1": ("q=1", "#7f7f7f", "--", 1.0),
        "fixed_q2": ("q=2", "#56B4E9", "--", 1.0),
        "fixed_q4": ("q=4", "#009E73", "--", 1.0),
        "fixed_q8": ("q=8", "#E69F00", "--", 1.0),
        "lyapunov_probe_commit": ("Lyapunov controller", "#0072B2", "-", 2.2),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35), sharey=True)
    for ax, regime in zip(axes, ("independent", "shared")):
        for method in styles:
            frame = summary[
                (summary.coupling == regime) & (summary.method == method)
            ].sort_values("budget_fraction")
            label, color, line, width = styles[method]
            x = frame["budget_fraction"].to_numpy(float)
            mean = frame["team_return_mean"].to_numpy(float)
            ci = 1.96 * frame["team_return_std"].fillna(0).to_numpy(float) / np.sqrt(
                frame["seeds"].to_numpy(float)
            )
            ax.plot(x, mean, label=label, color=color, linestyle=line, linewidth=width)
            if method == "lyapunov_probe_commit":
                ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)
        ax.set_title(regime.capitalize())
        ax.set_xlabel("Charged budget fraction")
        ax.grid(color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Evaluation team return")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    tsp = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=tsp / "experiments" / "marl_mpe_qgrid_audit.json",
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--replay-identical", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, curves = collect(args.existing_root, args.extension_root, config)
    summary = aggregate_curves(curves, int(config["evaluation"]["auc_grid_points"]))
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    plot(summary, args.figure)
    result = evaluate(
        args.existing_root, args.extension_root, args.config, args.replay_identical
    )
    args.gate_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": result["decision"], "runs": result["run_count"]}))


if __name__ == "__main__":
    main()
