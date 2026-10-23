"""Run preregistered EXP-010A multistate certificate transfer."""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multistate_certificate import (
    CORRELATIONS_TRANSFER,
    DELAYS_TRANSFER,
    PERSISTENCES,
    RESOURCE_BUDGET,
    build_transfer_mrp,
    candidate_actions,
    exact_pair_tv,
    generate_unit_paths,
    select_action,
    simulate_certified_action,
)
from run_state_correlation import bootstrap_ratio


POLICIES = ("joint", "q1", "q32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "multistate_certificate_transfer",
    )
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20261030)
    parser.add_argument("--bootstrap-replications", type=int, default=2000)
    return parser.parse_args()


def build_actions() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[float, dict]]:
    candidate_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    models = {}
    for persistence in PERSISTENCES:
        model = build_transfer_mrp(persistence)
        models[persistence] = model
        for maximum_delay in DELAYS_TRANSFER:
            for rho in CORRELATIONS_TRANSFER:
                candidates = candidate_actions(model, rho, maximum_delay)
                for row in candidates:
                    candidate_rows.append(
                        {"persistence": persistence, **row}
                    )
                choices = {
                    "joint": select_action(candidates),
                    "q1": select_action(candidates, restricted_q=1),
                    "q32": select_action(candidates, restricted_q=32),
                }
                for policy, action in choices.items():
                    selected_rows.append(
                        {
                            "persistence": persistence,
                            "policy": policy,
                            **action,
                        }
                    )
    return (
        pd.DataFrame(candidate_rows),
        pd.DataFrame(selected_rows),
        models,
    )


def independently_iterated_tv(model: dict, gap: int) -> float:
    distribution = np.eye(model["pair_transition"].shape[0])
    for _ in range(int(gap)):
        distribution = distribution.dot(model["pair_transition"])
    return float(
        0.5
        * np.max(
            np.sum(
                np.abs(distribution - model["pair_weights"]), axis=1
            )
        )
    )


def audit_actions(
    selected: pd.DataFrame, models: Dict[float, dict]
) -> pd.DataFrame:
    rows = []
    for action in selected.to_dict("records"):
        model = models[float(action["persistence"])]
        direct = exact_pair_tv(model, int(action["gap"]))
        iterated = independently_iterated_tv(model, int(action["gap"]))
        rows.append(
            {
                "persistence": action["persistence"],
                "rho": action["rho"],
                "maximum_delay": action["maximum_delay"],
                "policy": action["policy"],
                "direct_pair_tv": direct,
                "iterated_pair_tv": iterated,
                "absolute_tv_difference": abs(direct - iterated),
                "stationary_state_residual": float(
                    np.max(
                        np.abs(
                            model["stationary"].dot(model["transition"])
                            - model["stationary"]
                        )
                    )
                ),
                "stationary_pair_residual": float(
                    np.max(
                        np.abs(
                            model["pair_weights"].dot(
                                model["pair_transition"]
                            )
                            - model["pair_weights"]
                        )
                    )
                ),
                "transition_row_residual": float(
                    np.max(
                        np.abs(model["transition"].sum(axis=1) - 1.0)
                    )
                ),
                "pair_row_residual": float(
                    np.max(
                        np.abs(
                            model["pair_transition"].sum(axis=1) - 1.0
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def run_simulations(
    selected: pd.DataFrame,
    models: Dict[float, dict],
    num_seeds: int,
    base_seed: int,
) -> pd.DataFrame:
    rows = []
    selected_lookup = {
        (
            float(row["persistence"]),
            float(row["rho"]),
            int(row["maximum_delay"]),
            str(row["policy"]),
        ): row
        for row in selected.to_dict("records")
    }
    for persistence in PERSISTENCES:
        model = models[persistence]
        for seed_index in range(num_seeds):
            seed = base_seed + seed_index
            streams = generate_unit_paths(seed, model)
            for maximum_delay in DELAYS_TRANSFER:
                for rho in CORRELATIONS_TRANSFER:
                    for policy in POLICIES:
                        action = selected_lookup[
                            (persistence, rho, maximum_delay, policy)
                        ]
                        result = simulate_certified_action(
                            streams, model, action
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "seed_index": seed_index,
                                "persistence": persistence,
                                "rho": rho,
                                "maximum_delay": maximum_delay,
                                "policy": policy,
                                "num_agents": int(action["num_agents"]),
                                "gap": int(action["gap"]),
                                "eta": float(action["eta"]),
                                "updates": int(action["updates"]),
                                "update_cost": int(action["update_cost"]),
                                **result,
                            }
                        )
            if (seed_index + 1) % 4 == 0:
                print(
                    "persistence {0:g}: completed {1}/{2} paired seeds".format(
                        persistence, seed_index + 1, num_seeds
                    ),
                    flush=True,
                )
    return pd.DataFrame(rows)


def build_comparisons(
    metrics: pd.DataFrame, replications: int, base_seed: int
) -> pd.DataFrame:
    rows = []
    scenario_keys = ["persistence", "rho", "maximum_delay"]
    for scenario_index, (key, group) in enumerate(
        metrics.groupby(scenario_keys, sort=True)
    ):
        indexed = {
            policy: group[group["policy"] == policy]
            .sort_values("seed_index")["squared_parameter_error"]
            .to_numpy(dtype=float)
            for policy in POLICIES
        }
        for baseline_index, baseline in enumerate(("q1", "q32")):
            rows.append(
                {
                    "persistence": key[0],
                    "rho": key[1],
                    "maximum_delay": key[2],
                    "numerator": "joint",
                    "denominator": baseline,
                    **bootstrap_ratio(
                        indexed["joint"],
                        indexed[baseline],
                        replications,
                        base_seed
                        + 10_000
                        + 10 * scenario_index
                        + baseline_index,
                    ),
                }
            )
    return pd.DataFrame(rows)


def evaluate_gates(
    selected: pd.DataFrame,
    audit: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    num_seeds: int,
) -> Dict[str, object]:
    stationarity_max = float(
        audit[
            [
                "stationary_state_residual",
                "stationary_pair_residual",
                "transition_row_residual",
                "pair_row_residual",
            ]
        ].to_numpy().max()
    )
    numerical = {
        "stochasticity_stationarity": {
            "pass": bool(
                stationarity_max <= 1e-12
                and (selected["monotonicity"] > 0.0).all()
                and (selected["effective_monotonicity"] > 0.0).all()
            ),
            "maximum_residual": stationarity_max,
            "minimum_monotonicity": float(selected["monotonicity"].min()),
            "minimum_effective_monotonicity": float(
                selected["effective_monotonicity"].min()
            ),
        },
        "certificate_accounting_numerics": {
            "pass": bool(
                (selected["sharp_factor"] < 1.0).all()
                and (selected["eta"] > 0.0).all()
                and metrics["finite"].all()
                and metrics["within_budget"].all()
                and not metrics["diverged"].any()
                and len(metrics) == num_seeds * 12 * len(POLICIES)
            ),
            "rows": int(len(metrics)),
            "expected_rows": int(num_seeds * 12 * len(POLICIES)),
            "maximum_sharp_factor": float(selected["sharp_factor"].max()),
            "divergences": int(metrics["diverged"].sum()),
        },
        "independent_tv_reproduction": {
            "pass": bool(
                audit["absolute_tv_difference"].max() <= 1e-12
            ),
            "maximum_absolute_difference": float(
                audit["absolute_tv_difference"].max()
            ),
        },
    }
    joint = selected[selected["policy"] == "joint"].copy()
    pivot_q = joint.pivot(
        index=["persistence", "maximum_delay"],
        columns="rho",
        values="num_agents",
    )
    weak_q = int((pivot_q[0.9] <= pivot_q[0.0]).sum())
    strict_q = int((pivot_q[0.9] < pivot_q[0.0]).sum())
    gaps = joint.groupby("persistence")["gap"].median()
    pivot_eta = joint.pivot(
        index=["persistence", "rho"],
        columns="maximum_delay",
        values="eta",
    )
    delay_nonincrease = int((pivot_eta[8] <= pivot_eta[0] + 1e-15).sum())
    mean_errors = (
        metrics.groupby(
            ["persistence", "rho", "maximum_delay", "policy"],
            as_index=False,
        )["squared_parameter_error"]
        .mean()
        .pivot(
            index=["persistence", "rho", "maximum_delay"],
            columns="policy",
            values="squared_parameter_error",
        )
    )
    beats_both = int(
        (
            (mean_errors["joint"] < mean_errors["q1"])
            & (mean_errors["joint"] < mean_errors["q32"])
        ).sum()
    )
    ratios = mean_errors["joint"] / mean_errors[["q1", "q32"]].min(axis=1)
    scientific = {
        "nonvacuity": {
            "pass": bool(
                (joint["updates"] >= 50).all() and (joint["eta"] > 0).all()
            ),
            "minimum_updates": int(joint["updates"].min()),
            "minimum_eta": float(joint["eta"].min()),
        },
        "correlation_response": {
            "pass": bool(weak_q == 6 and strict_q >= 4),
            "weak_decreases": weak_q,
            "strict_decreases": strict_q,
        },
        "mixing_response": {
            "pass": bool(gaps.loc[0.98] >= 4.0 * gaps.loc[0.0]),
            "median_gap_kappa_0": float(gaps.loc[0.0]),
            "median_gap_kappa_098": float(gaps.loc[0.98]),
            "ratio": float(gaps.loc[0.98] / gaps.loc[0.0]),
        },
        "delay_response": {
            "pass": bool(delay_nonincrease >= 5),
            "nonincreasing_cells": delay_nonincrease,
            "total_cells": 6,
        },
        "endpoint_value": {
            "pass": bool(beats_both >= 8 and ratios.max() <= 1.25),
            "beats_both_scenarios": beats_both,
            "maximum_ratio_to_better_endpoint": float(ratios.max()),
        },
    }
    numerical_passes = sum(int(row["pass"]) for row in numerical.values())
    scientific_passes = sum(int(row["pass"]) for row in scientific.values())
    return {
        "numerical": numerical,
        "scientific": scientific,
        "overall": {
            "pass": bool(numerical_passes == 3 and scientific_passes >= 4),
            "numerical_passes": numerical_passes,
            "scientific_passes": scientific_passes,
            "criterion": "3/3 numerical and at least 4/5 scientific gates",
        },
        "maximum_joint_to_endpoint_ratio": float(
            comparisons["ratio"].max()
        ),
    }


def plot_outputs(
    selected: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    joint = selected[selected["policy"] == "joint"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for maximum_delay in DELAYS_TRANSFER:
        for rho in CORRELATIONS_TRANSFER:
            curve = joint[
                (joint["maximum_delay"] == maximum_delay)
                & np.isclose(joint["rho"], rho)
            ]
            label = "D={0}, rho={1:g}".format(maximum_delay, rho)
            axes[0].plot(
                curve["persistence"],
                curve["num_agents"],
                marker="o",
                label=label,
            )
            axes[1].plot(
                curve["persistence"],
                curve["gap"],
                marker="o",
                label=label,
            )
    axes[0].set_yscale("log", base=2)
    axes[0].set_yticks((1, 2, 4, 8, 16, 32))
    axes[0].get_yaxis().set_major_formatter(
        matplotlib.ticker.ScalarFormatter()
    )
    axes[0].set_xlabel("Laziness kappa")
    axes[0].set_ylabel("Selected participation q")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Laziness kappa")
    axes[1].set_ylabel("Certified separation b")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_certified_actions.png", dpi=220)
    plt.close(fig)

    aggregate = (
        metrics.groupby(
            ["persistence", "rho", "maximum_delay", "policy"],
            as_index=False,
        )["squared_parameter_error"]
        .mean()
    )
    scenarios = aggregate[
        ["persistence", "rho", "maximum_delay"]
    ].drop_duplicates()
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    width = 0.25
    for index, policy in enumerate(POLICIES):
        values = aggregate[aggregate["policy"] == policy][
            "squared_parameter_error"
        ].to_numpy()
        ax.bar(x + (index - 1) * width, values, width, label=policy)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "{0:g}/{1:g}/{2}".format(*row)
            for row in scenarios.to_numpy()
        ],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("kappa / rho / D")
    ax.set_ylabel("Mean squared parameter error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_endpoint_errors.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Building exact multistate certificates", flush=True)
    candidates, selected, models = build_actions()
    audit = audit_actions(selected, models)
    print(
        "Starting EXP-010A with {0} paired seeds".format(args.num_seeds),
        flush=True,
    )
    metrics = run_simulations(
        selected, models, args.num_seeds, args.base_seed
    )
    comparisons = build_comparisons(
        metrics, args.bootstrap_replications, args.base_seed
    )
    gates = evaluate_gates(
        selected, audit, metrics, comparisons, args.num_seeds
    )
    candidates.to_csv(args.output_dir / "candidate_actions.csv", index=False)
    selected.to_csv(args.output_dir / "selected_actions.csv", index=False)
    audit.to_csv(args.output_dir / "certificate_audit.csv", index=False)
    metrics.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    comparisons.to_csv(
        args.output_dir / "paired_bootstrap_comparisons.csv", index=False
    )
    plot_outputs(selected, metrics, args.output_dir)
    summary = {
        "experiment_id": "EXP-010A-multistate-certificate-transfer",
        "status": "COMPLETED_PENDING_REPRODUCTION",
        "num_paired_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "bootstrap_replications": args.bootstrap_replications,
        "resource_budget": RESOURCE_BUDGET,
        "gates": gates,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": [
            "candidate_actions.csv",
            "selected_actions.csv",
            "certificate_audit.csv",
            "per_seed_metrics.csv",
            "paired_bootstrap_comparisons.csv",
            "fig1_certified_actions.png",
            "fig2_endpoint_errors.png",
            "summary.json",
        ],
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(gates, indent=2), flush=True)
    print("Outputs written to {0}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
