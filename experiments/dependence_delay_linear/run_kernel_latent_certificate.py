"""Pilot and formal runner for unknown-baseline kernel certification."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dual_anytime_controller import (
    mixture_upper_confidence,
    rounded_upper,
    select_dual_action,
)
from kernel_latent_certificate import (
    kernel_latent_rho_upper,
    lazy_joint_tv_upper,
    minimum_kernel_gap,
    periodic_rbf_independent_mean,
    sample_kernel_probe,
)
from predictable_mixing_controller import (
    RESOURCE_BUDGET,
    SERVER_OVERHEAD,
    exact_policy_metrics,
)


MIXING_EIGENVALUES = (0.0, 0.8, 0.96)
CORRELATIONS = (0.0, 0.5, 0.9)
DELAYS = (0, 2)
LENGTHSCALE = 0.35
INITIAL_MIXING_TRIALS = 128
BLOCK_BUDGET = 2000
TARGET_TV = 0.01
ALPHA_MIXING = 0.01 / 3.0
ALPHA_SIMILARITY = 0.01 / 3.0
ALPHA_CONTROL = 0.01 / 3.0
PAIR_OVERHEAD = SERVER_OVERHEAD + 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "kernel_latent_certificate_smoke",
    )
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20270311)
    return parser.parse_args()


def run_certificate(
    mixing: float,
    rho: float,
    seed: int,
    action_cache: Dict[Tuple[object, ...], Dict[str, float]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rng = np.random.RandomState(seed)
    states = rng.random_sample(3)
    previous_first = None
    mixing_trials = INITIAL_MIXING_TRIALS
    stays = int(rng.binomial(mixing_trials, mixing))
    similarity_sum = 0.0
    similarity_trials = 0
    similarity_bias_sum = 0.0
    control_sum = 0.0
    control_trials = 0
    control_bias_sum = 0.0
    spent = INITIAL_MIXING_TRIALS
    decision = 0
    jointly_covered = True
    rows: List[Dict[str, object]] = []
    estimate = {
        "rho_upper": 1.0,
        "baseline_lower": 0.0,
        "similarity_upper": 1.0,
    }
    while spent < RESOURCE_BUDGET:
        decision += 1
        mixing_upper = mixture_upper_confidence(
            stays, mixing_trials, ALPHA_MIXING
        )
        if similarity_trials > 0 and control_trials > 0:
            estimate = kernel_latent_rho_upper(
                similarity_sum,
                similarity_trials,
                similarity_bias_sum,
                control_sum,
                control_trials,
                control_bias_sum,
                ALPHA_SIMILARITY,
                ALPHA_CONTROL,
            )
        jointly_covered = bool(
            jointly_covered
            and mixing <= mixing_upper
            and rho <= estimate["rho_upper"]
        )
        certified_mixing = rounded_upper(mixing_upper, 0.002)
        gap = minimum_kernel_gap(certified_mixing, TARGET_TV)
        block = min(BLOCK_BUDGET, RESOURCE_BUDGET - spent)
        cost = gap + PAIR_OVERHEAD
        probes = 0 if cost > block else block // cost
        used = 0
        if probes > 0:
            similarity_bias = lazy_joint_tv_upper(
                certified_mixing, gap
            )
            control_bias = certified_mixing ** int(gap)
            for _ in range(int(probes)):
                sample = sample_kernel_probe(
                    rng,
                    states,
                    mixing,
                    rho,
                    gap,
                    LENGTHSCALE,
                    previous_first,
                )
                states = sample["states"]
                previous_first = sample["first"]
                similarity_sum += float(sample["similarity"])
                similarity_trials += 1
                similarity_bias_sum += similarity_bias
                if sample["control"] is not None:
                    control_sum += float(sample["control"])
                    control_trials += 1
                    control_bias_sum += control_bias
            raw = int(probes) * int(gap)
            stays += int(rng.binomial(raw, mixing))
            mixing_trials += raw
            used = int(probes) * int(cost)
        leftover = int(block) - used
        if leftover > 0:
            stays += int(rng.binomial(leftover, mixing))
            mixing_trials += leftover
        spent += int(block)
        rows.append(
            {
                "mixing_eigenvalue": mixing,
                "equivalent_persistence": 0.5 * (1.0 + mixing),
                "rho": rho,
                "seed": seed,
                "decision": decision,
                "mixing_upper": mixing_upper,
                "rho_upper_before": estimate["rho_upper"],
                "baseline_lower_before": estimate["baseline_lower"],
                "gap": gap,
                "probes": probes,
                "similarity_trials": similarity_trials,
                "control_trials": control_trials,
                "mixing_trials": mixing_trials,
                "jointly_covered_so_far": jointly_covered,
                "spent": spent,
            }
        )
    mixing_upper = mixture_upper_confidence(
        stays, mixing_trials, ALPHA_MIXING
    )
    if similarity_trials > 0 and control_trials > 0:
        estimate = kernel_latent_rho_upper(
            similarity_sum,
            similarity_trials,
            similarity_bias_sum,
            control_sum,
            control_trials,
            control_bias_sum,
            ALPHA_SIMILARITY,
            ALPHA_CONTROL,
        )
    jointly_covered = bool(
        jointly_covered
        and mixing <= mixing_upper
        and rho <= estimate["rho_upper"]
    )
    true_persistence = 0.5 * (1.0 + mixing)
    certified_persistence = rounded_upper(
        0.5 * (1.0 + mixing_upper), 0.002
    )
    certified_rho = rounded_upper(estimate["rho_upper"], 0.02)
    action_rows = {}
    for delay in DELAYS:
        key = (certified_persistence, certified_rho, delay)
        if key not in action_cache:
            action_cache[key] = select_dual_action(
                certified_persistence,
                certified_rho,
                delay,
                RESOURCE_BUDGET,
            )
        action = action_cache[key]
        exact = exact_policy_metrics(
            action, true_persistence, rho, delay
        )
        action_rows[delay] = {
            "q": int(action["num_agents"]),
            "eta": float(action["eta"]),
            "updates": int(action["updates"]),
            "exact_radius": float(exact["exact_radius"]),
        }
    return (
        {
            "mixing_eigenvalue": mixing,
            "equivalent_persistence": true_persistence,
            "rho": rho,
            "seed": seed,
            "joint_simultaneous_coverage": jointly_covered,
            "final_mixing_upper": mixing_upper,
            "final_rho_upper": estimate["rho_upper"],
            "similarity_upper": estimate["similarity_upper"],
            "baseline_lower": estimate["baseline_lower"],
            "similarity_trials": similarity_trials,
            "control_trials": control_trials,
            **{
                f"d{delay}_{name}": value
                for delay, action in action_rows.items()
                for name, value in action.items()
            },
        },
        rows,
    )


def run_experiment(
    num_seeds: int, base_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    action_cache: Dict[Tuple[object, ...], Dict[str, float]] = {}
    metrics: List[Dict[str, object]] = []
    traces: List[Dict[str, object]] = []
    scenario = 0
    for mixing in MIXING_EIGENVALUES:
        for rho in CORRELATIONS:
            for index in range(num_seeds):
                seed = base_seed + scenario * 10000 + index
                metric, rows = run_certificate(
                    mixing, rho, seed, action_cache
                )
                metrics.append(metric)
                traces.extend(rows)
            scenario += 1
    return pd.DataFrame(metrics), pd.DataFrame(traces)


def evaluate(metrics: pd.DataFrame) -> Dict[str, object]:
    baseline_true = periodic_rbf_independent_mean(LENGTHSCALE)
    covered = metrics[metrics["joint_simultaneous_coverage"]]
    scenario = (
        metrics.groupby(["mixing_eigenvalue", "rho"], as_index=False)
        .agg(
            rho_upper=("final_rho_upper", "median"),
            baseline_lower=("baseline_lower", "median"),
            probes=("similarity_trials", "median"),
            q_d0=("d0_q", "median"),
            q_d2=("d2_q", "median"),
        )
    )
    ordered = all(
        np.all(np.diff(group.sort_values("rho")["rho_upper"]) >= 0)
        for _, group in scenario.groupby("mixing_eigenvalue")
    )
    fast = scenario[scenario["mixing_eigenvalue"] == 0.0].set_index(
        "rho"
    )
    persistent = scenario[scenario["mixing_eigenvalue"] == 0.96]
    q_response = {}
    for delay in DELAYS:
        column = f"q_d{delay}"
        pivot = scenario.pivot(
            index="mixing_eigenvalue", columns="rho", values=column
        )
        q_response[delay] = {
            "weak": int((pivot[0.9] <= pivot[0.0]).sum()),
            "strict": int((pivot[0.9] < pivot[0.0]).sum()),
        }
    updating_radii = []
    for delay in DELAYS:
        updating = covered[covered[f"d{delay}_updates"] > 0]
        updating_radii.extend(
            updating[f"d{delay}_exact_radius"].tolist()
        )
    gates = {
        "joint_anytime_coverage": {
            "value": float(
                metrics["joint_simultaneous_coverage"].mean()
            ),
            "pass": bool(
                metrics["joint_simultaneous_coverage"].mean() >= 0.975
            ),
        },
        "conditional_action_safety": {
            "maximum_updating_radius": float(max(updating_radii)),
            "pass": bool(max(updating_radii) < 1.0),
        },
        "correlation_ordering": {"pass": bool(ordered)},
        "fast_mixing_identification": {
            "rho0_upper": float(fast.loc[0.0, "rho_upper"]),
            "rho05_upper": float(fast.loc[0.5, "rho_upper"]),
            "rho09_upper": float(fast.loc[0.9, "rho_upper"]),
            "pass": bool(
                fast.loc[0.0, "rho_upper"] <= 0.3
                and 0.5 <= fast.loc[0.5, "rho_upper"] <= 0.9
                and fast.loc[0.9, "rho_upper"] >= 0.9
            ),
        },
        "baseline_learning": {
            "true_baseline": baseline_true,
            "minimum_fast_baseline_lower": float(
                fast["baseline_lower"].min()
            ),
            "maximum_baseline_lower": float(
                scenario["baseline_lower"].max()
            ),
            "pass": bool(
                fast["baseline_lower"].min() >= 0.02
                and scenario["baseline_lower"].max()
                <= baseline_true + 1e-12
            ),
        },
        "persistent_probe_nonvacuity": {
            "minimum_median_probes": float(persistent["probes"].min()),
            "pass": bool(persistent["probes"].min() >= 50),
        },
        "participation_response": {
            "by_delay": q_response,
            "pass": bool(
                all(item["weak"] == 3 for item in q_response.values())
                and any(
                    item["strict"] >= 1 for item in q_response.values()
                )
            ),
        },
    }
    scientific_names = (
        "correlation_ordering",
        "fast_mixing_identification",
        "baseline_learning",
        "persistent_probe_nonvacuity",
        "participation_response",
    )
    return {
        "gates": gates,
        "validity_pass": bool(
            gates["joint_anytime_coverage"]["pass"]
            and gates["conditional_action_safety"]["pass"]
        ),
        "scientific_passes": int(
            sum(gates[name]["pass"] for name in scientific_names)
        ),
        "overall_pass": bool(
            gates["joint_anytime_coverage"]["pass"]
            and gates["conditional_action_safety"]["pass"]
            and sum(gates[name]["pass"] for name in scientific_names) >= 4
        ),
        "scenario_summary": scenario.to_dict(orient="records"),
    }


def save_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    scenario = (
        metrics.groupby(["mixing_eigenvalue", "rho"], as_index=False)
        .agg(
            rho_upper=("final_rho_upper", "median"),
            baseline_lower=("baseline_lower", "median"),
            probes=("similarity_trials", "median"),
            q_d0=("d0_q", "median"),
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for mixing, marker in ((0.0, "o"), (0.8, "s"), (0.96, "^")):
        group = scenario[scenario["mixing_eigenvalue"] == mixing]
        axes[0].plot(
            group["rho"],
            group["rho_upper"],
            marker=marker,
            label=f"$\\lambda={mixing:g}$",
        )
        axes[1].plot(
            group["rho"],
            group["baseline_lower"],
            marker=marker,
        )
    axes[0].plot((0, 1), (0, 1), "k--", linewidth=1)
    axes[1].axhline(
        periodic_rbf_independent_mean(LENGTHSCALE),
        color="black",
        linestyle="--",
        linewidth=1,
    )
    group = scenario.groupby("rho", as_index=False).agg(
        q=("q_d0", "median")
    )
    axes[2].plot(group["rho"], group["q"], marker="o")
    axes[0].set_ylabel("median kernel $\\rho$ upper bound")
    axes[1].set_ylabel("median baseline lower bound")
    axes[2].set_ylabel("median certified $q$, $D=0$")
    for axis in axes:
        axis.set_xlabel("true $\\rho$")
        axis.grid(alpha=0.3)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "kernel_latent_summary.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, traces = run_experiment(args.num_seeds, args.base_seed)
    evaluation = evaluate(metrics)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    traces.to_csv(args.output_dir / "traces.csv", index=False)
    save_figure(metrics, args.output_dir)
    summary = {
        "experiment": "EXP-012B-pilot"
        if args.num_seeds < 64
        else "EXP-012B",
        "num_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "lengthscale": LENGTHSCALE,
        **evaluation,
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
