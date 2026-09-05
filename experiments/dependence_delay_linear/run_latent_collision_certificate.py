"""Pilot and formal runner for latent-collision correlation certification."""

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
from latent_collision_certificate import (
    latent_rho_upper,
    minimum_collision_gap,
    sample_hidden_collision,
    symmetric_joint_tv_upper,
)
from predictable_mixing_controller import (
    RESOURCE_BUDGET,
    SERVER_OVERHEAD,
    exact_policy_metrics,
)


PERSISTENCES = (0.5, 0.9, 0.98)
CORRELATIONS = (0.0, 0.5, 0.9)
DELAYS = (0, 2)
INITIAL_P_TRIALS = 128
BLOCK_BUDGET = 2000
TARGET_TV = 0.01
ALPHA_P = 0.005
ALPHA_RHO = 0.005
INDEPENDENT_COLLISION = 0.5
PAIR_OVERHEAD = SERVER_OVERHEAD + 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "latent_collision_certificate_smoke",
    )
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20270111)
    return parser.parse_args()


def run_certificate(
    persistence: float,
    rho: float,
    seed: int,
    action_cache: Dict[Tuple[object, ...], Dict[str, float]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rng = np.random.RandomState(seed)
    states = rng.randint(0, 2, size=3).astype(np.int64)
    p_trials = INITIAL_P_TRIALS
    stays = int(rng.binomial(p_trials, persistence))
    collisions = 0
    collision_trials = 0
    cumulative_bias = 0.0
    spent = INITIAL_P_TRIALS
    decision = 0
    jointly_covered = True
    rows: List[Dict[str, object]] = []
    rho_result = {
        "rho_upper": 1.0,
        "collision_upper": 1.0,
        "hoeffding_radius": 1.0,
        "average_tv_bias": 1.0,
    }
    while spent < RESOURCE_BUDGET:
        decision += 1
        p_upper = mixture_upper_confidence(
            stays, p_trials, ALPHA_P
        )
        if collision_trials > 0:
            rho_result = latent_rho_upper(
                collisions,
                collision_trials,
                cumulative_bias,
                ALPHA_RHO,
                INDEPENDENT_COLLISION,
            )
        jointly_covered = bool(
            jointly_covered
            and persistence <= p_upper
            and rho <= rho_result["rho_upper"]
        )
        certified_p = rounded_upper(p_upper, 0.002)
        gap = minimum_collision_gap(certified_p, TARGET_TV)
        block = min(BLOCK_BUDGET, RESOURCE_BUDGET - spent)
        cost = gap + PAIR_OVERHEAD
        probes = 0 if cost > block else block // cost
        used = 0
        if probes > 0:
            delta = symmetric_joint_tv_upper(certified_p, gap)
            for _ in range(int(probes)):
                sample = sample_hidden_collision(
                    rng, states, persistence, rho, gap
                )
                states = sample["states"]
                collisions += int(sample["collision"])
                collision_trials += 1
                cumulative_bias += delta
            raw_transitions = int(probes) * int(gap)
            stays += int(rng.binomial(raw_transitions, persistence))
            p_trials += raw_transitions
            used = int(probes) * int(cost)
        leftover = int(block) - used
        if leftover > 0:
            stays += int(rng.binomial(leftover, persistence))
            p_trials += leftover
        spent += int(block)
        rows.append(
            {
                "persistence": persistence,
                "rho": rho,
                "seed": seed,
                "decision": decision,
                "persistence_upper": p_upper,
                "certified_persistence": certified_p,
                "rho_upper_before": rho_result["rho_upper"],
                "gap": gap,
                "probes": probes,
                "collision_trials": collision_trials,
                "collisions": collisions,
                "cumulative_tv_bias": cumulative_bias,
                "p_trials": p_trials,
                "jointly_covered_so_far": jointly_covered,
                "spent": spent,
            }
        )
    p_upper = mixture_upper_confidence(stays, p_trials, ALPHA_P)
    if collision_trials > 0:
        rho_result = latent_rho_upper(
            collisions,
            collision_trials,
            cumulative_bias,
            ALPHA_RHO,
            INDEPENDENT_COLLISION,
        )
    jointly_covered = bool(
        jointly_covered
        and persistence <= p_upper
        and rho <= rho_result["rho_upper"]
    )
    certified_p = rounded_upper(p_upper, 0.002)
    certified_rho = rounded_upper(rho_result["rho_upper"], 0.02)
    action_rows = {}
    for delay in DELAYS:
        key = (certified_p, certified_rho, delay)
        if key not in action_cache:
            action_cache[key] = select_dual_action(
                certified_p,
                certified_rho,
                delay,
                RESOURCE_BUDGET,
            )
        action = action_cache[key]
        exact = exact_policy_metrics(
            action, persistence, rho, delay
        )
        action_rows[delay] = {
            "q": int(action["num_agents"]),
            "gap": int(action["gap"]),
            "eta": float(action["eta"]),
            "updates": int(action["updates"]),
            "exact_radius": float(exact["exact_radius"]),
        }
    return (
        {
            "persistence": persistence,
            "rho": rho,
            "seed": seed,
            "joint_simultaneous_coverage": jointly_covered,
            "final_p_upper": p_upper,
            "final_rho_upper": rho_result["rho_upper"],
            "collision_upper": rho_result["collision_upper"],
            "hoeffding_radius": rho_result["hoeffding_radius"],
            "average_tv_bias": rho_result["average_tv_bias"],
            "collision_trials": collision_trials,
            "p_trials": p_trials,
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
    for persistence in PERSISTENCES:
        for rho in CORRELATIONS:
            for index in range(num_seeds):
                seed = base_seed + scenario * 10000 + index
                metric, rows = run_certificate(
                    persistence, rho, seed, action_cache
                )
                metrics.append(metric)
                traces.extend(rows)
            scenario += 1
    return pd.DataFrame(metrics), pd.DataFrame(traces)


def evaluate(metrics: pd.DataFrame) -> Dict[str, object]:
    covered = metrics[metrics["joint_simultaneous_coverage"]]
    scenario = (
        metrics.groupby(["persistence", "rho"], as_index=False)
        .agg(
            rho_upper=("final_rho_upper", "median"),
            p_upper=("final_p_upper", "median"),
            probes=("collision_trials", "median"),
            q_d0=("d0_q", "median"),
            q_d2=("d2_q", "median"),
            radius_d0=("d0_exact_radius", "max"),
            radius_d2=("d2_exact_radius", "max"),
        )
    )
    ordered = all(
        np.all(np.diff(group.sort_values("rho")["rho_upper"]) >= 0)
        for _, group in scenario.groupby("persistence")
    )
    low = scenario[scenario["persistence"] == 0.5].set_index("rho")
    persistent = scenario[scenario["persistence"] == 0.98]
    q_response = {}
    for delay in DELAYS:
        column = f"q_d{delay}"
        pivot = scenario.pivot(
            index="persistence", columns="rho", values=column
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
        "correlation_ordering": {
            "pass": bool(ordered),
        },
        "fast_mixing_identification": {
            "rho0_upper": float(low.loc[0.0, "rho_upper"]),
            "rho05_upper": float(low.loc[0.5, "rho_upper"]),
            "rho09_upper": float(low.loc[0.9, "rho_upper"]),
            "pass": bool(
                low.loc[0.0, "rho_upper"] <= 0.25
                and 0.5 <= low.loc[0.5, "rho_upper"] <= 0.85
                and low.loc[0.9, "rho_upper"] >= 0.9
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
    return {
        "gates": gates,
        "validity_pass": bool(
            gates["joint_anytime_coverage"]["pass"]
            and gates["conditional_action_safety"]["pass"]
        ),
        "scientific_passes": int(
            sum(
                gates[name]["pass"]
                for name in (
                    "correlation_ordering",
                    "fast_mixing_identification",
                    "persistent_probe_nonvacuity",
                    "participation_response",
                )
            )
        ),
        "overall_pass": bool(
            gates["joint_anytime_coverage"]["pass"]
            and gates["conditional_action_safety"]["pass"]
            and sum(
                gates[name]["pass"]
                for name in (
                    "correlation_ordering",
                    "fast_mixing_identification",
                    "persistent_probe_nonvacuity",
                    "participation_response",
                )
            )
            >= 3
        ),
        "scenario_summary": scenario.to_dict(orient="records"),
    }


def save_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    scenario = (
        metrics.groupby(["persistence", "rho"], as_index=False)
        .agg(
            rho_upper=("final_rho_upper", "median"),
            probes=("collision_trials", "median"),
            q_d0=("d0_q", "median"),
            q_d2=("d2_q", "median"),
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for persistence, marker in ((0.5, "o"), (0.9, "s"), (0.98, "^")):
        group = scenario[scenario["persistence"] == persistence]
        axes[0].plot(
            group["rho"],
            group["rho_upper"],
            marker=marker,
            label=f"$p={persistence:g}$",
        )
        axes[1].plot(
            group["rho"],
            group["probes"],
            marker=marker,
        )
    axes[0].plot((0, 1), (0, 1), "k--", linewidth=1)
    for delay, marker in ((0, "o"), (2, "s")):
        group = scenario.groupby("rho", as_index=False).agg(
            q=(f"q_d{delay}", "median")
        )
        axes[2].plot(
            group["rho"],
            group["q"],
            marker=marker,
            label=f"$D={delay}$",
        )
    axes[0].set_ylabel("median latent $\\rho$ upper bound")
    axes[1].set_ylabel("median collision probes")
    axes[2].set_ylabel("median certified $q$")
    for axis in axes:
        axis.set_xlabel("true $\\rho$")
        axis.grid(alpha=0.3)
    axes[0].legend()
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "latent_collision_summary.png", dpi=180
    )
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
        "experiment": "EXP-012A-pilot"
        if args.num_seeds < 64
        else "EXP-012A",
        "num_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        **evaluation,
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
