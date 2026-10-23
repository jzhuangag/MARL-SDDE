"""Run preregistered EXP-011A correlation minimax phase audit."""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from correlation_speedup_lower_bound import (
    adaptive_budget_lower_bound,
    effective_speedup,
    fisher_information,
    information_per_cost,
    minimax_risk,
    observation_covariance,
    optimal_integer_participation,
)


AGENT_COUNTS = (1, 2, 4, 8, 16, 32)
CORRELATIONS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
OVERHEADS = (1.0, 4.0, 16.0, 64.0)
BUDGET = 128000.0
TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "correlation_minimax_phase",
    )
    return parser.parse_args()


def build_grid() -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for rho in CORRELATIONS:
        common = rho
        private = 1.0 - rho
        risk_one = minimax_risk(1, 1, common, private)
        for overhead in OVERHEADS:
            selected = optimal_integer_participation(
                AGENT_COUNTS, overhead, common, private
            )
            lower = adaptive_budget_lower_bound(
                BUDGET, AGENT_COUNTS, overhead, common, private
            )
            for q in AGENT_COUNTS:
                covariance = observation_covariance(q, common, private)
                direct_information = float(
                    np.ones(q)
                    @ np.linalg.solve(covariance, np.ones(q))
                )
                closed_information = fisher_information(q, common, private)
                risk = minimax_risk(1, q, common, private)
                speedup = effective_speedup(q, rho)
                rows.append(
                    {
                        "rho": rho,
                        "overhead": overhead,
                        "q": q,
                        "fisher_closed": closed_information,
                        "fisher_direct": direct_information,
                        "fisher_relative_error": abs(
                            direct_information - closed_information
                        )
                        / closed_information,
                        "minimax_risk": risk,
                        "risk_ratio_speedup": risk_one / risk,
                        "effective_speedup": speedup,
                        "speedup_relative_error": abs(
                            risk_one / risk - speedup
                        )
                        / speedup,
                        "information_per_cost": information_per_cost(
                            q, overhead, common, private
                        ),
                        "selected_q": selected["q"],
                        "selected_efficiency": selected[
                            "information_per_cost"
                        ],
                        "adaptive_budget_lower_bound": lower,
                        "budget_identity_relative_error": abs(
                            lower
                            - 1.0
                            / (
                                BUDGET
                                * selected["information_per_cost"]
                            )
                        )
                        / lower,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_gates(grid: pd.DataFrame) -> Dict[str, object]:
    selected = grid[
        ["rho", "overhead", "selected_q"]
    ].drop_duplicates()
    numerical = {
        "fisher_matrix_identity": bool(
            grid["fisher_relative_error"].max() <= TOLERANCE
        ),
        "risk_speedup_identity": bool(
            grid["speedup_relative_error"].max() <= TOLERANCE
        ),
        "adaptive_budget_identity": bool(
            grid["budget_identity_relative_error"].max() <= TOLERANCE
        ),
    }
    independent = grid[grid["rho"] == 0.0]
    positive = grid[grid["rho"] > 0.0]
    monotone_rho = all(
        np.all(np.diff(group.sort_values("rho")["selected_q"]) <= 0)
        for _, group in selected.groupby("overhead")
    )
    monotone_overhead = all(
        np.all(
            np.diff(group.sort_values("overhead")["selected_q"]) >= 0
        )
        for rho, group in selected.groupby("rho")
        if rho > 0.0
    )
    endpoint_rows = selected[
        selected["rho"].isin((0.0, 0.99))
    ]
    endpoints = all(
        (
            group.loc[group["rho"] == 0.0, "selected_q"].iloc[0] == 32
            and group.loc[group["rho"] == 0.99, "selected_q"].iloc[0]
            == 1
        )
        for _, group in endpoint_rows.groupby("overhead")
    )
    high = grid[(grid["rho"] >= 0.5) & (grid["q"] == 32)]
    scientific = {
        "independent_linear_speedup": bool(
            np.allclose(
                independent["effective_speedup"],
                independent["q"],
                rtol=0.0,
                atol=TOLERANCE,
            )
        ),
        "correlation_speedup_ceiling": bool(
            np.all(
                positive["effective_speedup"]
                <= np.minimum(positive["q"], 1.0 / positive["rho"])
                + TOLERANCE
            )
        ),
        "optimal_q_nonincreasing_in_rho": bool(monotone_rho),
        "optimal_q_nondecreasing_in_overhead": bool(monotone_overhead),
        "endpoint_phase_transition": bool(endpoints),
        "high_correlation_twofold_ceiling": bool(
            np.all(high["effective_speedup"] <= 2.0 + TOLERANCE)
        ),
    }
    return {
        "numerical": numerical,
        "scientific": scientific,
        "numerical_passes": int(sum(numerical.values())),
        "scientific_passes": int(sum(scientific.values())),
        "overall_pass": bool(
            all(numerical.values()) and sum(scientific.values()) >= 5
        ),
        "metrics": {
            "maximum_fisher_relative_error": float(
                grid["fisher_relative_error"].max()
            ),
            "maximum_speedup_relative_error": float(
                grid["speedup_relative_error"].max()
            ),
            "maximum_budget_identity_relative_error": float(
                grid["budget_identity_relative_error"].max()
            ),
            "q32_speedup_at_rho_0_5": float(
                grid[
                    (grid["rho"] == 0.5) & (grid["q"] == 32)
                ]["effective_speedup"].iloc[0]
            ),
            "q32_speedup_at_rho_0_9": float(
                grid[
                    (grid["rho"] == 0.9) & (grid["q"] == 32)
                ]["effective_speedup"].iloc[0]
            ),
        },
    }


def make_figures(grid: pd.DataFrame, output_dir: Path) -> None:
    base = grid[grid["overhead"] == OVERHEADS[0]]
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    for rho in CORRELATIONS:
        group = base[base["rho"] == rho].sort_values("q")
        axis.plot(
            group["q"],
            group["effective_speedup"],
            marker="o",
            label=f"$\\rho={rho:g}$",
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("participating agents $q$")
    axis.set_ylabel("exact speedup")
    axis.grid(alpha=0.3)
    axis.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "speedup_ceiling.png", dpi=180)
    plt.close(fig)

    selected = grid[
        ["rho", "overhead", "selected_q"]
    ].drop_duplicates()
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    for overhead in OVERHEADS:
        group = selected[selected["overhead"] == overhead].sort_values("rho")
        axis.step(
            group["rho"],
            group["selected_q"],
            where="post",
            marker="o",
            label=f"$h={overhead:g}$",
        )
    axis.set_xlabel("cross-agent correlation $\\rho$")
    axis.set_ylabel("resource-optimal $q^\\star$")
    axis.set_yscale("log", base=2)
    axis.set_yticks(AGENT_COUNTS)
    axis.set_yticklabels([str(q) for q in AGENT_COUNTS])
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optimal_participation_phase.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = build_grid()
    gates = evaluate_gates(grid)
    selected = grid[
        [
            "rho",
            "overhead",
            "selected_q",
            "selected_efficiency",
            "adaptive_budget_lower_bound",
        ]
    ].drop_duplicates()
    grid.to_csv(args.output_dir / "phase_grid.csv", index=False)
    selected.to_csv(
        args.output_dir / "selected_participation.csv", index=False
    )
    make_figures(grid, args.output_dir)
    summary = {
        "experiment": "EXP-011A",
        "design": {
            "agent_counts": list(AGENT_COUNTS),
            "correlations": list(CORRELATIONS),
            "overheads": list(OVERHEADS),
            "budget": BUDGET,
            "random_seeds": None,
        },
        "gates": gates,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    if not gates["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
