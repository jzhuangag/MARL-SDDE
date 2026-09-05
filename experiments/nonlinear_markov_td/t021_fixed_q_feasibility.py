"""T-021 CPU-only fixed-q claim and power-feasibility audit.

The audit consumes only committed T-019/T-020 design artifacts.  It does not
step an environment, reconstruct EXP-017A trajectories, tune a gate, allocate
formal seeds, or authorize a GPU run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any


Q_LEVELS = (1, 4, 16, 32)
RHO_LEVELS = (0.0, 0.5, 0.9)
PRIMARY_FAMILIES = (
    "correlation_saturation",
    "dual_budget_phase",
    "delay_degradation",
)
EXPECTED_T019_ROWS = 432
EXPECTED_T019_CELLS = 72
PRACTICAL_RATIO = 1.05
FAMILYWISE_ALPHA = 0.05
TARGET_POWERS = (0.80, 0.90)
SD_GRID = (0.10, 0.15, 0.20)
MAX_PLANNED_REPLICATIONS = 192


def variance_factor(q: int, rho: float) -> float:
    """Equicorrelated variance of an average, relative to one agent."""

    if q < 1:
        raise ValueError("q must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    return rho + (1.0 - rho) / float(q)


def effective_speedup(q: int, rho: float) -> float:
    return 1.0 / variance_factor(q, rho)


def required_replications(
    effect_log_ratio: float,
    paired_sd: float,
    target_power: float,
    alpha: float,
) -> int:
    """Normal-approximation sample size for a one-sided paired contrast."""

    if effect_log_ratio <= 0.0 or paired_sd <= 0.0:
        raise ValueError("effect and standard deviation must be positive")
    if not 0.0 < alpha < 1.0 or not 0.0 < target_power < 1.0:
        raise ValueError("alpha and target_power must lie in (0, 1)")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha)
    z_power = normal.inv_cdf(target_power)
    return int(math.ceil(((z_alpha + z_power) * paired_sd / effect_log_ratio) ** 2))


def achieved_power(
    replications: int,
    effect_log_ratio: float,
    paired_sd: float,
    alpha: float,
) -> float:
    if replications < 1:
        raise ValueError("replications must be positive")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha)
    noncentrality = math.sqrt(replications) * effect_log_ratio / paired_sd
    return normal.cdf(noncentrality - z_alpha)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_phase_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "task",
        "mixing",
        "rho",
        "delay_trace",
        "budget",
        "policy",
        "nominal_q",
        "best_fixed_q",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("T-019 phase table has an incompatible schema")
    return rows


def build_summary(repository_root: Path) -> dict[str, Any]:
    docs = repository_root / "docs"
    phase_rows = _load_phase_rows(docs / "t019_fixed_q_phase_diagram.csv")
    t019 = _load_json(docs / "t019_fixed_q_phase_summary.json")
    t020 = _load_json(docs / "t020_adaptation_value_ceiling.json")

    cells = {
        (
            row["task"],
            row["mixing"],
            row["rho"],
            row["delay_trace"],
            row["budget"],
        )
        for row in phase_rows
    }
    if len(phase_rows) != EXPECTED_T019_ROWS or len(cells) != EXPECTED_T019_CELLS:
        raise ValueError("unexpected T-019 phase-table dimensions")
    if t019["phase_rows"] != EXPECTED_T019_ROWS or t019["phase_cells"] != EXPECTED_T019_CELLS:
        raise ValueError("T-019 JSON disagrees with the phase table")
    if not t020["gate_audit"]["exp017b_permanently_stopped"]:
        raise ValueError("T-020 stop decision must remain immutable")

    variance_grid = {
        str(rho): {
            str(q): {
                "variance_factor": variance_factor(q, rho),
                "effective_speedup": effective_speedup(q, rho),
            }
            for q in Q_LEVELS
        }
        for rho in RHO_LEVELS
    }

    alpha_per_family = FAMILYWISE_ALPHA / len(PRIMARY_FAMILIES)
    practical_log_effect = math.log(PRACTICAL_RATIO)
    power_rows = []
    for paired_sd in SD_GRID:
        row: dict[str, Any] = {"paired_log_ratio_sd": paired_sd}
        for target in TARGET_POWERS:
            row[f"n_for_{int(100 * target)}pct_power"] = required_replications(
                effect_log_ratio=practical_log_effect,
                paired_sd=paired_sd,
                target_power=target,
                alpha=alpha_per_family,
            )
        row["power_at_192"] = achieved_power(
            replications=MAX_PLANNED_REPLICATIONS,
            effect_log_ratio=practical_log_effect,
            paired_sd=paired_sd,
            alpha=alpha_per_family,
        )
        power_rows.append(row)

    phase = t019["phase_direction"]
    descriptive_checks = {
        "multiple_fixed_q_optima_observed": len(t019["best_fixed_q_counts"]) == 4,
        "rho_direction_fraction": phase["rho_expected_nonincreasing_paths"]
        / phase["rho_total_paths"],
        "delay_direction_fraction": phase["delay_expected_nonincreasing_paths"]
        / phase["delay_total_paths"],
        "budget_direction_fraction": phase["environment_q_at_least_message_q_pairs"]
        / phase["budget_total_pairs"],
        "adaptive_controller_value_gate_passed": t020["gate_audit"][
            "all_static_oracle_gates_pass"
        ],
    }

    return {
        "experiment": "T-021",
        "execution_mode": "cpu_static_outcome_free_no_trajectory",
        "source_artifacts": {
            "t019_phase_rows": len(phase_rows),
            "t019_phase_cells": len(cells),
            "t019_pilot_seeds_per_cell_arm": t019["pilot_seed_count_per_cell_arm"],
            "t020_evidence_status": t020["evidence_status"],
            "t020_exp017b_permanently_stopped": True,
        },
        "theoretical_variance_grid": variance_grid,
        "descriptive_design_checks": descriptive_checks,
        "power_design": {
            "primary_families": list(PRIMARY_FAMILIES),
            "familywise_alpha": FAMILYWISE_ALPHA,
            "one_sided_alpha_per_family": alpha_per_family,
            "practical_ratio": PRACTICAL_RATIO,
            "practical_log_effect": practical_log_effect,
            "normal_approximation_rows": power_rows,
            "replication_rule": (
                "A separate implementation pilot may estimate only the paired "
                "log-ratio SD. Freeze formal N as the smallest listed value "
                "achieving 90% power, capped at 192; if the required N exceeds "
                "192, stop rather than weaken the effect or alpha."
            ),
            "maximum_authorizable_replications": MAX_PLANNED_REPLICATIONS,
        },
        "claim_decision": {
            "online_adaptive_controller_main_claim": "exclude",
            "fixed_q_correlation_delay_mainline": "retain",
            "sdde_role": "interpretation_only_until_discretization_error_is_closed",
            "nonlinear_role": "mechanism_validation_not_general_convergence_theorem",
            "future_gpu_preregistration_authorized": False,
            "next_stage": "outcome_free_implementation_and_preregistration_under_new_id",
        },
        "scientific_trajectories_generated": 0,
        "gpu_jobs_submitted": 0,
    }


def validate_summary(summary: dict[str, Any]) -> None:
    source = summary["source_artifacts"]
    if source["t019_phase_rows"] != EXPECTED_T019_ROWS:
        raise ValueError("phase-row validation failed")
    if source["t019_phase_cells"] != EXPECTED_T019_CELLS:
        raise ValueError("phase-cell validation failed")
    if not source["t020_exp017b_permanently_stopped"]:
        raise ValueError("EXP-017B stop decision was lost")
    if summary["claim_decision"]["online_adaptive_controller_main_claim"] != "exclude":
        raise ValueError("failed controller cannot remain a main claim")
    if summary["scientific_trajectories_generated"] != 0:
        raise ValueError("T-021 must be outcome-free")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = build_summary(args.repository_root.resolve())
    validate_summary(summary)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

