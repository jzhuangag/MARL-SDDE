"""EXP-016A Amendment 1 feasibility audit.

This module is deliberately static: it reads the frozen preregistration
manifest, computes analytic feasibility quantities, and never draws a
scientific trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence

from run_exp016a import (
    DELTA,
    PILOT_SEEDS,
    POLICIES,
    Action,
    Probe,
    asymptotic_action,
    budget_ray,
    budgets,
    canonical_json,
    config_hash,
    exact_markov_terminal_mse,
    information_only_score,
    load_frozen_manifest,
    probe_catalogue,
    workload_estimate,
)


ORIGINAL_CONFIGURATION_SHA256 = (
    "bb3ab51bc64d4ee334e7c5da6b6e7a4e7ffd303692abb6a5e48d06e48bb9baf5"
)
AMENDMENT_SCHEMA_VERSION = 1
AMENDMENT_ALPHA = 0.05
ABOVE_BS_POINTS_PER_SCENARIO = 2
G6_CELLS_PER_DIRECTION = 108
PRACTICAL_G4_RELATIVE_GAIN = 0.02
PRACTICAL_G8_RELATIVE_GAIN = 0.03
ANOMALY_SINGLE_CELL_ERROR_THRESHOLD = 0.10
ANALYSIS_SEED = 20400101


@dataclass(frozen=True)
class StaticPlan:
    policy: str
    probe_q: int
    probe_b: int
    probe_samples: int
    selected_q: int
    selected_b: int
    commit_updates: int
    message_budget: int
    environment_budget: int
    start_time: int
    risk: float


def one_sided_zero_event_bound(n: int, tail_alpha: float) -> float:
    return float(1.0 - tail_alpha ** (1.0 / n))


def g6_zero_error_feasibility(
    pilot_seed_count: int = len(PILOT_SEEDS),
    cells_per_direction: int = G6_CELLS_PER_DIRECTION,
    alpha: float = AMENDMENT_ALPHA,
    delta: float = DELTA,
) -> dict[str, object]:
    a_min = alpha / cells_per_direction
    n_zero = math.ceil(math.log(a_min) / math.log(1.0 - delta))
    n_uncorrected = math.ceil(math.log(alpha) / math.log(1.0 - delta))
    return {
        "status": "RED_FLAG",
        "reason": "per-cell directional rare-event CI cannot pass with 64 seeds, even with zero observed errors",
        "positive_base_scenarios": 54,
        "above_bs_budget_points_per_scenario": ABOVE_BS_POINTS_PER_SCENARIO,
        "cells_per_direction": cells_per_direction,
        "pilot_seeds_per_cell": pilot_seed_count,
        "familywise_alpha": alpha,
        "directional_delta": delta,
        "a_min": a_min,
        "n_zero_events_with_holm_floor": n_zero,
        "n_zero_events_without_multiplicity": n_uncorrected,
        "zero_event_bound_at_64_with_holm_floor": one_sided_zero_event_bound(
            pilot_seed_count, a_min
        ),
        "zero_event_bound_at_64_without_multiplicity": one_sided_zero_event_bound(
            pilot_seed_count, alpha
        ),
        "original_per_cell_gate_feasible": pilot_seed_count >= n_zero,
        "amendment_rule": "replace per-cell mandatory rare-event CI with deterministic theorem/runtime audit plus aggregate directional calibration",
    }


def _available_updates(
    action: Action,
    message_budget: int,
    environment_budget: int,
    overhead: int,
    delay: int,
) -> int:
    scheduled = min(
        message_budget // (overhead + action.q),
        environment_budget // action.b,
    )
    return max(0, int(scheduled) - delay)


def _planned_risk(
    theta: float,
    mixing: float,
    action: Action,
    message_budget: int,
    environment_budget: int,
    overhead: int,
    delay: int,
) -> float:
    return exact_markov_terminal_mse(
        theta, mixing, action, message_budget, environment_budget, overhead, delay
    )


def _plan_for_policy(
    scenario: Mapping[str, object],
    point: Mapping[str, object],
    regime: str,
    policy: str,
) -> StaticPlan:
    theta = float(scenario["theta_low"] if regime == "low" else scenario["theta_high"])
    mixing = float(scenario["lambda"])
    overhead = int(scenario["overhead"])
    delay = int(scenario["delay"])
    maximum_agents = int(scenario["maximum_agents"])
    message_budget = int(point["message_budget"])
    environment_budget = int(point["environment_budget"])
    baseline = Action(**{k: int(scenario["baseline_action"][k]) for k in ("q", "b")})
    high_action = Action(**{k: int(scenario["high_oracle_action"][k]) for k in ("q", "b")})
    probe = Probe(**scenario["sufficient_probe"])
    selected = baseline
    probe_samples = 0
    probe_q = 0
    probe_b = 0
    remaining_message = message_budget
    remaining_environment = environment_budget
    plan_delay = delay

    if policy == "oracle":
        selected, _ = asymptotic_action(theta, mixing, overhead, budget_ray(
            scenario["budget_ray"]["name"], overhead, maximum_agents
        ), maximum_agents)
    elif policy == "always_all":
        selected = baseline
    elif policy == "fixed_small_q":
        selected = Action(2, 1)
    elif policy == "exp015a_paid_etc_frozen":
        selected = baseline if point["region"] != "above_bs" else high_action
    elif policy == "learning_aware":
        if point["region"] == "above_bs":
            selected = baseline if regime == "low" else high_action
            probe_q, probe_b, probe_samples = probe.q, probe.b, probe.n_sufficient
    elif policy == "information_only":
        ray = budget_ray(scenario["budget_ray"]["name"], overhead, maximum_agents)
        probes = probe_catalogue(
            float(scenario["theta_high"]), mixing, overhead, maximum_agents
        )
        best = min(
            probes,
            key=lambda candidate: (
                -information_only_score(
                    float(scenario["theta_low"]),
                    float(scenario["theta_high"]),
                    mixing,
                    candidate.q,
                    candidate.b,
                    overhead,
                    ray,
                ),
                candidate.n_sufficient,
                candidate.q,
                candidate.b,
            ),
        )
        if point["region"] == "above_bs":
            selected = baseline if regime == "low" else high_action
            probe_q, probe_b, probe_samples = best.q, best.b, best.n_sufficient
    elif policy == "no_delay_ablation":
        plan_delay = 0
        if point["region"] == "above_bs":
            selected = baseline if regime == "low" else high_action
            probe_q, probe_b, probe_samples = probe.q, probe.b, probe.n_sufficient
    elif policy == "ignore_message_budget":
        if point["region"] == "above_bs":
            selected = baseline if regime == "low" else high_action
            probe_q, probe_b, probe_samples = probe.q, probe.b, probe.n_sufficient
    elif policy == "ignore_environment_budget":
        if point["region"] == "above_bs":
            selected = high_action if regime == "high" else baseline
            probe_q, probe_b, probe_samples = probe.q, probe.b, probe.n_sufficient
    elif policy == "no_mixing_correction":
        no_mix_action, _ = asymptotic_action(
            theta, 0.0, overhead, budget_ray(
                scenario["budget_ray"]["name"], overhead, maximum_agents
            ), maximum_agents
        )
        selected = no_mix_action
    else:
        raise ValueError(f"unknown policy: {policy}")

    if probe_samples:
        remaining_message -= probe_samples * (overhead + probe_q)
        remaining_environment -= probe_samples * probe_b
    risk = _planned_risk(
        theta, mixing, selected, remaining_message, remaining_environment, overhead, delay
    )
    updates = _available_updates(
        selected, remaining_message, remaining_environment, overhead, delay
    )
    start_time = probe_samples * probe_b + (delay if probe_samples else 0)
    if plan_delay != delay:
        start_time = probe_samples * probe_b
    return StaticPlan(
        policy=policy,
        probe_q=probe_q,
        probe_b=probe_b,
        probe_samples=probe_samples,
        selected_q=selected.q,
        selected_b=selected.b,
        commit_updates=updates,
        message_budget=message_budget,
        environment_budget=environment_budget,
        start_time=start_time,
        risk=risk,
    )


def same_path(first: StaticPlan, second: StaticPlan) -> bool:
    return (
        first.probe_q,
        first.probe_b,
        first.probe_samples,
        first.selected_q,
        first.selected_b,
        first.commit_updates,
        first.start_time,
    ) == (
        second.probe_q,
        second.probe_b,
        second.probe_samples,
        second.selected_q,
        second.selected_b,
        second.commit_updates,
        second.start_time,
    )


@lru_cache(maxsize=None)
def _common_time_sum(
    first_updates: int,
    first_b: int,
    first_start: int,
    second_updates: int,
    second_b: int,
    second_start: int,
    mixing: float,
) -> float:
    total = 0.0
    for i in range(first_updates):
        time_i = first_start + i * first_b
        for j in range(second_updates):
            time_j = second_start + j * second_b
            total += mixing ** abs(time_i - time_j)
    return float(total)


def _same_time_count(first: StaticPlan, second: StaticPlan) -> int:
    first_times = {
        first.start_time + update * first.selected_b
        for update in range(first.commit_updates)
    }
    return sum(
        1
        for update in range(second.commit_updates)
        if second.start_time + update * second.selected_b in first_times
    )


def covariance_between_plans(
    theta: float, mixing: float, first: StaticPlan, second: StaticPlan
) -> float:
    if first.commit_updates <= 0 or second.commit_updates <= 0:
        return float("nan")
    common = theta * _common_time_sum(
        first.commit_updates,
        first.selected_b,
        first.start_time,
        second.commit_updates,
        second.selected_b,
        second.start_time,
        mixing,
    ) / (first.commit_updates * second.commit_updates)
    private = (
        _same_time_count(first, second)
        * min(first.selected_q, second.selected_q)
        / (
            first.commit_updates
            * second.commit_updates
            * first.selected_q
            * second.selected_q
        )
    )
    return float(common + private)


def paired_difference_moments(
    theta: float, mixing: float, first: StaticPlan, second: StaticPlan
) -> dict[str, float]:
    covariance = covariance_between_plans(theta, mixing, first, second)
    variance = 2.0 * (
        first.risk * first.risk + second.risk * second.risk - 2.0 * covariance * covariance
    )
    return {
        "expected_difference": float(first.risk - second.risk),
        "paired_variance": float(max(0.0, variance)),
        "covariance": float(covariance),
    }


def prospective_se(paired_variance: float, seed_count: int = len(PILOT_SEEDS)) -> float:
    return math.sqrt(max(0.0, paired_variance) / seed_count)


def conservative_critical_value(family_size: int, alpha: float = AMENDMENT_ALPHA) -> float:
    adjusted = 1.0 - alpha / max(1, family_size)
    return float(NormalDist().inv_cdf(adjusted))


def _above_bs_points(scenario: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [point for point in scenario["budget_points"] if point["region"] == "above_bs"]


def feasibility_cells(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    cells = []
    for scenario in manifest["positive_scenarios"]:
        for point in _above_bs_points(scenario):
            for regime in ("low", "high"):
                learning = _plan_for_policy(scenario, point, regime, "learning_aware")
                always_all = _plan_for_policy(scenario, point, regime, "always_all")
                information = _plan_for_policy(scenario, point, regime, "information_only")
                no_delay = _plan_for_policy(scenario, point, regime, "no_delay_ablation")
                ignore_message = _plan_for_policy(scenario, point, regime, "ignore_message_budget")
                ignore_environment = _plan_for_policy(scenario, point, regime, "ignore_environment_budget")
                theta = float(scenario["theta_low"] if regime == "low" else scenario["theta_high"])
                mixing = float(scenario["lambda"])
                oracle = _plan_for_policy(scenario, point, regime, "oracle")
                high_gain = paired_difference_moments(theta, mixing, always_all, learning)
                novelty = paired_difference_moments(theta, mixing, information, learning)
                delay_effect = paired_difference_moments(theta, mixing, no_delay, learning)
                message_effect = paired_difference_moments(theta, mixing, ignore_message, learning)
                environment_effect = paired_difference_moments(theta, mixing, ignore_environment, learning)
                always_regret = max(always_all.risk - oracle.risk, 1e-15)
                info_regret = max(information.risk - oracle.risk, 1e-15)
                cells.append({
                    "scenario_id": scenario["scenario_id"],
                    "budget_point": point["name"],
                    "regime": regime,
                    "epsilon_safe": scenario["epsilon_safe"],
                    "delay": scenario["delay"],
                    "budget_ray": scenario["budget_ray"]["name"],
                    "learning_risk": learning.risk,
                    "always_all_risk": always_all.risk,
                    "information_only_risk": information.risk,
                    "oracle_risk": oracle.risk,
                    "g4_expected_gain": high_gain["expected_difference"],
                    "g4_expected_relative_gain": high_gain["expected_difference"] / always_regret,
                    "g4_paired_variance": high_gain["paired_variance"],
                    "g4_prospective_se_64": prospective_se(high_gain["paired_variance"]),
                    "g5_expected_safety_deficit": max(0.0, learning.risk - always_all.risk) / always_all.risk,
                    "g8_expected_gain": novelty["expected_difference"],
                    "g8_expected_relative_gain": novelty["expected_difference"] / info_regret,
                    "g8_paired_variance": novelty["paired_variance"],
                    "g8_prospective_se_64": prospective_se(novelty["paired_variance"]),
                    "learning_information_identical_path": same_path(learning, information),
                    "learning_value_active": (
                        not same_path(learning, information)
                        and novelty["expected_difference"] / info_regret >= PRACTICAL_G8_RELATIVE_GAIN
                    ),
                    "delay_identical_path": same_path(learning, no_delay),
                    "delay_active": int(scenario["delay"]) == 12 and not same_path(learning, no_delay),
                    "message_identical_path": same_path(learning, ignore_message),
                    "message_active": (
                        scenario["budget_ray"]["name"] == "message_limited"
                        and not same_path(learning, ignore_message)
                    ),
                    "environment_identical_path": same_path(learning, ignore_environment),
                    "environment_active": (
                        scenario["budget_ray"]["name"] == "environment_limited"
                        and not same_path(learning, ignore_environment)
                    ),
                    "delay_effect": delay_effect["expected_difference"],
                    "message_effect": message_effect["expected_difference"],
                    "environment_effect": environment_effect["expected_difference"],
                })
    return cells


def revised_gate_table(original_hash: str, amendment_hash: str | None = None) -> dict[str, object]:
    return {
        "schema_version": 2,
        "experiment": "EXP-016A",
        "supersedes": "docs/exp016a_gate_table.json",
        "original_configuration_sha256": original_hash,
        "amendment_configuration_sha256": amendment_hash,
        "all_mandatory": True,
        "gray_zone_core_gate_eligible": False,
        "familywise_alpha": AMENDMENT_ALPHA,
        "scientific_outcomes_present": False,
        "gates": [
            {
                "id": "G1",
                "name": "validity_and_leakage",
                "scope": "all generated rows plus static source/runtime audit",
                "pass_rule": "all metrics finite, zero charged budget violations, zero hidden-state leakage, and no analytic risk copied into observed MSE",
                "mandatory": True,
            },
            {
                "id": "G2",
                "name": "below_B_N_fallback",
                "scope": "all theorem-scope below_bn cells",
                "pass_rule": "learning_aware fallback fraction equals 1 exactly by deterministic runtime path audit",
                "mandatory": True,
            },
            {
                "id": "G3",
                "name": "above_B_S_qualification",
                "scope": "all theorem-scope above_bs cells",
                "pass_rule": "100% of paths satisfy frozen probe, delay, dual-budget, epsilon_safe, and commit-reserve qualification predicates",
                "mandatory": True,
            },
            {
                "id": "G4",
                "name": "high_regime_gain",
                "scope": "all theorem-scope above_bs high-regime cells, with practical-effect subset reported separately",
                "pass_rule": "all cells must have positive analytic expected paired gain direction; only the outcome-free practical-effect subset with analytic relative gain >=0.02 is required to pass the 2% practical-effect claim and simultaneous lower-bound calibration",
                "mandatory": True,
            },
            {
                "id": "G5",
                "name": "low_regime_safety",
                "scope": "all theorem-scope above_bs low-regime cells",
                "pass_rule": "mandatory theorem/runtime compliance audit for epsilon_safe; empirical CI calibration is reported as calibration and may not be an impossible per-cell pass/fail gate",
                "mandatory": True,
            },
            {
                "id": "G6a",
                "name": "directional_identification_theorem_runtime_audit",
                "scope": "all theorem-scope above_bs cells, low-to-high and high-to-low separately",
                "pass_rule": "frozen bidirectional test thresholds are used, both directional analytic bounds are <=0.025, runtime likelihood/stopping/q/b/delay/budget path equals manifest, zero threshold mismatch, and zero hidden-state leakage",
                "mandatory": True,
            },
            {
                "id": "G6b",
                "name": "directional_identification_empirical_calibration",
                "scope": "all preregistered above_bs seed blocks aggregated by direction, with per-cell intervals reported",
                "pass_rule": "one-sided exact binomial/Clopper-Pearson simultaneous aggregate directional upper bound <=0.025; any single-cell observed error rate >0.10 triggers failure or manual implementation audit",
                "mandatory": True,
            },
            {
                "id": "G7",
                "name": "break_even_bracket",
                "scope": "positive theorem scenarios only",
                "pass_rule": "at least 0.75 of base scenarios place first registered positive paired mean gain in closed [B_N,B_S]",
                "mandatory": True,
            },
            {
                "id": "G8",
                "name": "novelty_learning_value",
                "scope": "learning-value-active delayed dual-budget subset defined before outcomes",
                "pass_rule": "subset must be nonempty; within it, learning_aware must beat information_only by >=0.03 relative with positive simultaneous lower-bound calibration and no charged-violation advantage; identical-path cells are reported but excluded from the novelty effect requirement",
                "mandatory": True,
            },
            {
                "id": "G9",
                "name": "delay_mechanism",
                "scope": "delay-active D=12 cells only",
                "pass_rule": "no-delay ablation must have worse intended plan or worse oracle regret only where planning paths differ; identical-path cells receive consistency checks only",
                "mandatory": True,
            },
            {
                "id": "G10",
                "name": "dual_budget_mechanism",
                "scope": "message-active and environment-active cells only",
                "pass_rule": "single-budget ablations must ignore the corresponding budget and produce a different intended plan before worse-outcome claims are tested; nonbinding/identical cells are consistency-only",
                "mandatory": True,
            },
            {
                "id": "G11",
                "name": "exp015a_freeze",
                "scope": "repository and report audit",
                "pass_rule": "EXP-015A remains 7/8 failure; threshold 0.80 and observed 0.777778 are unchanged",
                "mandatory": True,
            },
            {
                "id": "G12",
                "name": "reproducibility",
                "scope": "clean rerun with identical pilot seeds and committed configuration",
                "pass_rule": "core raw CSV, aggregate CSV, gate JSON, metadata JSON, and configuration hash reproduce byte-for-byte",
                "mandatory": True,
            },
        ],
        "progression": "All revised mandatory gates must pass. There is no aggregate k-of-n override.",
    }


def build_feasibility_audit(manifest: Mapping[str, object] | None = None) -> dict[str, object]:
    if manifest is None:
        manifest = load_frozen_manifest()
    if manifest["configuration_sha256"] != ORIGINAL_CONFIGURATION_SHA256:
        raise ValueError("unexpected original manifest hash")
    cells = feasibility_cells(manifest)
    high_cells = [cell for cell in cells if cell["regime"] == "high"]
    low_cells = [cell for cell in cells if cell["regime"] == "low"]
    g4_subset = [
        cell for cell in high_cells
        if cell["g4_expected_relative_gain"] >= PRACTICAL_G4_RELATIVE_GAIN
        and cell["g4_expected_gain"] > 0.0
    ]
    g8_subset = [cell for cell in high_cells if cell["learning_value_active"]]
    delay_active = [cell for cell in high_cells if cell["delay_active"]]
    message_active = [cell for cell in high_cells if cell["message_active"]]
    environment_active = [cell for cell in high_cells if cell["environment_active"]]
    g4_critical = conservative_critical_value(len(high_cells))
    g8_critical = conservative_critical_value(max(1, len(g8_subset)))
    g5_critical = conservative_critical_value(len(low_cells))
    g4_power_ready = sum(
        1 for cell in g4_subset
        if cell["g4_expected_gain"] > g4_critical * cell["g4_prospective_se_64"]
    )
    g8_power_ready = sum(
        1 for cell in g8_subset
        if cell["g8_expected_gain"] > g8_critical * cell["g8_prospective_se_64"]
    )
    g5_power_ready = sum(
        1 for cell in low_cells
        if float(cell["epsilon_safe"]) - cell["g5_expected_safety_deficit"] > 0.0
    )
    estimate = workload_estimate(manifest)
    gate_feasibility = {
        "G1": "feasible_static_and_runtime_audit",
        "G2": "feasible_deterministic_fallback",
        "G3": "feasible_deterministic_theorem_qualification",
        "G4": "feasible_after_practical_effect_subset_revision",
        "G5": "feasible_as_theorem_compliance_plus_empirical_calibration",
        "G6a": "feasible_deterministic_theorem_runtime_audit",
        "G6b": "feasible_aggregate_directional_calibration_not_per_cell_CI",
        "G7": "feasible_empirical_bracket_gate",
        "G8": "feasible_if_learning_value_active_subset_nonempty",
        "G9": "feasible_on_delay_active_subset",
        "G10": "feasible_on_binding_budget_active_subsets",
        "G11": "feasible_byte_freeze_audit",
        "G12": "feasible_byte_reproducibility_audit",
    }
    stop_decision = "A"
    if not g8_subset:
        stop_decision = "B"
        gate_feasibility["G8"] = "infeasible_empty_learning_value_active_subset"
    return {
        "schema_version": AMENDMENT_SCHEMA_VERSION,
        "experiment": "EXP-016A",
        "audit": "preregistration_feasibility_amendment_1",
        "scientific_outcomes_present": False,
        "original_configuration_sha256": ORIGINAL_CONFIGURATION_SHA256,
        "original_manifest_sha256": manifest["configuration_sha256"],
        "g6_zero_error_feasibility": g6_zero_error_feasibility(),
        "prospective_family_sizes": {
            "above_bs_high_cells": len(high_cells),
            "above_bs_low_cells": len(low_cells),
            "g4_practical_effect_subset": len(g4_subset),
            "g8_learning_value_active_subset": len(g8_subset),
            "g9_delay_active_high_cells": len(delay_active),
            "g10_message_active_high_cells": len(message_active),
            "g10_environment_active_high_cells": len(environment_active),
        },
        "prospective_power_proxy": {
            "method": "outcome-free Gaussian paired moments plus Bonferroni normal critical values",
            "g4_critical_value": g4_critical,
            "g4_subset_power_ready_cells_at_64": g4_power_ready,
            "g5_cells_with_positive_analytic_margin_at_64": g5_power_ready,
            "g8_critical_value": g8_critical,
            "g8_subset_power_ready_cells_at_64": g8_power_ready,
        },
        "minimum_margins": {
            "g4_min_expected_relative_gain_all_high_cells": min(
                cell["g4_expected_relative_gain"] for cell in high_cells
            ),
            "g4_min_expected_relative_gain_practical_subset": min(
                (cell["g4_expected_relative_gain"] for cell in g4_subset), default=0.0
            ),
            "g5_min_expected_safety_margin_to_epsilon": min(
                float(cell["epsilon_safe"]) - cell["g5_expected_safety_deficit"]
                for cell in low_cells
            ),
            "g8_min_expected_relative_gain_active_subset": min(
                (cell["g8_expected_relative_gain"] for cell in g8_subset), default=0.0
            ),
        },
        "identical_path_counts": {
            "learning_vs_information_only_high": sum(
                1 for cell in high_cells if cell["learning_information_identical_path"]
            ),
            "learning_vs_no_delay_high": sum(
                1 for cell in high_cells if cell["delay_identical_path"]
            ),
            "learning_vs_ignore_message_high": sum(
                1 for cell in high_cells if cell["message_identical_path"]
            ),
            "learning_vs_ignore_environment_high": sum(
                1 for cell in high_cells if cell["environment_identical_path"]
            ),
        },
        "gate_feasibility": gate_feasibility,
        "final_pilot_seed_count": len(PILOT_SEEDS),
        "seed_decision": "keep_original_64_pilot_seeds",
        "workload": estimate,
        "pilot_authorization": stop_decision == "A",
        "stop_gate_decision": stop_decision,
        "notes": [
            "352000 rows equals 550 expanded cells times 10 policies times 64 seeds",
            "each future row is one real trajectory outcome; CRN pairing does not increase effective sample size",
            "bootstrap resampling unit remains the complete seed block",
            "cross-budget shared random prefixes must not be treated as independent repetitions",
            "CVaR90 at 64 seeds is secondary only",
            "gray-zone and negative-control cells cannot rescue core theorem gates",
            "analytic risks remain forbidden as observed MSE",
        ],
    }


def amendment_configuration_hash(
    audit: Mapping[str, object] | None = None,
    gate_table: Mapping[str, object] | None = None,
) -> str:
    if audit is None:
        audit = build_feasibility_audit()
    if gate_table is None:
        gate_table = revised_gate_table(ORIGINAL_CONFIGURATION_SHA256)
    payload = {
        "original_configuration_sha256": ORIGINAL_CONFIGURATION_SHA256,
        "amendment_files": [
            "docs/exp016a_preregistration_amendment_1.md",
            "docs/exp016a_gate_table_v2.json",
            "docs/exp016a_analysis_plan_v2.md",
            "docs/exp016a_feasibility_audit.md",
            "docs/exp016a_feasibility_audit.json",
        ],
        "audit": audit,
        "gate_table": gate_table,
    }
    return config_hash(payload)


def emit_payload() -> dict[str, object]:
    audit = build_feasibility_audit()
    gate_table = revised_gate_table(
        ORIGINAL_CONFIGURATION_SHA256,
        amendment_configuration_hash(audit, revised_gate_table(ORIGINAL_CONFIGURATION_SHA256)),
    )
    audit = dict(audit)
    audit["amendment_configuration_sha256"] = gate_table["amendment_configuration_sha256"]
    return {
        "audit": audit,
        "gate_table_v2": gate_table,
    }


def feasibility_markdown(audit: Mapping[str, object]) -> str:
    g6 = audit["g6_zero_error_feasibility"]
    sizes = audit["prospective_family_sizes"]
    workload = audit["workload"]
    margins = audit["minimum_margins"]
    return f"""# EXP-016A feasibility audit for Amendment 1

This is a preregistration feasibility audit. It contains no trajectory,
pilot, formal, HPC4, GPU, result-row, or scientific outcome generation.

## Frozen inputs

- Original preregistration commit:
  `592986466ff55281914dd76a4faad7338ea91914`
- Original configuration SHA-256:
  `{audit["original_configuration_sha256"]}`
- Positive base scenarios: 54
- Above-`B_S` budget points per scenario: 2
- Cells per identification direction: {g6["cells_per_direction"]}
- Pilot seeds per cell: {g6["pilot_seeds_per_cell"]}
- Familywise alpha: {g6["familywise_alpha"]}
- Directional delta: {g6["directional_delta"]}

The original preregistration files remain immutable provenance. Amendment 1
supersedes only the analysis rules and feasibility decision.

## G6 design-level red flag

The original G6 required a per-cell directional Clopper-Pearson/Holm upper
bound <= `0.025`. With 108 cells per direction, the smallest Holm tail level is
`alpha/108 = {g6["a_min"]:.17g}`. Even if a cell observes zero errors, the
sample count needed is

```text
ceil(log(alpha/108) / log(1 - 0.025)) = {g6["n_zero_events_with_holm_floor"]}
```

Without multiplicity correction the corresponding zero-error requirement is
{g6["n_zero_events_without_multiplicity"]} seeds. Therefore the frozen 64
pilot seeds cannot pass the original per-cell G6 gate, even under perfect
zero-error observations. This is a design-level `RED_FLAG`, not something to
confirm by running a doomed pilot.

## Revised feasibility findings

- G4 practical-effect subset: {sizes["g4_practical_effect_subset"]} of
  {sizes["above_bs_high_cells"]} high above-`B_S` cells.
- G8 learning-value-active subset: {sizes["g8_learning_value_active_subset"]}.
- G9 delay-active high cells: {sizes["g9_delay_active_high_cells"]}.
- G10 message-active high cells: {sizes["g10_message_active_high_cells"]}.
- G10 environment-active high cells: {sizes["g10_environment_active_high_cells"]}.
- Minimum G4 expected relative gain over all high cells:
  `{margins["g4_min_expected_relative_gain_all_high_cells"]:.17g}`.
- Minimum G5 analytic margin to epsilon:
  `{margins["g5_min_expected_safety_margin_to_epsilon"]:.17g}`.

Because the learning-value-active subset is empty, revised G8 is infeasible
under the current frozen policy definitions. Amendment 1 selects stop gate
decision **B**: stop the ICML adaptation-cost main line and do not run the
EXP-016A pilot.

## Workload decision

The seed count remains 64. No fresh pilot seeds are added because adding
seeds cannot repair an empty novelty subset or the original G6 role mismatch.

- Expanded cells: {workload["expanded_cells"]}
- Policies: {workload["policy_count"]}
- Estimated rows/trajectories: {workload["estimated_rows"]}
- Estimated single-process CPU hours: `{workload["estimated_cpu_wall_hours_single_process"]:.17g}`
- Estimated peak memory GB: `{workload["estimated_peak_memory_gb"]:.17g}`
- Estimated disk GB: `{workload["estimated_disk_gb"]:.17g}`

Local CPU would have remained within the preregistered resource envelope, but
the pilot is not authorized after this audit.
"""


def preregistration_amendment_markdown(audit: Mapping[str, object]) -> str:
    return f"""# EXP-016A preregistration Amendment 1

Amendment 1 supersedes the EXP-016A analysis rules while preserving the
original preregistration files as immutable provenance.

## Binding hashes

- Original configuration SHA-256:
  `{audit["original_configuration_sha256"]}`
- Amendment configuration SHA-256:
  `{audit["amendment_configuration_sha256"]}`

The amendment hash binds the original configuration hash, the Amendment 1 file
set, the revised gate table payload, and the static feasibility audit payload.

## Reason for amendment

The original G6 gate was a mandatory per-cell rare-event confidence gate. With
108 cells per direction and 64 pilot seeds per cell, even zero observed errors
would give a Holm-floor upper bound above `0.025`. The frozen calculation is
recorded in `exp016a_feasibility_audit.json` and marks this as a design-level
`RED_FLAG`.

The amendment separates deterministic theorem/runtime compliance from
empirical calibration:

- G6a is a mandatory deterministic audit of thresholds, likelihoods, stopping
  boundaries, `q/b`, delay accounting, budget accounting, and hidden-state
  leakage.
- G6b is mandatory aggregate directional empirical calibration, with per-cell
  intervals reported descriptively and a single-cell anomaly threshold of
  observed error rate `>0.10`.

## Outcome-free active subsets

Amendment 1 also prevents identical policy paths from being forced to produce
nonzero novelty or mechanism effects. G8, G9, and G10 operate only on
outcome-free active subsets where the frozen intended plans differ before any
trajectory is generated.

The current frozen definitions produce a G8 learning-value-active subset of
size `{audit["prospective_family_sizes"]["g8_learning_value_active_subset"]}`.
Because this subset is empty, the EXP-016A pilot is not authorized. This
selects stop gate decision **B** from the preregistered feasibility audit.

## Authorization

No scientific trajectory, pilot, formal run, HPC4 job, GPU job, result row, or
scientific outcome was generated. The next stage should redesign the novelty
comparison or stop the adaptation-cost ICML main line; it should not run the
current EXP-016A pilot.
"""


def analysis_plan_v2_markdown(audit: Mapping[str, object]) -> str:
    return f"""# EXP-016A analysis plan v2

This document supersedes `exp016a_analysis_plan.md` for any future EXP-016A
work. The original plan remains immutable provenance.

## Static eligibility before outcomes

All core cells keep their original scenario, regime, budget, policy, and seed
assignments. Gray-zone cells remain outside core theorem gates. Negative
controls remain explanatory only and cannot rescue positive gates.

For every mechanism comparison, the runtime must first record whether the two
policies have identical probe, commit, action, delay, and budget paths. A cell
with identical paths is a consistency cell, not an effect cell.

## Revised G6

G6a is a deterministic theorem/runtime audit. Every above-`B_S` cell must use
the frozen bidirectional thresholds, registered likelihood, stopping boundary,
`q/b`, true public delay, and true dual-budget accounting. Both analytic
directional bounds must be <= `0.025`. Threshold mismatches and hidden-state
leakage must be exactly zero.

G6b is empirical calibration. Low-to-high and high-to-low are aggregated by
direction across all preregistered above-`B_S` seed blocks using exact
one-sided binomial/Clopper-Pearson simultaneous bounds. The aggregate
directional upper bound must be <= `0.025`. Per-cell errors and intervals are
reported, but a 64-seed per-cell rare-event CI is not a mandatory pass gate.
Any single-cell observed error rate above `0.10` triggers failure or manual
implementation audit.

## Revised practical-effect gates

G4 keeps the mandatory positive high-regime direction over all above-`B_S`
high cells. The 2% practical-effect claim applies only to the outcome-free
subset whose analytic expected relative gain is at least `0.02`; the current
subset size is `{audit["prospective_family_sizes"]["g4_practical_effect_subset"]}`.

G5 is first a deterministic theorem compliance audit against each scenario's
`epsilon_safe`. Empirical safety intervals are calibration evidence and may
not be formulated as an impossible per-cell CI gate.

G8 applies only to the outcome-free learning-value-active subset, defined by
different intended learning-aware and information-only plans plus analytic
expected relative gain at least `0.03`. The current active subset size is
`{audit["prospective_family_sizes"]["g8_learning_value_active_subset"]}`.
An empty subset is an immediate novelty failure.

G9 applies only to delay-active `D=12` cells where no-delay planning and true
delay planning differ. G10 applies only to message-active or environment-active
cells where the corresponding single-budget ablation changes the intended
plan. Identical-path cells are reported as consistency checks.

## Resampling and reporting

CRN pairing does not increase effective sample size. The bootstrap resampling
unit remains the complete seed block, and cross-budget shared random prefixes
must not be counted as independent repetitions. CVaR90 with 64 seeds is
secondary only. Analytic risks may be used for thresholds, oracle joins, and
feasibility audits, but never as observed MSE.
"""


def freeze_amendment_files() -> dict[str, str]:
    payload = emit_payload()
    audit = payload["audit"]
    gate_table = payload["gate_table_v2"]
    root = Path(__file__).resolve().parents[2]
    targets = {
        "docs/exp016a_feasibility_audit.json": json.dumps(
            audit, indent=2, sort_keys=True
        ) + "\n",
        "docs/exp016a_gate_table_v2.json": json.dumps(
            gate_table, indent=2, sort_keys=True
        ) + "\n",
        "docs/exp016a_feasibility_audit.md": feasibility_markdown(audit),
        "docs/exp016a_preregistration_amendment_1.md": preregistration_amendment_markdown(audit),
        "docs/exp016a_analysis_plan_v2.md": analysis_plan_v2_markdown(audit),
    }
    for relative, text in targets.items():
        (root / relative).write_text(text, encoding="utf-8")
    return {relative: config_hash(text) for relative, text in targets.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("audit", "gate-table-v2", "freeze-amendment"),
        nargs="?",
        default="audit",
    )
    return parser.parse_args()


def main() -> None:
    payload = emit_payload()
    args = parse_args()
    if args.command == "gate-table-v2":
        print(json.dumps(payload["gate_table_v2"], indent=2, sort_keys=True))
    elif args.command == "freeze-amendment":
        print(json.dumps(freeze_amendment_files(), indent=2, sort_keys=True))
    else:
        print(json.dumps(payload["audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
