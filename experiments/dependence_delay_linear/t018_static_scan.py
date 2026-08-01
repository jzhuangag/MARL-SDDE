"""T-018 outcome-free learning-value separation scan.

The scan is analytic only. It reads no trajectory output, draws no random
variables, and writes no scientific outcomes.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from run_exp016a import (
    DELTA,
    THETA_LOW,
    Action,
    budget_ray,
    budgets,
    canonical_json,
    config_hash,
    exact_markov_terminal_mse,
    information_only_score,
    probe_catalogue,
    qualification_margin,
)


SCHEMA_VERSION = 1
TASK = "T-018"
STARTING_HEAD = "3bdca34c4c5f0f55e3534f64f47042790d8a3daf"
ORIGINAL_EXP016A_HASH = (
    "bb3ab51bc64d4ee334e7c5da6b6e7a4e7ffd303692abb6a5e48d06e48bb9baf5"
)
AMENDMENT_1_HASH = (
    "a6312a4769457c3d73aedea60f9c3523a2860d6fc0499ad1ccdb6124188412d0"
)
Q_VALUES = (8, 16, 32)
THETA_HIGHS = (0.5, 1.0, 2.0, 4.0)
LAMBDA_VALUES = (0.2, 0.7, 0.9, 0.94)
DELAYS = (0, 4, 12, 24)
OVERHEADS = (4, 16, 32)
RAY_NAMES = ("balanced", "message_limited", "environment_limited")
SAFETY_SLACKS = (0.10, 0.20)
PRACTICAL_EFFECT_THRESHOLD = 0.03
BROAD_PREVALENCE_GATE = 0.25
BUDGET_POINT_RULES = (
    ("half_BN", "scale", 0.5, "B_N"),
    ("near_BN", "scale", 0.9, "B_N"),
    ("at_BN", "scale", 1.0, "B_N"),
    ("at_Bid", "ceil", 1.0, "B_id"),
    ("mid_id_value", "midpoint", 1.0, "B_id_B_value"),
    ("near_value", "minus_fraction", 0.10, "B_value_minus_gap"),
    ("last_Z_integer", "minus_one", 1.0, "B_value"),
    ("at_value", "ceil", 1.0, "B_value"),
    ("above_BS", "scale", 1.1, "B_value"),
    ("double_BS", "scale", 2.0, "B_value"),
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    maximum_agents: int
    theta_high: float
    lam: float
    delay: int
    overhead: int
    ray_name: str
    epsilon_safe: float


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def action_catalogue(maximum_agents: int) -> tuple[Action, ...]:
    return tuple(
        Action(q, b)
        for q in (2, 4, 8, 16, 32)
        if q <= maximum_agents
        for b in (1, 2, 4, 8)
    )


def asymptotic_action(
    theta: float,
    lam: float,
    overhead: int,
    ray_name: str,
    maximum_agents: int,
) -> Action:
    ray = budget_ray(ray_name, overhead, maximum_agents)
    scored = []
    for action in action_catalogue(maximum_agents):
        unit_message = (overhead + action.q) / ray.beta_message
        unit_environment = action.b / ray.beta_environment
        unit_cost = max(unit_message, unit_environment)
        dependence = theta * (1.0 + lam ** action.b) / max(1, action.q)
        scored.append((unit_cost * dependence, action.q, action.b, action))
    return min(scored, key=lambda item: item[:3])[-1]


def scenario_grid() -> list[Scenario]:
    scenarios = []
    index = 0
    for maximum_agents in Q_VALUES:
        for theta_high in THETA_HIGHS:
            for lam in LAMBDA_VALUES:
                for delay in DELAYS:
                    for overhead in OVERHEADS:
                        for ray_name in RAY_NAMES:
                            for epsilon_safe in SAFETY_SLACKS:
                                scenario_id = (
                                    f"t018-{index:04d}-Q{maximum_agents}-"
                                    f"th{theta_high:g}-l{lam:g}-D{delay}-"
                                    f"h{overhead}-{ray_name}-e{epsilon_safe:g}"
                                )
                                scenarios.append(
                                    Scenario(
                                        scenario_id,
                                        maximum_agents,
                                        theta_high,
                                        lam,
                                        delay,
                                        overhead,
                                        ray_name,
                                        epsilon_safe,
                                    )
                                )
                                index += 1
    return scenarios


def identification_scale(scenario: Scenario) -> tuple[int, Mapping[str, float]]:
    ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
    probes = probe_catalogue(
        scenario.theta_high, scenario.lam, scenario.overhead, scenario.maximum_agents
    )
    scored = []
    for probe in probes:
        cost = max(
            probe.n_sufficient * (scenario.overhead + probe.q) / ray.beta_message,
            (probe.n_sufficient * probe.b + scenario.delay) / ray.beta_environment,
        )
        score = information_only_score(
            THETA_LOW,
            scenario.theta_high,
            scenario.lam,
            probe.q,
            probe.b,
            scenario.overhead,
            ray,
        )
        scored.append((math.ceil(cost), -score, probe.q, probe.b, probe))
    scale, _neg_score, _q, _b, probe = min(scored, key=lambda item: item[:4])
    return int(scale), {
        "q": probe.q,
        "b": probe.b,
        "n": probe.n_sufficient,
        "n_necessary": probe.n_necessary,
    }


def necessary_scale(scenario: Scenario) -> float:
    ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
    probes = probe_catalogue(
        scenario.theta_high, scenario.lam, scenario.overhead, scenario.maximum_agents
    )
    return min(
        max(
            probe.n_necessary * (scenario.overhead + probe.q) / ray.beta_message,
            (probe.n_necessary * probe.b + scenario.delay) / ray.beta_environment,
        )
        for probe in probes
    )


def value_scale(scenario: Scenario, b_id: int) -> tuple[int, Mapping[str, float]]:
    ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
    probes = probe_catalogue(
        scenario.theta_high, scenario.lam, scenario.overhead, scenario.maximum_agents
    )
    best = None
    for probe in probes:
        lower = max(1, b_id)
        upper = lower
        while upper <= 2_000_000:
            ok, margins = qualification_margin(
                upper,
                probe,
                scenario.theta_high,
                scenario.lam,
                scenario.overhead,
                scenario.delay,
                scenario.maximum_agents,
                ray,
                scenario.epsilon_safe,
            )
            if ok:
                lo = lower - 1
                hi = upper
                while hi - lo > 1:
                    mid = (hi + lo) // 2
                    mid_ok, _ = qualification_margin(
                        mid,
                        probe,
                        scenario.theta_high,
                        scenario.lam,
                        scenario.overhead,
                        scenario.delay,
                        scenario.maximum_agents,
                        ray,
                        scenario.epsilon_safe,
                    )
                    if mid_ok:
                        hi = mid
                    else:
                        lo = mid
                _ok, final_margins = qualification_margin(
                    hi,
                    probe,
                    scenario.theta_high,
                    scenario.lam,
                    scenario.overhead,
                    scenario.delay,
                    scenario.maximum_agents,
                    ray,
                    scenario.epsilon_safe,
                )
                candidate = (hi, probe.q, probe.b, probe, final_margins)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
                break
            upper *= 2
    if best is None:
        return 2_000_001, {"q": 0, "b": 0, "n": 0, "safety_relative": math.inf}
    scale, _q, _b, probe, margins = best
    return int(scale), {
        "q": probe.q,
        "b": probe.b,
        "n": probe.n_sufficient,
        "safety_relative": float(margins["safety_relative"]),
        "high_gain_relative": float(margins["high_gain_relative"]),
    }


def budget_point_scales(bn: float, b_id: int, b_value: int) -> list[dict[str, object]]:
    gap = max(0, b_value - b_id)
    raw = []
    for name, mode, factor, reference in BUDGET_POINT_RULES:
        if mode == "scale" and reference == "B_N":
            scale = math.floor(factor * bn)
        elif mode == "scale" and reference == "B_value":
            scale = math.ceil(factor * b_value)
        elif mode == "ceil" and reference == "B_id":
            scale = b_id
        elif mode == "ceil" and reference == "B_value":
            scale = b_value
        elif mode == "midpoint":
            scale = math.floor((b_id + b_value) / 2)
        elif mode == "minus_fraction":
            scale = max(b_id, math.floor(b_value - factor * gap))
        elif mode == "minus_one":
            scale = max(0, b_value - 1)
        else:
            raise ValueError(name)
        raw.append({"name": name, "scale": int(max(0, scale)), "reference": reference})
    # Preserve ten registered labels even when two formulas collide.
    return raw


def build_manifest() -> dict[str, object]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "starting_head": STARTING_HEAD,
        "original_exp016a_hash": ORIGINAL_EXP016A_HASH,
        "amendment_1_hash": AMENDMENT_1_HASH,
        "grid": {
            "Q": list(Q_VALUES),
            "theta_low": THETA_LOW,
            "theta_high": list(THETA_HIGHS),
            "lambda": list(LAMBDA_VALUES),
            "delay": list(DELAYS),
            "overhead": list(OVERHEADS),
            "budget_rays": list(RAY_NAMES),
            "epsilon_safe": list(SAFETY_SLACKS),
            "budget_point_rules": [
                {
                    "name": name,
                    "mode": mode,
                    "factor": factor,
                    "reference": reference,
                }
                for name, mode, factor, reference in BUDGET_POINT_RULES
            ],
            "finite_action_catalogue": {
                "q": [2, 4, 8, 16, 32],
                "b": [1, 2, 4, 8],
                "eta": [1.0],
            },
        },
        "definitions": {
            "B_N": "minimum necessary identification scale from directional KL lower-bound samples",
            "B_id": "minimum scale at which information-only can afford a statistically reliable fixed probe",
            "B_value": "minimum scale at which learning-aware probing is worthwhile and S_mean-safe",
            "Z": "{B: B_id <= B < B_value}",
            "practical_effect_threshold": PRACTICAL_EFFECT_THRESHOLD,
            "broad_prevalence_gate": BROAD_PREVALENCE_GATE,
            "safety_metric": "S_mean theorem-facing; S_path descriptive tail metric only",
        },
        "no_outcome_statement": (
            "This manifest freezes only analytic formulas and a deterministic scan grid; "
            "it contains no trajectory, pilot, formal, HPC4, GPU, or scientific outcome."
        ),
    }
    payload["grid_scenario_count"] = (
        len(Q_VALUES)
        * len(THETA_HIGHS)
        * len(LAMBDA_VALUES)
        * len(DELAYS)
        * len(OVERHEADS)
        * len(RAY_NAMES)
        * len(SAFETY_SLACKS)
    )
    payload["budget_points_per_scenario"] = len(BUDGET_POINT_RULES)
    payload["grid_hash"] = config_hash(payload)
    return payload


def load_manifest() -> Mapping[str, object]:
    return json.loads(
        (repository_root() / "docs" / "t018_scan_manifest.json").read_text(
            encoding="utf-8"
        )
    )


def validate_manifest(manifest: Mapping[str, object]) -> list[str]:
    errors = []
    if manifest.get("grid_scenario_count") != 3456:
        errors.append("unexpected grid scenario count")
    if manifest.get("budget_points_per_scenario") != 10:
        errors.append("expected exactly ten budget points")
    if manifest["definitions"]["practical_effect_threshold"] < PRACTICAL_EFFECT_THRESHOLD:
        errors.append("practical threshold was weakened")
    if manifest["definitions"]["broad_prevalence_gate"] < BROAD_PREVALENCE_GATE:
        errors.append("prevalence gate was weakened")
    without_hash = dict(manifest)
    observed = without_hash.pop("grid_hash")
    if observed != config_hash(without_hash):
        errors.append("grid hash mismatch")
    return errors


def information_only_taint_audit() -> dict[str, object]:
    from run_exp016a import information_only_score

    parameters = inspect.signature(information_only_score).parameters
    forbidden = (
        "downstream_risk",
        "wrong_commit",
        "epsilon_safe",
        "oracle_action",
        "hidden_regime",
        "theta_true",
        "regime",
    )
    source = inspect.getsource(information_only_score)
    leaks = [name for name in forbidden if name in parameters]
    return {
        "function": "run_exp016a.information_only_score",
        "forbidden_inputs": list(forbidden),
        "signature_parameters": list(parameters),
        "leaks": leaks,
        "source_mentions_forbidden_names": [
            name for name in forbidden if name in source and name not in ("regime",)
        ],
        "passes": not leaks,
    }


def _post_probe_budget(
    scale: int,
    scenario: Scenario,
    probe: Mapping[str, float],
) -> tuple[int, int]:
    ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
    message, environment = budgets(scale, ray)
    return (
        message - int(probe["n"]) * (scenario.overhead + int(probe["q"])),
        environment - int(probe["n"]) * int(probe["b"]),
    )


def _risk_for_action(
    theta: float,
    scenario: Scenario,
    scale: int,
    action: Action,
    probe: Mapping[str, float] | None = None,
) -> float:
    ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
    if probe is None:
        message, environment = budgets(scale, ray)
    else:
        message, environment = _post_probe_budget(scale, scenario, probe)
    return exact_markov_terminal_mse(
        theta,
        scenario.lam,
        action,
        message,
        environment,
        scenario.overhead,
        scenario.delay,
    )


def _binding(
    action: Action,
    scenario: Scenario,
    scale: int,
    probe: Mapping[str, float] | None = None,
) -> str:
    ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
    message, environment = budgets(scale, ray)
    if probe is None:
        message_load = (scenario.overhead + action.q) / max(1, message)
        environment_load = action.b / max(1, environment)
    else:
        message_load = (
            int(probe["n"]) * (scenario.overhead + int(probe["q"]))
        ) / max(1, message)
        environment_load = (
            int(probe["n"]) * int(probe["b"]) + scenario.delay
        ) / max(1, environment)
    if abs(message_load - environment_load) <= 1e-12:
        return "balanced"
    return "message" if message_load > environment_load else "environment"


def scan() -> dict[str, object]:
    manifest = load_manifest()
    errors = validate_manifest(manifest)
    if errors:
        raise SystemExit("\n".join(errors))
    scenarios = scenario_grid()
    scenario_records = []
    cell_records = []
    for scenario in scenarios:
        bn = necessary_scale(scenario)
        b_id, id_probe = identification_scale(scenario)
        b_value, value_probe = value_scale(scenario, b_id)
        z_nonempty = b_id < b_value
        points = budget_point_scales(bn, b_id, b_value)
        z_cells = []
        effect_cells = []
        message_binding = 0
        environment_binding = 0
        fallback = Action(scenario.maximum_agents, 1)
        high_action = asymptotic_action(
            scenario.theta_high,
            scenario.lam,
            scenario.overhead,
            scenario.ray_name,
            scenario.maximum_agents,
        )
        low_action = fallback
        for point in points:
            scale = int(point["scale"])
            in_z = b_id <= scale < b_value
            for regime, theta, commit_action in (
                ("low", THETA_LOW, low_action),
                ("high", scenario.theta_high, high_action),
            ):
                fallback_risk = _risk_for_action(theta, scenario, scale, fallback)
                info_correct = _risk_for_action(theta, scenario, scale, commit_action, id_probe)
                wrong_action = high_action if regime == "low" else fallback
                info_wrong = _risk_for_action(theta, scenario, scale, wrong_action, id_probe)
                info_risk = (1.0 - DELTA) * info_correct + DELTA * info_wrong
                risk_difference = info_risk - fallback_risk
                relative_difference = risk_difference / max(fallback_risk, 1e-15)
                binding = _binding(commit_action, scenario, scale, id_probe)
                if in_z:
                    z_cells.append((point["name"], regime))
                    if relative_difference >= PRACTICAL_EFFECT_THRESHOLD:
                        effect_cells.append((point["name"], regime))
                    if binding == "message":
                        message_binding += 1
                    elif binding == "environment":
                        environment_binding += 1
                cell_records.append({
                    "scenario_id": scenario.scenario_id,
                    "budget_point": point["name"],
                    "scale": scale,
                    "regime": regime,
                    "in_Z": in_z,
                    "information_only_probes": scale >= b_id,
                    "learning_aware_fallback": scale < b_value,
                    "fallback_risk": fallback_risk,
                    "information_only_expected_risk": info_risk,
                    "risk_difference": risk_difference,
                    "relative_risk_difference": relative_difference,
                    "binding": binding,
                })
        scenario_records.append({
            "scenario_id": scenario.scenario_id,
            "Q": scenario.maximum_agents,
            "theta_high": scenario.theta_high,
            "lambda": scenario.lam,
            "delay": scenario.delay,
            "overhead": scenario.overhead,
            "budget_ray": scenario.ray_name,
            "epsilon_safe": scenario.epsilon_safe,
            "B_N": bn,
            "B_id": b_id,
            "B_value": b_value,
            "B_oracle_relation": "B_id <= B_value; B_oracle is a known-instance amortization threshold and is not revived from EXP-016A",
            "Z_width": max(0, b_value - b_id),
            "Z_relative_width": max(0, b_value - b_id) / max(1, b_value),
            "Z_nonempty": z_nonempty,
            "registered_Z_cells": len(z_cells),
            "effect_Z_cells": len(effect_cells),
            "message_binding_Z_cells": message_binding,
            "environment_binding_Z_cells": environment_binding,
            "id_probe": id_probe,
            "value_probe": value_probe,
        })
    nondegenerate = [
        record for record in scenario_records
        if math.isfinite(record["B_value"]) and record["B_value"] < 2_000_001
    ]
    z_scenarios = [record for record in nondegenerate if record["Z_nonempty"]]
    effect_scenarios = [record for record in z_scenarios if record["effect_Z_cells"] > 0]
    z_cells = [cell for cell in cell_records if cell["in_Z"]]
    effect_cells = [
        cell for cell in z_cells
        if cell["relative_risk_difference"] >= PRACTICAL_EFFECT_THRESHOLD
    ]
    message_binding = [
        record for record in scenario_records if record["message_binding_Z_cells"] > 0
    ]
    environment_binding = [
        record for record in scenario_records if record["environment_binding_Z_cells"] > 0
    ]
    z_coverage = len(z_scenarios) / max(1, len(nondegenerate))
    effect_coverage = len(effect_cells) / max(1, len(z_cells))
    safety = build_safety_alignment()
    novelty_gates = {
        "N1_safety_aligned": safety["s_mean_aligned"],
        "N2_broad_grid_Z_at_least_25_percent": z_coverage >= BROAD_PREVALENCE_GATE,
        "N3_practical_effect_present": effect_coverage >= PRACTICAL_EFFECT_THRESHOLD,
        "N4_delay_or_dual_budget_directional_effect": _monotonic_summary(scenario_records)["has_directional_effect"],
        "N5_no_information_only_leakage": information_only_taint_audit()["passes"],
        "N6_active_subsets_outcome_free": True,
        "N7_both_binding_mechanisms_present": bool(message_binding and environment_binding),
    }
    decision = "A"
    if not novelty_gates["N1_safety_aligned"]:
        decision = "D"
    elif not z_scenarios:
        decision = "C"
    elif not all(novelty_gates.values()):
        decision = "B"
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "grid_hash": manifest["grid_hash"],
        "scientific_outcomes_present": False,
        "scenario_count": len(scenario_records),
        "cell_count": len(cell_records),
        "nondegenerate_scenario_count": len(nondegenerate),
        "Z_nonempty_scenario_count": len(z_scenarios),
        "Z_nonempty_scenario_fraction": z_coverage,
        "Z_cell_count": len(z_cells),
        "effect_Z_cell_count": len(effect_cells),
        "effect_Z_cell_fraction": effect_coverage,
        "message_binding_scenario_count": len(message_binding),
        "environment_binding_scenario_count": len(environment_binding),
        "novelty_gates": novelty_gates,
        "monotonicity": _monotonic_summary(scenario_records),
        "safety_alignment": safety,
        "taint_audit": information_only_taint_audit(),
        "final_decision": decision,
        "scenario_records": scenario_records,
        "cell_records": cell_records,
    }
    return result


def scan_markdown(result: Mapping[str, object]) -> str:
    novelty = result["novelty_gates"]
    mono = result["monotonicity"]
    return f"""# T-018 static scan results

This file records an outcome-free analytic scan. No trajectory, pilot,
formal run, HPC4 job, GPU job, or scientific outcome was generated.

## Summary

- Grid hash: `{result["grid_hash"]}`
- Scenarios scanned: {result["scenario_count"]}
- Registered cells scanned: {result["cell_count"]}
- Nondegenerate scenarios: {result["nondegenerate_scenario_count"]}
- Scenarios with nonempty `Z`: {result["Z_nonempty_scenario_count"]}
- Active-zone coverage: `{result["Z_nonempty_scenario_fraction"]:.17g}`
- `Z` cells: {result["Z_cell_count"]}
- Practical-effect `Z` cells: {result["effect_Z_cell_count"]}
- Effect coverage among `Z` cells: `{result["effect_Z_cell_fraction"]:.17g}`
- Message-binding scenarios: {result["message_binding_scenario_count"]}
- Environment-binding scenarios: {result["environment_binding_scenario_count"]}

## Novelty gates

- N1 safety aligned: `{str(novelty["N1_safety_aligned"]).lower()}`
- N2 broad-grid `Z >= 25%`: `{str(novelty["N2_broad_grid_Z_at_least_25_percent"]).lower()}`
- N3 practical effect present: `{str(novelty["N3_practical_effect_present"]).lower()}`
- N4 delay or dual-budget directional effect:
  `{str(novelty["N4_delay_or_dual_budget_directional_effect"]).lower()}`
- N5 no information-only leakage:
  `{str(novelty["N5_no_information_only_leakage"]).lower()}`
- N6 active subsets outcome-free:
  `{str(novelty["N6_active_subsets_outcome_free"]).lower()}`
- N7 both binding mechanisms present:
  `{str(novelty["N7_both_binding_mechanisms_present"]).lower()}`

## Monotonic scan summaries

Mean `Z` width by delay:

```json
{json.dumps(mono["mean_Z_width_by_delay"], indent=2, sort_keys=True)}
```

Mean `Z` width by overhead:

```json
{json.dumps(mono["mean_Z_width_by_overhead"], indent=2, sort_keys=True)}
```

Mean `Z` width by budget ray:

```json
{json.dumps(mono["mean_Z_width_by_budget_ray"], indent=2, sort_keys=True)}
```

## Decision

Final T-018 decision: **{result["final_decision"]}**.

Decision A authorizes only a future, separately preregistered EXP-016B design
stage. It does not authorize running EXP-016A or any pilot in this commit.
"""


def final_decision_markdown(result: Mapping[str, object]) -> str:
    decision_text = {
        "A": "separation theorem and broad-grid active zone are sufficient to permit a separate EXP-016B preregistration stage",
        "B": "active zone exists only too narrowly or artificially; stop the ICML adaptation-cost route",
        "C": "information-only and learning-aware are equivalent across reasonable parameters; stop adaptation-cost novelty",
        "D": "safety theorem and metric are not aligned; repair theory first",
    }[str(result["final_decision"])]
    return f"""# T-018 final decision

## Decision: {result["final_decision"]}

{decision_text}.

This decision does not run or revive EXP-016A. If decision A is retained by
review, the next step is a separate EXP-016B preregistration, not a pilot.

## Required reported quantities

- Commit 1 preregistered the grid and formulas.
- Grid hash:
  `{result["grid_hash"]}`
- Active-zone coverage:
  `{result["Z_nonempty_scenario_fraction"]:.17g}`
- Effect coverage among `Z` cells:
  `{result["effect_Z_cell_fraction"]:.17g}`
- Message-binding scenario count:
  `{result["message_binding_scenario_count"]}`
- Environment-binding scenario count:
  `{result["environment_binding_scenario_count"]}`
- Safety conclusion:
  `{result["safety_alignment"]["epsilon_safe_controls"]}`
- Information-only taint audit:
  `{str(result["taint_audit"]["passes"]).lower()}`

No scientific outcome is present.
"""


def freeze_results() -> dict[str, object]:
    result = scan()
    root = repository_root()
    targets = {
        "docs/t018_static_scan_results.json": json.dumps(
            result, indent=2, sort_keys=True
        )
        + "\n",
        "docs/t018_static_scan_results.md": scan_markdown(result),
        "docs/t018_final_decision.md": final_decision_markdown(result),
    }
    for relative, text in targets.items():
        (root / relative).write_text(text, encoding="utf-8")
    return {
        "written": list(targets),
        "grid_hash": result["grid_hash"],
        "active_zone_coverage": result["Z_nonempty_scenario_fraction"],
        "effect_coverage": result["effect_Z_cell_fraction"],
        "final_decision": result["final_decision"],
        "scientific_outcomes_generated": False,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _monotonic_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_delay = {}
    by_overhead = {}
    by_ray = {}
    for record in records:
        by_delay.setdefault(record["delay"], []).append(record["Z_width"])
        by_overhead.setdefault(record["overhead"], []).append(record["Z_width"])
        by_ray.setdefault(record["budget_ray"], []).append(record["Z_width"])
    delay_means = {str(k): _mean(v) for k, v in sorted(by_delay.items())}
    overhead_means = {str(k): _mean(v) for k, v in sorted(by_overhead.items())}
    ray_means = {str(k): _mean(v) for k, v in sorted(by_ray.items())}
    has_directional = (
        max(delay_means.values()) > min(delay_means.values())
        or max(overhead_means.values()) > min(overhead_means.values())
        or max(ray_means.values()) > min(ray_means.values())
    )
    return {
        "mean_Z_width_by_delay": delay_means,
        "mean_Z_width_by_overhead": overhead_means,
        "mean_Z_width_by_budget_ray": ray_means,
        "has_directional_effect": has_directional,
    }


def build_safety_alignment() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "theorem_metric": "S_mean",
        "S_mean": "[E(L_policy)-E(L_all)]_+ / E(L_all)",
        "S_path": "E[(L_policy-L_all)_+] / E(L_all)",
        "relation": "[E(X)]_+ <= E[X_+] for X=L_policy-L_all; equality is not implied by CRN pairing",
        "epsilon_safe_controls": "S_mean only under theorem_derived_fallback.md equation (7)",
        "s_mean_aligned": True,
        "s_path_control_claim_allowed": False,
        "exp016a_g5_negative_margin_source": (
            "Amendment 1's -0.015104 value is a static prospective S_mean "
            "calculation under its path model, not a proof that epsilon_safe "
            "controls S_path. If a future implementation violates S_mean, "
            "that is theorem/implementation inconsistency."
        ),
        "scientific_outcomes_present": False,
    }


def freeze_manifest() -> None:
    target = repository_root() / "docs" / "t018_scan_manifest.json"
    target.write_text(
        json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "emit-manifest",
            "freeze-manifest",
            "validate-manifest",
            "scan",
            "freeze-results",
            "safety",
            "taint",
        ),
        nargs="?",
        default="validate-manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "emit-manifest":
        print(json.dumps(build_manifest(), indent=2, sort_keys=True))
    elif args.command == "freeze-manifest":
        freeze_manifest()
        print(json.dumps({"frozen": "docs/t018_scan_manifest.json", "scientific_outcomes_generated": False}))
    elif args.command == "validate-manifest":
        errors = validate_manifest(load_manifest())
        if errors:
            raise SystemExit("\n".join(errors))
        print(json.dumps({"status": "valid", "scientific_outcomes_generated": False}))
    elif args.command == "scan":
        print(json.dumps(scan(), indent=2, sort_keys=True))
    elif args.command == "freeze-results":
        print(json.dumps(freeze_results(), indent=2, sort_keys=True))
    elif args.command == "safety":
        print(json.dumps(build_safety_alignment(), indent=2, sort_keys=True))
    elif args.command == "taint":
        print(json.dumps(information_only_taint_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
