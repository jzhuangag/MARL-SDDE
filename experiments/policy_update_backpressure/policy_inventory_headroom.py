"""Outcome-free policy-inventory controller headroom scan.

The calculation evaluates the exact certified one-event drift on a frozen grid
of deterministic workload rays.  It is a design ceiling, not sampled efficacy
or formal paper evidence.  ``run`` is forbidden until the plan is committed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .policy_inventory_theory import (
    CompletedInventoryProposal,
    PendingInventory,
    importance_variance_inflation,
    inventory_lyapunov_drift,
    inventory_optimal_step,
)


EVENTS = 72
POTENTIAL_WEIGHT = 4.0
MAX_STEP = 1.0


@dataclass(frozen=True)
class Scenario:
    n_agents: int
    workload: str
    geometry: str
    policy_variance: float

    @property
    def scenario_id(self) -> str:
        return (
            f"n{self.n_agents}-{self.workload}-{self.geometry}"
            f"-v{self.policy_variance:g}"
        )


def declared_scenarios() -> tuple[Scenario, ...]:
    return tuple(
        Scenario(n, workload, geometry, variance)
        for n in (3, 5, 8)
        for workload in ("steady_low", "steady_high", "alternating", "bursty")
        for geometry in ("away", "mixed", "rotating")
        for variance in (0.75, 1.25)
    )


def pending_count(scenario: Scenario, event: int) -> int:
    low = 0 if scenario.n_agents == 3 else 1
    high = scenario.n_agents-1
    if scenario.workload == "steady_low":
        return low
    if scenario.workload == "steady_high":
        return high
    if scenario.workload == "alternating":
        return low if (event//8) % 2 == 0 else high
    if scenario.workload == "bursty":
        return high if event % 18 in range(11, 16) else low
    raise ValueError("unknown workload")


def baseline_specs() -> tuple[tuple[str, str, float], ...]:
    specs: list[tuple[str, str, float]] = [("certified_base", "scale", 1.0)]
    specs.extend((f"scale:{x:g}", "scale", x) for x in (0.25, 0.5, 0.75))
    for family, values in (
        ("count", (0.1, 0.25, 0.5, 1.0, 2.0)),
        ("risk", (0.25, 0.5, 1.0, 2.0, 4.0)),
        ("absolute_linear", (0.25, 0.5, 1.0, 2.0, 4.0)),
        ("count_threshold", (0.0, 1.0, 2.0, 4.0, 6.0)),
        ("risk_threshold", (0.25, 0.5, 1.0, 2.0, 4.0)),
    ):
        specs.extend((f"{family}:{x:g}", family, x) for x in values)
    return tuple(specs)


def design_payload() -> dict[str, object]:
    scenarios = declared_scenarios()
    return {
        "events_per_scenario": EVENTS,
        "geometries": ["away", "mixed", "rotating"],
        "max_step": MAX_STEP,
        "n_agents": [3, 5, 8],
        "policies": 1+len(baseline_specs()),
        "policy_variances": [0.75, 1.25],
        "potential_weight": POTENTIAL_WEIGHT,
        "scenarios": len(scenarios),
        "states": len(scenarios)*EVENTS,
        "workloads": ["steady_low", "steady_high", "alternating", "bursty"],
    }


def design_hash() -> str:
    encoded = json.dumps(
        design_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_design() -> dict[str, object]:
    scenarios = declared_scenarios()
    specs = baseline_specs()
    if len(scenarios) != 72 or len(specs) != 29:
        raise AssertionError("policy-inventory design cardinality changed")
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise AssertionError("duplicate scenario identifier")
    return {
        "design": design_payload(),
        "design_hash": design_hash(),
        "status": "static_design_valid_no_outcomes",
    }


def _state(
    scenario: Scenario, event: int
) -> tuple[CompletedInventoryProposal, tuple[PendingInventory, ...], dict[str, float]]:
    count = pending_count(scenario, event)
    signal = 0.28+1.72*(0.978**event)
    signal *= 1.0+0.12*math.sin(2.0*math.pi*event/18.0)
    z_completed = 0.03+0.055*count+0.025*(event % 4)
    error_radius = 0.32*math.exp(0.5*z_completed)/math.sqrt(16.0)
    proposal = CompletedInventoryProposal(
        direction_norm=signal,
        error_radius=error_radius,
        block_smoothness=1.0,
        log_second_moment=z_completed,
        max_step=MAX_STEP,
    )
    pending: list[PendingInventory] = []
    for index in range(count):
        displacement = (0.035+0.012*index)*(1.0+0.05*count)
        if scenario.geometry == "away":
            sign = 1.0
        elif scenario.geometry == "mixed":
            sign = 1.0 if index % 2 == 0 else -1.0
        elif scenario.geometry == "rotating":
            sign = 1.0 if (event//6+index) % 3 else -1.0
        else:
            raise ValueError("unknown inventory geometry")
        linear = 2.0*sign*displacement*signal/scenario.policy_variance
        quadratic = signal*signal/scenario.policy_variance
        minimum_reduction = displacement*displacement/scenario.policy_variance
        z0 = (
            0.08+0.055*(index+1)+0.02*(event % 3)+minimum_reduction
        )
        pending.append(PendingInventory(
            log_second_moment=z0,
            linear=linear,
            quadratic=quadratic,
            weight=0.45,
        ))
    observables = {
        "count": float(count),
        "risk": float(sum(
            importance_variance_inflation(item.log_second_moment)
            for item in pending
        )),
        "absolute_linear": float(sum(abs(item.linear) for item in pending)),
    }
    return proposal, tuple(pending), observables


def _base_step(proposal: CompletedInventoryProposal) -> float:
    if proposal.direction_norm == 0:
        return 0.0
    return float(np.clip(
        (1.0-proposal.error_radius/proposal.direction_norm)
        / proposal.block_smoothness,
        0.0,
        proposal.max_step,
    ))


def _baseline_step(
    spec: tuple[str, str, float],
    proposal: CompletedInventoryProposal,
    observables: dict[str, float],
) -> float:
    _, family, value = spec
    base = _base_step(proposal)
    if family == "scale":
        return value*base
    if family in ("count", "risk", "absolute_linear"):
        return base/(1.0+value*observables[family])
    if family == "count_threshold":
        return base if observables["count"] <= value else 0.0
    if family == "risk_threshold":
        return base if observables["risk"] <= value else 0.0
    raise ValueError("unknown baseline family")


def _action_objective(
    proposal: CompletedInventoryProposal,
    pending: tuple[PendingInventory, ...],
    step: float,
) -> float:
    # Remove the action-independent benefit from consuming the completed item.
    return float(
        inventory_lyapunov_drift(
            proposal, pending, step, potential_weight=POTENTIAL_WEIGHT
        )
        + proposal.weight*importance_variance_inflation(
            proposal.log_second_moment
        )
    )


def run(output: Path) -> dict[str, object]:
    validation = validate_design()
    specs = baseline_specs()
    rows: list[dict[str, object]] = []
    scenario_results: list[dict[str, object]] = []
    low_recovery: list[bool] = []
    high_nonzero: list[bool] = []
    for scenario in declared_scenarios():
        dynamic_utility = 0.0
        baseline_utility = {name: 0.0 for name, _, _ in specs}
        for event in range(EVENTS):
            proposal, pending, observables = _state(scenario, event)
            dynamic_step = inventory_optimal_step(
                proposal, pending, potential_weight=POTENTIAL_WEIGHT
            )
            dynamic_objective = _action_objective(
                proposal, pending, dynamic_step
            )
            dynamic_utility += max(-dynamic_objective, 0.0)
            if not pending:
                low_recovery.append(abs(dynamic_step-_base_step(proposal)) <= 1e-9)
            if int(observables["count"]) == scenario.n_agents-1:
                high_nonzero.append(dynamic_step > 1e-9)
            rows.append({
                "scenario_id": scenario.scenario_id,
                "event": event,
                "policy": "inventory_continuous",
                "step": dynamic_step,
                "action_objective": dynamic_objective,
                "safe_utility": max(-dynamic_objective, 0.0),
                **observables,
            })
            for spec in specs:
                name = spec[0]
                proposed_step = _baseline_step(spec, proposal, observables)
                objective = _action_objective(proposal, pending, proposed_step)
                # Give every comparator the same exact no-harm fallback.
                step = proposed_step if objective < 0 else 0.0
                safe_utility = max(-objective, 0.0)
                baseline_utility[name] += safe_utility
                rows.append({
                    "scenario_id": scenario.scenario_id,
                    "event": event,
                    "policy": name,
                    "step": step,
                    "action_objective": min(objective, 0.0),
                    "safe_utility": safe_utility,
                    **observables,
                })
        best_name = max(baseline_utility, key=lambda name: (baseline_utility[name], name))
        best_utility = baseline_utility[best_name]
        ratio = dynamic_utility/max(best_utility, 1e-15)
        scenario_results.append({
            **asdict(scenario),
            "scenario_id": scenario.scenario_id,
            "dynamic_utility": dynamic_utility,
            "best_baseline": best_name,
            "best_baseline_utility": best_utility,
            "utility_ratio": ratio,
            "relative_gain": ratio-1.0,
        })
    ratios = np.asarray([row["utility_ratio"] for row in scenario_results], dtype=float)
    aggregate_dynamic = float(sum(row["dynamic_utility"] for row in scenario_results))
    aggregate_baseline = float(sum(row["best_baseline_utility"] for row in scenario_results))
    aggregate_ratio = aggregate_dynamic/aggregate_baseline
    finite = all(
        np.isfinite(float(row[key]))
        for row in rows
        for key in ("step", "action_objective", "safe_utility")
    )
    gates = {
        "complete_finite": finite and len(rows) == int(
            validation["design"]["states"]*validation["design"]["policies"]
        ),
        "aggregate_utility_ratio_ge_1_05": aggregate_ratio >= 1.05,
        "scenario_five_percent_fraction_ge_0_60": float(np.mean(ratios >= 1.05)) >= 0.60,
        "zero_pending_base_recovery_ge_0_99": float(np.mean(low_recovery)) >= 0.99,
        "high_inventory_nonzero_fraction_ge_0_30": float(np.mean(high_nonzero)) >= 0.30,
        "pointwise_dynamic_dominance": all(
            float(row["utility_ratio"]) >= 1.0-1e-10
            for row in scenario_results
        ),
    }
    payload = {
        "kind": "outcome_free_policy_inventory_headroom",
        **validation,
        "aggregate": {
            "dynamic_utility": aggregate_dynamic,
            "best_baseline_utility": aggregate_baseline,
            "utility_ratio": aggregate_ratio,
            "scenario_five_percent_fraction": float(np.mean(ratios >= 1.05)),
            "median_relative_gain": float(np.median(ratios-1.0)),
            "zero_pending_base_recovery": float(np.mean(low_recovery)),
            "high_inventory_nonzero_fraction": float(np.mean(high_nonzero)),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "stochastic_pilot_authorized": False,
        "scenario_results": scenario_results,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: Iterable[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    if args.mode == "validate":
        return validate_design()
    if args.output is None:
        raise ValueError("run requires --output")
    return run(args.output)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))

