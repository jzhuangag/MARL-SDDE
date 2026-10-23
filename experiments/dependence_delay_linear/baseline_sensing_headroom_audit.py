"""Outcome-free oracle-headroom audit for baseline-as-sensing.

This module deliberately evaluates only the public T-018 analytic model.  It
does not import runners, read trajectories, or use pilot/formal outcomes.  A
perfect regime-aware selector is an *upper bound* on what a sensing algorithm
could obtain: it incurs no probe cost and observes the regime for free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from run_exp016a import THETA_LOW, budgets, exact_markov_terminal_mse
from t018_static_scan import (
    action_catalogue,
    budget_point_scales,
    budget_ray,
    scenario_grid,
)


TASK = "baseline-sensing-headroom-audit"
SCHEMA_VERSION = 1
HEADROOM_GATE = 0.10
PREVALENCE_GATE = 0.60
T018_STATIC_THRESHOLDS = "t018_corrected_scan_results.json"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _static_thresholds() -> Mapping[str, Mapping[str, object]]:
    """Load only T-018's deterministic threshold table, never trajectories."""

    payload = json.loads(
        (repository_root() / "docs" / T018_STATIC_THRESHOLDS).read_text(encoding="utf-8")
    )
    return {record["scenario_id"]: record for record in payload["scenario_records"]}


def _risk(theta: float, scenario: object, scale: int, action: object) -> float:
    ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
    message, environment = budgets(scale, ray)
    return exact_markov_terminal_mse(
        theta,
        scenario.lam,
        action,
        message,
        environment,
        scenario.overhead,
        scenario.delay,
    )


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)

    def at(probability: float) -> float:
        return ordered[math.ceil((len(ordered) - 1) * probability)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": at(0.90),
        "max": ordered[-1],
    }


def _stratify(records: Iterable[Mapping[str, object]], field: str) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(float(record["ideal_headroom"]))
    summary = {}
    for key, values in sorted(groups.items()):
        payload = _quantiles(values)
        payload["fraction_at_least_10_percent"] = sum(
            value >= HEADROOM_GATE for value in values
        ) / len(values)
        summary[key] = payload
    return summary


def _records_sha256(records: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strong_fixed(
    risks: Mapping[object, tuple[float, float]],
) -> tuple[object, float, tuple[float, float]]:
    """Return minimax normalized fixed action under low/high regimes."""

    oracle_low = min(pair[0] for pair in risks.values())
    oracle_high = min(pair[1] for pair in risks.values())
    scored = []
    for action, (low, high) in risks.items():
        normalized = (low / oracle_low, high / oracle_high)
        scored.append((max(normalized), action.q, action.b, action, normalized))
    objective, _q, _b, action, normalized = min(scored, key=lambda item: item[:3])
    return action, float(objective), (float(normalized[0]), float(normalized[1]))


def evaluate_budget_cell(
    scenario: object, point: Mapping[str, object], b_value_status: str
) -> dict[str, object] | None:
    """Evaluate one budget cell, returning ``None`` only when no action is feasible."""

    scale = int(point["scale"])
    risks: dict[object, tuple[float, float]] = {}
    for action in action_catalogue(scenario.maximum_agents):
        low = _risk(THETA_LOW, scenario, scale, action)
        high = _risk(scenario.theta_high, scenario, scale, action)
        if math.isfinite(low) and math.isfinite(high):
            risks[action] = (low, high)
    if not risks:
        return None
    oracle_low = min(value[0] for value in risks.values())
    oracle_high = min(value[1] for value in risks.values())
    fixed_action, fixed_objective, fixed_ratios = _strong_fixed(risks)
    all_agent = next(
        (action for action in risks if action.q == scenario.maximum_agents and action.b == 1),
        None,
    )
    all_agent_objective = None
    if all_agent is not None:
        all_agent_objective = max(
            risks[all_agent][0] / oracle_low,
            risks[all_agent][1] / oracle_high,
        )
    return {
        "scenario_id": scenario.scenario_id,
        "Q": scenario.maximum_agents,
        "theta_high": scenario.theta_high,
        "lambda": scenario.lam,
        "delay": scenario.delay,
        "overhead": scenario.overhead,
        "budget_ray": scenario.ray_name,
        "epsilon_safe": scenario.epsilon_safe,
        "budget_point": point["name"],
        "budget_scale": scale,
        "B_value_status": b_value_status,
        "oracle_worst_normalized_ratio": 1.0,
        "strong_fixed_action": {"q": fixed_action.q, "b": fixed_action.b},
        "strong_fixed_worst_normalized_ratio": fixed_objective,
        "strong_fixed_normalized_by_regime": {"low": fixed_ratios[0], "high": fixed_ratios[1]},
        "all_agent_action": ({"q": all_agent.q, "b": all_agent.b} if all_agent is not None else None),
        "all_agent_worst_normalized_ratio": all_agent_objective,
        "ideal_headroom": 1.0 - 1.0 / fixed_objective,
        "feasible_action_count": len(risks),
    }


@lru_cache(maxsize=1)
def audit() -> dict[str, object]:
    """Evaluate all finite T-018 scenario-by-budget cells deterministically."""

    records: list[dict[str, object]] = []
    skipped_no_feasible_action = 0
    censored_scenarios = 0
    thresholds = _static_thresholds()
    for scenario in scenario_grid():
        threshold = thresholds[scenario.scenario_id]
        bn = float(threshold["B_N"])
        bid = int(threshold["B_id"])
        b_value = threshold["B_value"]
        if b_value is None:
            censored_scenarios += 1
        else:
            b_value = int(b_value)
        for point in budget_point_scales(bn, bid, b_value):
            record = evaluate_budget_cell(scenario, point, str(threshold["B_value_status"]))
            if record is None:
                skipped_no_feasible_action += 1
                continue
            records.append(record)

    headrooms = [float(record["ideal_headroom"]) for record in records]
    median = statistics.median(headrooms) if headrooms else float("nan")
    prevalence = sum(value >= HEADROOM_GATE for value in headrooms) / len(headrooms)
    gate_passes = bool(median >= HEADROOM_GATE and prevalence >= PREVALENCE_GATE)
    stratified = {
        field: _stratify(records, field)
        for field in ("Q", "theta_high", "lambda", "delay", "budget_ray")
    }
    all_agent_comparable = [
        record for record in records if record["all_agent_worst_normalized_ratio"] is not None
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "scope": (
            "Deterministic upper-bound audit on the public T-018 scenario grid. "
            "The perfect oracle receives the hidden regime for free; all actions have "
            "identical dual budgets and no probe charge. This is not an experiment and "
            "does not authorize a new algorithm or scientific run."
        ),
        "input_provenance": {
            "scenario_grid": "t018_static_scan.scenario_grid",
            "budget_point_rules": "t018_static_scan.budget_point_scales",
            "action_catalogue": "t018_static_scan.action_catalogue",
            "risk": "run_exp016a.exact_markov_terminal_mse",
            "static_thresholds": "docs/t018_corrected_scan_results.json (deterministic T-018 scan only)",
            "trajectory_or_result_input": False,
        },
        "definitions": {
            "regimes": ["low", "high"],
            "oracle": "R_j^* = min_a R_j(a), separately for each hidden regime j",
            "strong_fixed": "min_a max_j R_j(a) / R_j^* over the full feasible catalogue",
            "ideal_headroom": "1 - 1/U_fixed, where U_fixed is the strong-fixed minimax normalized risk",
            "all_agent": "Action(Q, 1), reported descriptively and never substituted for strong_fixed",
        },
        "counts": {
            "scenario_count": len(scenario_grid()),
            "finite_budget_cells": len(records),
            "skipped_no_feasible_action": skipped_no_feasible_action,
            "censored_scenarios": censored_scenarios,
        },
        "records_sha256": _records_sha256(records),
        "distribution": _quantiles(headrooms),
        "fraction_at_least_10_percent": prevalence,
        "stratified": stratified,
        "all_agent_check": {
            "comparable_cells": len(all_agent_comparable),
            "strong_fixed_no_worse_than_all_agent": all(
                float(record["strong_fixed_worst_normalized_ratio"])
                <= float(record["all_agent_worst_normalized_ratio"]) + 1e-12
                for record in all_agent_comparable
            ),
        },
        "gate": {
            "median_ideal_headroom_at_least": HEADROOM_GATE,
            "fraction_at_least_10_percent_at_least": PREVALENCE_GATE,
            "observed_median": median,
            "observed_fraction": prevalence,
            "passes": gate_passes,
            "decision": (
                "analytic headroom sufficient; a separate outcome-free design review is required"
                if gate_passes
                else "STOP: ideal oracle headroom gate fails; do not authorize a scientific experiment"
            ),
        },
        "records": records,
    }


def markdown(result: Mapping[str, object]) -> str:
    distribution = result["distribution"]
    gate = result["gate"]
    counts = result["counts"]
    return f"""# Baseline-as-sensing analytic headroom audit

This is an outcome-free deterministic calculation on the frozen public T-018
model grid. It reads no trajectory, pilot, or formal result. The oracle is
given the hidden low/high regime free of charge, so its advantage is an upper
bound rather than evidence that a sensing algorithm attains it.

- Finite scenario-by-budget cells: `{counts['finite_budget_cells']}`
- Cells with no feasible catalogue action: `{counts['skipped_no_feasible_action']}`
- Median ideal headroom: `{distribution['median']:.6f}`
- Fraction with ideal headroom at least 10%: `{result['fraction_at_least_10_percent']:.6f}`
- Canonical per-cell record SHA-256: `{result['records_sha256']}`
- Frozen gate (median >= 10% and prevalence >= 60%): `{str(gate['passes']).lower()}`
- Decision: **{gate['decision']}**

The strong baseline is the regime-blind minimax action from the full catalogue,
not the all-agent action. The all-agent calculation is descriptive only.

## Stratified results

```json
{json.dumps(result['stratified'], indent=2, sort_keys=True)}
```
"""


def write_report(result: Mapping[str, object], docs_dir: Path | None = None) -> tuple[Path, Path]:
    target = docs_dir or (repository_root() / "docs")
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "baseline_sensing_headroom_audit_20260905.json"
    markdown_path = target / "baseline_sensing_headroom_audit_20260905.md"
    # Per-cell records remain reproducible from the public analytic inputs and
    # are intentionally omitted from the tracked summary.  Their canonical
    # hash above preserves an exact audit anchor without committing a large
    # generated file.
    summary = dict(result)
    summary.pop("records", None)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(result), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write JSON and Markdown reports under docs")
    args = parser.parse_args()
    result = audit()
    if args.write:
        json_path, markdown_path = write_report(result)
        print(json_path)
        print(markdown_path)
    else:
        print(json.dumps({"gate": result["gate"], "counts": result["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
