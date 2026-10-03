"""EXP-016A preregistered runner and static configuration auditor.

The default command is static validation.  This preregistration-stage runner
intentionally exposes no pilot or formal scientific-run command.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

from run_adaptation_cost_pilot import (
    Action,
    binary_kl,
    exact_markov_terminal_mse,
    identification_threshold,
)
from theory_audit_t017 import budget_risk_coefficient


SCHEMA_VERSION = 1
PREREG_PARENT = "41bd4c696b49c9876cff537d3fb03c571393a7b2"
DELTA = 0.025
GAMMA = 0.05
THETA_LOW = 0.05
THETA_HIGHS = (0.5, 2.0)
MAXIMUM_AGENTS = (8, 16, 32)
MIXING_VALUES = (0.2, 0.7, 0.94)
DELAYS = (0, 4, 12)
OVERHEADS = (4, 16)
SAFETY_SLACKS = (0.10, 0.20)
Q_CATALOGUE = (2, 4, 8, 16, 32)
B_CATALOGUE = (1, 2, 4, 8)
ETA_CATALOGUE = (1.0,)
PILOT_SEEDS = tuple(range(20300101, 20300165))
FORMAL_SEED_COUNT = 128
G_MIN_RELATIVE = 0.03
MIN_HIGH_GAIN_RELATIVE = 0.005
MAX_THRESHOLD_SCALE = 1000000
BUDGET_POINTS = (
    ("half_bn", "below_bn", 0.5, "floor", "B_N"),
    ("near_bn", "below_bn", 0.9, "floor", "B_N"),
    ("gray_mid", "gray_zone", 0.5, "midpoint", "bracket"),
    ("above_bs", "above_bs", 1.1, "ceil", "B_S"),
    ("double_bs", "above_bs", 2.0, "ceil", "B_S"),
)
POLICIES = (
    "oracle",
    "always_all",
    "fixed_small_q",
    "exp015a_paid_etc_frozen",
    "learning_aware",
    "information_only",
    "no_delay_ablation",
    "ignore_message_budget",
    "ignore_environment_budget",
    "no_mixing_correction",
)


@dataclass(frozen=True)
class BudgetRay:
    name: str
    beta_message: float
    beta_environment: float


@dataclass(frozen=True)
class Probe:
    q: int
    b: int
    eta: float
    n_necessary: int
    n_sufficient: int


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def action_catalogue(maximum_agents: int) -> tuple[Action, ...]:
    return tuple(
        Action(q=q, b=b)
        for q in Q_CATALOGUE
        if q <= maximum_agents
        for b in B_CATALOGUE
    )


def budget_ray(name: str, overhead: int, maximum_agents: int) -> BudgetRay:
    if name == "balanced":
        return BudgetRay(name, overhead + maximum_agents, 2.0)
    if name == "message_limited":
        return BudgetRay(name, overhead + 4, 8.0)
    if name == "environment_limited":
        return BudgetRay(name, overhead + maximum_agents, 1.5)
    raise ValueError(f"unknown budget ray: {name}")


@lru_cache(maxsize=None)
def probe_catalogue(
    theta_high: float, mixing: float, overhead: int, maximum_agents: int
) -> tuple[Probe, ...]:
    del overhead
    probes = []
    for action in action_catalogue(maximum_agents):
        if action.q < 2:
            continue
        sufficient, necessary = identification_threshold(
            THETA_LOW,
            theta_high,
            action.q,
            action.b,
            mixing,
            delta=DELTA,
        )
        probes.append(Probe(action.q, action.b, ETA_CATALOGUE[0], necessary, sufficient))
    return tuple(probes)


def scale_cost(samples: int, action: Action, overhead: int, delay: int, ray: BudgetRay) -> float:
    return max(
        samples * (overhead + action.q) / ray.beta_message,
        (samples * action.b + delay) / ray.beta_environment,
    )


def necessary_threshold(probes: Sequence[Probe], overhead: int, delay: int, ray: BudgetRay) -> tuple[float, Probe]:
    scored = [
        (scale_cost(p.n_necessary, Action(p.q, p.b), overhead, delay, ray), p)
        for p in probes
    ]
    return min(scored, key=lambda item: (item[0], item[1].q, item[1].b))


def asymptotic_action(
    theta: float,
    mixing: float,
    overhead: int,
    ray: BudgetRay,
    maximum_agents: int,
) -> tuple[Action, float]:
    scored = [
        (
            budget_risk_coefficient(
                theta,
                mixing,
                ray.beta_message,
                ray.beta_environment,
                overhead,
                action.q,
                action.b,
            ),
            action,
        )
        for action in action_catalogue(maximum_agents)
    ]
    coefficient, action = min(scored, key=lambda item: (item[0], item[1].q, item[1].b))
    return action, float(coefficient)


def budgets(scale: int, ray: BudgetRay) -> tuple[int, int]:
    if scale < 0:
        raise ValueError("scale must be nonnegative")
    return int(math.floor(scale * ray.beta_message)), int(math.floor(scale * ray.beta_environment))


def _risk_after_probe(
    theta: float,
    mixing: float,
    action: Action,
    scale: int,
    ray: BudgetRay,
    overhead: int,
    delay: int,
    probe: Probe,
) -> float:
    message_budget, environment_budget = budgets(scale, ray)
    remaining_message = message_budget - probe.n_sufficient * (overhead + probe.q)
    remaining_environment = environment_budget - probe.n_sufficient * probe.b
    if min(remaining_message, remaining_environment) < 0:
        return float("inf")
    return exact_markov_terminal_mse(
        theta,
        mixing,
        action,
        remaining_message,
        remaining_environment,
        overhead,
        delay,
    )


def qualification_margin(
    scale: int,
    probe: Probe,
    theta_high: float,
    mixing: float,
    overhead: int,
    delay: int,
    maximum_agents: int,
    ray: BudgetRay,
    epsilon_safe: float,
) -> tuple[bool, Mapping[str, float]]:
    message_budget, environment_budget = budgets(scale, ray)
    baseline = Action(maximum_agents, 1)
    high_action, _ = asymptotic_action(theta_high, mixing, overhead, ray, maximum_agents)
    baseline_low = exact_markov_terminal_mse(
        THETA_LOW, mixing, baseline, message_budget, environment_budget, overhead, delay
    )
    baseline_high = exact_markov_terminal_mse(
        theta_high, mixing, baseline, message_budget, environment_budget, overhead, delay
    )
    low_correct = _risk_after_probe(
        THETA_LOW, mixing, baseline, scale, ray, overhead, delay, probe
    )
    low_wrong = _risk_after_probe(
        THETA_LOW, mixing, high_action, scale, ray, overhead, delay, probe
    )
    high_correct = _risk_after_probe(
        theta_high, mixing, high_action, scale, ray, overhead, delay, probe
    )
    high_wrong = _risk_after_probe(
        theta_high, mixing, baseline, scale, ray, overhead, delay, probe
    )
    values = (baseline_low, baseline_high, low_correct, low_wrong, high_correct, high_wrong)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        return False, {"safety_relative": float("inf"), "high_gain_relative": float("-inf")}
    low_upper = (1.0 - DELTA) * low_correct + DELTA * low_wrong
    high_upper = (1.0 - DELTA) * high_correct + DELTA * high_wrong
    safety_relative = max(0.0, low_upper / baseline_low - 1.0)
    high_gain_relative = (baseline_high - high_upper) / baseline_high
    return (
        safety_relative <= epsilon_safe and high_gain_relative >= MIN_HIGH_GAIN_RELATIVE,
        {
            "safety_relative": float(safety_relative),
            "high_gain_relative": float(high_gain_relative),
        },
    )


def sufficient_threshold(
    probes: Sequence[Probe],
    bn: float,
    theta_high: float,
    mixing: float,
    overhead: int,
    delay: int,
    maximum_agents: int,
    ray: BudgetRay,
    epsilon_safe: float,
) -> tuple[int, Probe, Mapping[str, float]]:
    best = None
    start = max(1, int(math.ceil(bn)))
    for probe in probes:
        scale = max(start, int(math.ceil(scale_cost(probe.n_sufficient, Action(probe.q, probe.b), overhead, delay, ray))))
        lower = scale - 1
        while scale <= MAX_THRESHOLD_SCALE:
            qualified, margins = qualification_margin(
                scale, probe, theta_high, mixing, overhead, delay,
                maximum_agents, ray, epsilon_safe,
            )
            if qualified:
                upper = scale
                while upper - lower > 1:
                    midpoint = (lower + upper) // 2
                    midpoint_ok, _ = qualification_margin(
                        midpoint, probe, theta_high, mixing, overhead, delay,
                        maximum_agents, ray, epsilon_safe,
                    )
                    if midpoint_ok:
                        upper = midpoint
                    else:
                        lower = midpoint
                _, margins = qualification_margin(
                    upper, probe, theta_high, mixing, overhead, delay,
                    maximum_agents, ray, epsilon_safe,
                )
                candidate = (upper, probe.q, probe.b, probe, margins)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
                break
            lower = scale
            scale = max(scale + 1, scale * 2)
    if best is None:
        raise RuntimeError(
            "B_S unavailable within frozen search limit for "
            f"theta_high={theta_high}, lambda={mixing}, h={overhead}, "
            f"D={delay}, Q={maximum_agents}, ray={ray.name}, "
            f"epsilon={epsilon_safe}"
        )
    return int(best[0]), best[3], best[4]


def oracle_threshold(
    probes: Sequence[Probe],
    bn: float,
    bs: int,
    theta_high: float,
    mixing: float,
    overhead: int,
    delay: int,
    maximum_agents: int,
    ray: BudgetRay,
) -> tuple[int, Probe]:
    """First known-instance scale amortizing a sufficient probe and delay."""

    baseline = Action(maximum_agents, 1)
    high_action, _ = asymptotic_action(
        theta_high, mixing, overhead, ray, maximum_agents
    )
    for scale in range(max(1, math.ceil(bn)), bs + 1):
        message_budget, environment_budget = budgets(scale, ray)
        baseline_risk = exact_markov_terminal_mse(
            theta_high,
            mixing,
            baseline,
            message_budget,
            environment_budget,
            overhead,
            delay,
        )
        for probe in probes:
            correct_risk = _risk_after_probe(
                theta_high,
                mixing,
                high_action,
                scale,
                ray,
                overhead,
                delay,
                probe,
            )
            if correct_risk < baseline_risk:
                return scale, probe
    raise RuntimeError("B_oracle must not exceed B_S on a qualified scenario")


def rounded_budget_scale(name: str, bn: float, bs: int) -> int:
    if name == "half_bn":
        return max(0, math.floor(0.5 * bn))
    if name == "near_bn":
        return max(0, math.floor(0.9 * bn))
    if name == "gray_mid":
        return max(math.ceil(bn), math.floor((math.ceil(bn) + bs) / 2))
    if name == "above_bs":
        return math.ceil(1.1 * bs)
    if name == "double_bs":
        return math.ceil(2.0 * bs)
    raise ValueError(name)


def threshold_region(scale: int, bn: float, bs: int) -> str:
    if scale < bn:
        return "below_bn"
    if scale >= bs:
        return "above_bs"
    return "gray_zone"


def controller_decision(
    scale: int,
    bn: float,
    bs: int,
    observed_log_likelihood_ratio: float | None,
) -> str:
    """Public learning-aware decision; no hidden instance is an input."""

    region = threshold_region(scale, bn, bs)
    if region != "above_bs":
        return "fallback"
    if observed_log_likelihood_ratio is None:
        return "probe"
    return "commit_high" if observed_log_likelihood_ratio > 0.0 else "commit_low"


def information_only_score(
    theta0: float,
    theta1: float,
    mixing: float,
    q: int,
    b: int,
    overhead: int,
    ray: BudgetRay,
) -> float:
    """Identification-only score; deliberately has no downstream-risk input."""

    sufficient, necessary = identification_threshold(theta0, theta1, q, b, mixing, DELTA)
    del sufficient
    information = binary_kl(1.0 - DELTA, DELTA) / necessary
    charged_cost = max((overhead + q) / ray.beta_message, b / ray.beta_environment)
    return float(information / charged_cost)


def crn_stream_key(
    pilot_seed: int,
    scenario_id: str,
    regime: str,
    physical_time: int,
    agent: int,
) -> int:
    """Policy-independent counter key for potential observations."""

    payload = f"{pilot_seed}|{scenario_id}|{regime}|{physical_time}|{agent}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def pilot_seed_registry() -> tuple[int, ...]:
    """Pilot path exposes only pilot seeds; no formal registry is loaded."""

    return PILOT_SEEDS


def _scenario_factors() -> Iterable[tuple[int, float, float, int, int, str, float]]:
    ray_names = ("balanced", "message_limited", "environment_limited")
    index = 0
    for maximum_agents in MAXIMUM_AGENTS:
        for theta_high in THETA_HIGHS:
            for mixing in MIXING_VALUES:
                for delay in DELAYS:
                    overhead = OVERHEADS[index % len(OVERHEADS)]
                    ray_name = ray_names[index % len(ray_names)]
                    epsilon_safe = SAFETY_SLACKS[index % len(SAFETY_SLACKS)]
                    yield maximum_agents, theta_high, mixing, delay, overhead, ray_name, epsilon_safe
                    index += 1


def build_positive_scenarios() -> list[dict[str, object]]:
    scenarios = []
    for index, factors in enumerate(_scenario_factors()):
        maximum_agents, theta_high, mixing, delay, overhead, ray_name, epsilon_safe = factors
        ray = budget_ray(ray_name, overhead, maximum_agents)
        probes = probe_catalogue(theta_high, mixing, overhead, maximum_agents)
        bn, necessary_probe = necessary_threshold(probes, overhead, delay, ray)
        bs, sufficient_probe, margins = sufficient_threshold(
            probes, bn, theta_high, mixing, overhead, delay, maximum_agents, ray, epsilon_safe
        )
        b_oracle, oracle_probe = oracle_threshold(
            probes, bn, bs, theta_high, mixing, overhead, delay,
            maximum_agents, ray,
        )
        baseline = Action(maximum_agents, 1)
        high_action, high_coefficient = asymptotic_action(theta_high, mixing, overhead, ray, maximum_agents)
        baseline_coefficient = budget_risk_coefficient(
            theta_high, mixing, ray.beta_message, ray.beta_environment,
            overhead, baseline.q, baseline.b,
        )
        oracle_gap_relative = (baseline_coefficient - high_coefficient) / baseline_coefficient
        if oracle_gap_relative < G_MIN_RELATIVE:
            raise RuntimeError("generated scenario violates frozen oracle-gap separation")
        scenario_id = f"p{index:03d}-Q{maximum_agents}-th{theta_high:g}-l{mixing:g}-D{delay}-h{overhead}-{ray_name}-e{epsilon_safe:g}"
        points = []
        for point_name, _, _, rounding, reference in BUDGET_POINTS:
            scale = rounded_budget_scale(point_name, bn, bs)
            message_budget, environment_budget = budgets(scale, ray)
            points.append({
                "name": point_name,
                "scale": scale,
                "region": threshold_region(scale, bn, bs),
                "rounding": rounding,
                "reference": reference,
                "message_budget": message_budget,
                "environment_budget": environment_budget,
            })
        scenarios.append({
            "scenario_id": scenario_id,
            "theorem_scope": True,
            "theta_low": THETA_LOW,
            "theta_high": theta_high,
            "theta_gap": theta_high - THETA_LOW,
            "lambda": mixing,
            "gamma": GAMMA,
            "delay": delay,
            "overhead": overhead,
            "maximum_agents": maximum_agents,
            "budget_ray": asdict(ray),
            "epsilon_safe": epsilon_safe,
            "delta": DELTA,
            "B_N_raw": bn,
            "B_N_ceil": math.ceil(bn),
            "B_oracle": b_oracle,
            "B_S": bs,
            "necessary_probe": asdict(necessary_probe),
            "oracle_probe": asdict(oracle_probe),
            "sufficient_probe": asdict(sufficient_probe),
            "high_oracle_action": {"q": high_action.q, "b": high_action.b, "eta": ETA_CATALOGUE[0]},
            "baseline_action": {"q": baseline.q, "b": baseline.b, "eta": ETA_CATALOGUE[0]},
            "oracle_gap_relative": oracle_gap_relative,
            "qualification_margins_at_B_S": dict(margins),
            "budget_points": points,
            "transition_sides": {"low": "baseline_side", "high": "adaptive_side"},
        })
    return scenarios


def build_negative_controls() -> list[dict[str, object]]:
    return [
        {
            "control_id": "nc-mixing-boundary",
            "theorem_scope": False,
            "positive_gate_eligible": False,
            "lambda": 0.9999,
            "theta_low": 0.05,
            "theta_high": 2.0,
            "delay": 12,
            "overhead": 16,
            "maximum_agents": 16,
            "budget_ray": asdict(budget_ray("balanced", 16, 16)),
            "frozen_scales": [10, 20, 50, 100, 200],
            "purpose": "illustrate the AC-8 near-nonmixing failure trend; never empirical proof",
        },
        {
            "control_id": "nc-oracle-gap",
            "theorem_scope": False,
            "positive_gate_eligible": False,
            "oracle_gap_target": 0.001,
            "theta_low": 0.20,
            "theta_high": 0.205,
            "lambda": 0.7,
            "delay": 4,
            "overhead": 4,
            "maximum_agents": 16,
            "budget_ray": asdict(budget_ray("balanced", 4, 16)),
            "frozen_scales": [10, 20, 50, 100, 200],
            "purpose": "illustrate AC-9 threshold-ratio degeneration; never a positive matching cell",
        },
    ]


def build_manifest() -> dict[str, object]:
    scenarios = build_positive_scenarios()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "EXP-016A",
        "prereg_parent": PREREG_PARENT,
        "scientific_outcomes_present": False,
        "positive_scenarios": scenarios,
        "negative_controls": build_negative_controls(),
        "policies": list(POLICIES),
        "budget_rounding": {
            "below_B_N": "floor",
            "above_B_S": "ceil",
            "gray_zone": "floor midpoint of ceil(B_N) and B_S",
            "budget_from_scale": "floor(scale * beta) independently for message and environment",
        },
    }
    payload["configuration_sha256"] = config_hash(payload)
    return payload


def workload_estimate(manifest: Mapping[str, object]) -> dict[str, object]:
    positive = len(manifest["positive_scenarios"])
    negative = len(manifest["negative_controls"])
    theorem_cells = positive * 2 * len(BUDGET_POINTS)
    negative_cells = negative * len(BUDGET_POINTS)
    cells = theorem_cells + negative_cells
    trajectories = cells * len(POLICIES) * len(PILOT_SEEDS)
    rows = trajectories
    # Conservative static accounting, not a timed scientific benchmark.
    seconds_per_trajectory = 0.012
    bytes_per_row = 1400
    maximum_updates = max(
        point["environment_budget"]
        for scenario in manifest["positive_scenarios"]
        for point in scenario["budget_points"]
    )
    memory_gb = max(1.5, (maximum_updates * max(MAXIMUM_AGENTS) * 8 * 12) / 1e9)
    return {
        "positive_base_scenarios": positive,
        "negative_control_families": negative,
        "expanded_cells": cells,
        "policy_count": len(POLICIES),
        "pilot_seed_count": len(PILOT_SEEDS),
        "estimated_trajectories": trajectories,
        "estimated_rows": rows,
        "estimated_cpu_wall_hours_single_process": trajectories * seconds_per_trajectory / 3600.0,
        "estimated_peak_memory_gb": memory_gb,
        "estimated_disk_gb": rows * bytes_per_row / 1e9,
        "estimate_basis": "static operation/row accounting; no scientific trajectory was executed",
    }


def validate_manifest(manifest: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    if manifest.get("scientific_outcomes_present") is not False:
        errors.append("manifest must not contain scientific outcomes")
    scenarios = manifest.get("positive_scenarios", [])
    if len(scenarios) != 54:
        errors.append("expected exactly 54 positive base scenarios")
    for scenario in scenarios:
        bn, bs = scenario["B_N_raw"], scenario["B_S"]
        if not 0 < bn <= bs:
            errors.append(f"invalid threshold order: {scenario['scenario_id']}")
        if not bn <= scenario["B_oracle"] <= bs:
            errors.append(f"invalid oracle threshold: {scenario['scenario_id']}")
        if scenario["lambda"] > 1.0 - scenario["gamma"]:
            errors.append(f"mixing outside theorem scope: {scenario['scenario_id']}")
        if scenario["oracle_gap_relative"] < G_MIN_RELATIVE:
            errors.append(f"oracle gap not separated: {scenario['scenario_id']}")
        for point in scenario["budget_points"]:
            if point["region"] != threshold_region(point["scale"], bn, bs):
                errors.append(f"wrong region label: {scenario['scenario_id']}/{point['name']}")
    for control in manifest.get("negative_controls", []):
        if control.get("theorem_scope") or control.get("positive_gate_eligible"):
            errors.append(f"negative control leaks into positive gate: {control['control_id']}")
    signature = inspect.signature(controller_decision).parameters
    for forbidden in ("theta_true", "lambda_true", "regime", "oracle_action"):
        if forbidden in signature:
            errors.append(f"hidden controller input: {forbidden}")
    return errors


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_frozen_manifest() -> Mapping[str, object]:
    path = repository_root() / "docs" / "exp016a_scenario_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("validate", "estimate", "emit", "freeze-manifest"),
        nargs="?",
        default="validate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = build_manifest()
    if args.command == "emit":
        print(json.dumps(generated, indent=2, sort_keys=True))
        return
    if args.command == "freeze-manifest":
        target = repository_root() / "docs" / "exp016a_scenario_manifest.json"
        target.write_text(
            json.dumps(generated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"frozen": str(target), "scientific_outcomes_generated": False}))
        return
    frozen = load_frozen_manifest()
    errors = validate_manifest(frozen)
    if canonical_json(generated) != canonical_json(frozen):
        errors.append("frozen manifest differs from deterministic generator")
    if errors:
        raise SystemExit("\n".join(errors))
    result = {
        "status": "valid",
        "configuration_sha256": frozen["configuration_sha256"],
        "scientific_outcomes_generated": False,
    }
    if args.command == "estimate":
        result["workload"] = workload_estimate(frozen)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
