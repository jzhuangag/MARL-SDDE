"""Exact finite-budget risk certificate for fingerprint classify-and-commit."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

from experiments.dependence_delay_linear.t047_scheduled_participation import (
    AffineRisk,
)
from experiments.dependence_delay_linear.t052_exact_binomial_probe import (
    exact_plugin_action_distribution,
)


@dataclass(frozen=True)
class ResourceUse:
    message: int
    environment: int
    delay_reserve: int


def feasible_fixed_horizon(
    *,
    message_budget: int,
    environment_budget: int,
    overhead: int,
    participation: int,
    probe_message: int = 0,
    probe_environment: int = 0,
    delay: int = 0,
) -> tuple[int, ResourceUse]:
    """Largest update horizon satisfying both charged budgets and delay."""

    values = (
        message_budget,
        environment_budget,
        overhead,
        participation,
        probe_message,
        probe_environment,
        delay,
    )
    if any(int(value) != value or value < 0 for value in values):
        raise ValueError("budgets, costs, participation, and delay must be integers")
    if participation < 1 or overhead + participation < 1:
        raise ValueError("participation must be positive")
    message_remaining = message_budget - probe_message
    environment_remaining = environment_budget - probe_environment - delay
    if message_remaining < 0 or environment_remaining < 0:
        raise ValueError("probe and delay reserve exceed the available budgets")
    updates = min(
        message_remaining // (overhead + participation), environment_remaining
    )
    use = ResourceUse(
        message=probe_message + updates * (overhead + participation),
        environment=probe_environment + updates + delay,
        delay_reserve=delay,
    )
    return int(updates), use


def evaluate_affine_risk(risk: AffineRisk, rho: float) -> float:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    value = risk.evaluate(rho)
    if not math.isfinite(value) or value < -1e-12:
        raise ValueError("finite-horizon risk must be finite and nonnegative")
    return max(0.0, float(value))


def exact_finite_plugin_certificate(
    *,
    rho: float,
    candidates: Iterable[int],
    overhead: float,
    probe_blocks: int,
    collision_probability: float,
    post_probe_risks: Mapping[int, AffineRisk],
    full_budget_baseline_q: int,
    full_budget_baseline_risk: AffineRisk,
) -> dict[str, object]:
    """Condition exactly on the independent Binomial fingerprint count.

    Each entry of ``post_probe_risks`` must already use the candidate-specific
    horizon after charging both probe resources and delay. The baseline risk
    must use its full no-probe horizon.
    """

    actions = sorted({int(q) for q in candidates})
    if set(actions) != set(post_probe_risks):
        raise ValueError("one post-probe finite risk is required per candidate")
    distribution = exact_plugin_action_distribution(
        rho=rho,
        candidates=actions,
        overhead=overhead,
        blocks=probe_blocks,
        collision_probability=collision_probability,
    )
    finite_risks = {
        q: evaluate_affine_risk(post_probe_risks[q], rho) for q in actions
    }
    expected = float(sum(distribution[q] * finite_risks[q] for q in actions))
    baseline = evaluate_affine_risk(full_budget_baseline_risk, rho)
    if baseline <= 0.0:
        raise ValueError("baseline risk must be positive for a ratio certificate")
    oracle_q = min(actions, key=lambda q: (finite_risks[q], q))
    oracle_risk = finite_risks[oracle_q]
    controller_to_oracle = (
        1.0 if expected == 0.0 and oracle_risk == 0.0
        else math.inf if oracle_risk == 0.0
        else expected / oracle_risk
    )
    return {
        "rho": float(rho),
        "action_distribution": distribution,
        "post_probe_finite_risks": finite_risks,
        "finite_oracle_q": int(oracle_q),
        "finite_oracle_risk": oracle_risk,
        "expected_controller_risk": expected,
        "full_budget_baseline_q": int(full_budget_baseline_q),
        "full_budget_baseline_risk": baseline,
        "expected_controller_to_baseline_ratio": expected / baseline,
        "expected_controller_to_oracle_ratio": controller_to_oracle,
        "expected_baseline_improvement": 1.0 - expected / baseline,
    }


def finite_no_harm_certificate(
    *, expected_controller_risk: float, baseline_risk: float, tolerance: float
) -> dict[str, float | bool]:
    if expected_controller_risk < 0.0 or baseline_risk <= 0.0 or tolerance < 0.0:
        raise ValueError("invalid risks or tolerance")
    ratio = expected_controller_risk / baseline_risk
    return {
        "ratio": ratio,
        "tolerance": tolerance,
        "certified": ratio <= 1.0 + tolerance,
        "margin": 1.0 + tolerance - ratio,
    }
