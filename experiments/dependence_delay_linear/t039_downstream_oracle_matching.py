"""Certified first-order downstream-risk matching on a separated class."""

from __future__ import annotations

import math


def logarithmic_probe_count(
    budget_scale: float, information_exponent: float, error_power: float = 3.0
) -> int:
    if budget_scale <= 1.0 or information_exponent <= 0.0 or error_power <= 0.0:
        raise ValueError("budget must exceed one and constants must be positive")
    return int(math.ceil(error_power * math.log(budget_scale) / information_exponent))


def oracle_matching_bound(
    *,
    budget_scale: float,
    information_exponent: float,
    probe_cost_per_sample: float,
    delay_cost: float,
    oracle_risk_lower_coefficient: float,
    oracle_budget_sensitivity: float,
    wrong_commit_coefficient: float,
    error_power: float = 3.0,
) -> dict[str, float | int | bool]:
    """Evaluate the explicit bound in the T-039 theorem."""

    values = (
        probe_cost_per_sample,
        delay_cost,
        oracle_risk_lower_coefficient,
        oracle_budget_sensitivity,
        wrong_commit_coefficient,
    )
    if budget_scale <= 1.0 or information_exponent <= 0.0:
        raise ValueError("invalid budget or information exponent")
    if any(value < 0.0 for value in values) or oracle_risk_lower_coefficient == 0.0:
        raise ValueError("costs must be nonnegative and oracle lower coefficient positive")
    probes = logarithmic_probe_count(
        budget_scale, information_exponent, error_power
    )
    charged_scale = probe_cost_per_sample * probes + delay_cost
    feasible = charged_scale <= budget_scale / 2.0
    error_probability = 0.5 * math.exp(-information_exponent * probes)
    if not feasible:
        return {
            "probe_count": probes,
            "charged_scale": charged_scale,
            "feasible": False,
            "error_probability": error_probability,
            "absolute_excess_bound": math.inf,
            "relative_excess_bound": math.inf,
        }
    absolute = (
        oracle_budget_sensitivity * charged_scale / budget_scale**2
        + error_probability
        * wrong_commit_coefficient
        / (budget_scale - charged_scale)
    )
    relative = absolute / (oracle_risk_lower_coefficient / budget_scale)
    return {
        "probe_count": probes,
        "charged_scale": charged_scale,
        "feasible": True,
        "error_probability": error_probability,
        "absolute_excess_bound": absolute,
        "relative_excess_bound": relative,
    }
