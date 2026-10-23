"""Pure fixed-design same-stream Gaussian gate; no observations or true regime input.

Hypotheses, directional-error validity and frozen weights are caller contracts.
This module does not estimate unknown parameters or certify adaptive stopping.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class InstanceRisk:
    correct_J: float
    wrong_J: float
    correct_d: float
    wrong_d: float
    correct_k: float
    wrong_k: float
    posterior_mean_variance: float
    directional_error_upper: float
    fallback_J: float


def risk_upper(instance):
    vals = tuple(vars(instance).values())
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("Non-finite public certificate input")
    if min(instance.correct_J, instance.wrong_J, instance.fallback_J,
           instance.posterior_mean_variance) < 0:
        raise ValueError("Negative risk/variance")
    delta = instance.directional_error_upper
    if not 0 <= delta <= 1:
        raise ValueError("Invalid directional probability bound")
    s2 = instance.posterior_mean_variance
    gamma2 = (4*(instance.wrong_d*instance.wrong_k-instance.correct_d*instance.correct_k)**2*s2
              + 2*(instance.wrong_k**2-instance.correct_k**2)**2*s2*s2)
    carry = math.sqrt(delta*gamma2)
    upper = instance.correct_J+delta*max(0, instance.wrong_J-instance.correct_J)+carry
    return upper, carry


def admit(low, high, epsilon, feasible):
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Invalid public safety allowance")
    U0, _ = risk_upper(low)
    U1, _ = risk_upper(high)
    return bool(feasible and U1 < high.fallback_J and U0 <= low.fallback_J+epsilon)
