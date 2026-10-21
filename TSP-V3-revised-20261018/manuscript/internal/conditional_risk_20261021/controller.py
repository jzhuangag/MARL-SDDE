"""Research prototype: deterministic scores only; no sampling or native runner.

Fixed centered binary Gaussian model and public initial error 1 are required.
This is not admitted into either completed scientific workflow.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Posterior:
    mean: float
    variance: float
    explained_variance: float
    log_density: float


def filter_probe(values, theta, lam, q):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("A finite, nonempty, one-dimensional probe is required")
    if not theta > 0 or not 0 <= lam < 1 or not q >= 1:
        raise ValueError("Invalid public Gaussian model")
    mean, variance, log_density = 0., float(theta), 0.
    for i, value in enumerate(values):
        innovation_variance = variance + 1 / q
        innovation = value - mean
        log_density -= .5 * (math.log(2 * math.pi * innovation_variance)
                             + innovation * innovation / innovation_variance)
        gain = variance / innovation_variance
        mean += gain * innovation
        variance *= 1 - gain
        if i + 1 < len(values):
            mean *= lam
            variance = lam * lam * variance + (1 - lam * lam) * theta
    return Posterior(float(mean), float(variance), float(theta - variance),
                     float(log_density))


def conditional_loss(program, posterior, mode="full"):
    # program=(J,d,k), with J charged at the post-probe remaining horizon.
    J, d, k = program
    if mode == "marginal":
        return float(J)
    if mode not in ("full", "even"):
        raise ValueError(mode)
    value = J + k * k * (posterior.mean ** 2 - posterior.explained_variance)
    if mode == "full":
        value += 2 * d * k * posterior.mean
    if value < -1e-10:
        raise ArithmeticError("Invalid negative Gaussian conditional risk")
    return float(value)


def choose(values, family, programs_by_regime, prior=.5, mode="full"):
    """Return a catalogue index; accepts no realized regime, seed or future stream.

    Both tuples of programs must be precomputed from public hypothetical models
    and EXACT remaining budgets. Catalogue ordering must match in both regimes.
    This routine does not decide whether to pay for a probe before observing it.
    """
    if not 0 < prior < 1:
        raise ValueError("An explicit interior prior is required")
    if len(programs_by_regime) != 2 or not programs_by_regime[0] or (
            len(programs_by_regime[0]) != len(programs_by_regime[1])):
        raise ValueError("Two equally indexed nonempty catalogues are required")
    post = tuple(filter_probe(values, theta, family.lam, family.Q)
                 for theta in (family.theta_low, family.theta_high))
    logs = np.array([math.log1p(-prior) + post[0].log_density,
                     math.log(prior) + post[1].log_density])
    probabilities = np.exp(logs - np.max(logs))
    probabilities /= probabilities.sum()
    scores = [sum(probabilities[j] * conditional_loss(programs_by_regime[j][a],
                                                     post[j], mode)
                  for j in (0, 1)) for a in range(len(programs_by_regime[0]))]
    if not np.all(np.isfinite(scores)):
        raise ArithmeticError("Nonfinite conditional scores")
    return min(range(len(scores)), key=lambda a: (scores[a], a)), scores, probabilities
