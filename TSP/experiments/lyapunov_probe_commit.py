"""Low-complexity probe-then-commit participation controller.

The controller uses fully charged, non-learning rollout blocks to construct an
observable cross-worker dependence certificate.  It then minimizes the
message-limited Lyapunov variance-cost factor over a finite participation set.
The module contains no benchmark-return input and is independent of HARL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CorrelationCertificate:
    estimate: float
    upper: float
    standard_error: float
    samples: int
    workers: int
    confidence: float


@dataclass(frozen=True)
class ParticipationDecision:
    selected_q: int
    certificate: CorrelationCertificate
    scores: dict[int, float]
    effective_overhead: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result["scores"] = {str(q): score for q, score in self.scores.items()}
        return result


@dataclass(frozen=True)
class ControllerAccounting:
    probe_messages: int
    probe_environment_ticks: int
    training_updates: int
    training_messages: int
    training_environment_ticks: int
    total_messages: int
    total_environment_ticks: int

    def to_dict(self) -> dict:
        return asdict(self)


def average_pairwise_correlation_certificate(
    fingerprints: np.ndarray, delta: float = 0.05
) -> CorrelationCertificate:
    """Estimate an equicorrelation proxy and a one-sided normal upper bound.

    Rows are separated trajectory fingerprints and columns are rollout workers.
    The standard error is computed from rowwise average cross-products after
    column standardization.  A zero-variance column returns the conservative
    upper bound one instead of creating artificial evidence for independence.
    """

    values = np.asarray(fingerprints, dtype=float)
    if values.ndim != 2:
        raise ValueError("fingerprints must be a samples-by-workers matrix")
    samples, workers = values.shape
    if samples < 4 or workers < 2:
        raise ValueError("at least four fingerprints and two workers are required")
    if not 0.0 < delta < 0.5:
        raise ValueError("delta must lie in (0, 0.5)")
    if not np.isfinite(values).all():
        raise ValueError("fingerprints must be finite")

    centered = values - values.mean(axis=0, keepdims=True)
    scales = centered.std(axis=0, ddof=1)
    if np.any(scales <= np.finfo(float).eps):
        return CorrelationCertificate(
            estimate=1.0,
            upper=1.0,
            standard_error=float("inf"),
            samples=samples,
            workers=workers,
            confidence=1.0 - delta,
        )

    standardized = centered / scales
    row_cross_products = (
        standardized.sum(axis=1) ** 2 - np.square(standardized).sum(axis=1)
    ) / (workers * (workers - 1))
    # ddof=1 column scaling makes the average row cross-product smaller than
    # the sample correlation by (n-1)/n.
    row_cross_products *= samples / (samples - 1)
    estimate = float(row_cross_products.mean())
    standard_error = float(row_cross_products.std(ddof=1) / np.sqrt(samples))
    quantile = NormalDist().inv_cdf(1.0 - delta)
    upper = float(np.clip(estimate + quantile * standard_error, 0.0, 1.0))
    return CorrelationCertificate(
        estimate=float(np.clip(estimate, -1.0, 1.0)),
        upper=upper,
        standard_error=standard_error,
        samples=samples,
        workers=workers,
        confidence=1.0 - delta,
    )


def message_limited_lyapunov_score(
    q: int, rho_upper: float, server_overhead: int, rollout_length: int
) -> float:
    """Return the variance-cost factor induced by the quadratic drift bound."""

    if q <= 0 or rollout_length <= 0 or server_overhead < 0:
        raise ValueError("q and rollout_length must be positive; overhead nonnegative")
    if not 0.0 <= rho_upper <= 1.0:
        raise ValueError("rho_upper must lie in [0,1]")
    effective_overhead = server_overhead / rollout_length
    variance_factor = rho_upper + (1.0 - rho_upper) / q
    return float((effective_overhead + q) * variance_factor)


def choose_participation(
    fingerprints: np.ndarray,
    candidates: Iterable[int],
    server_overhead: int,
    rollout_length: int,
    delta: float = 0.05,
) -> ParticipationDecision:
    """Construct the dependence certificate and minimize its Lyapunov score."""

    catalogue = sorted(set(int(q) for q in candidates))
    if not catalogue or catalogue[0] <= 0:
        raise ValueError("candidates must contain positive integers")
    certificate = average_pairwise_correlation_certificate(fingerprints, delta)
    scores = {
        q: message_limited_lyapunov_score(
            q, certificate.upper, server_overhead, rollout_length
        )
        for q in catalogue
    }
    selected_q = min(catalogue, key=lambda q: (scores[q], q))
    return ParticipationDecision(
        selected_q=selected_q,
        certificate=certificate,
        scores=scores,
        effective_overhead=server_overhead / rollout_length,
    )


def controller_accounting(
    *,
    selected_q: int,
    probe_q: int,
    probe_blocks: int,
    rollout_length: int,
    message_budget: int,
    environment_budget: int,
    server_overhead: int,
) -> ControllerAccounting:
    """Charge every probe and learning block against both physical budgets."""

    if min(selected_q, probe_q, probe_blocks, rollout_length) <= 0:
        raise ValueError("participation, probe blocks, and rollout length must be positive")
    if min(message_budget, environment_budget) <= 0 or server_overhead < 0:
        raise ValueError("budgets must be positive and overhead nonnegative")
    probe_messages = probe_blocks * (server_overhead + probe_q * rollout_length)
    probe_environment_ticks = probe_blocks * rollout_length
    remaining_messages = message_budget - probe_messages
    remaining_environment = environment_budget - probe_environment_ticks
    if remaining_messages <= 0 or remaining_environment <= 0:
        raise ValueError("probe exhausts a registered budget")
    message_cost = server_overhead + selected_q * rollout_length
    environment_cost = rollout_length
    training_updates = min(
        remaining_messages // message_cost,
        remaining_environment // environment_cost,
    )
    if training_updates <= 0:
        raise ValueError("no learning update remains after the probe")
    training_messages = training_updates * message_cost
    training_environment_ticks = training_updates * environment_cost
    return ControllerAccounting(
        probe_messages=probe_messages,
        probe_environment_ticks=probe_environment_ticks,
        training_updates=training_updates,
        training_messages=training_messages,
        training_environment_ticks=training_environment_ticks,
        total_messages=probe_messages + training_messages,
        total_environment_ticks=probe_environment_ticks
        + training_environment_ticks,
    )
