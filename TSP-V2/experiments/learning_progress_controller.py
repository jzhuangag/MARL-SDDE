"""Finite-catalogue Lyapunov learning-progress selector.

This module is benchmark independent.  It consumes already constructed
simultaneous upper certificates for the affine drift coefficients and exact
post-calibration resource horizons.  Estimation of those certificates is a
separate, domain-specific layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DriftCertificate:
    a_upper: float
    c_upper: float
    confidence: float
    samples: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.a_upper < 1.0:
            raise ValueError("a_upper must lie in [0,1)")
        if self.c_upper < 0.0:
            raise ValueError("c_upper must be nonnegative")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0,1)")
        if self.samples <= 0:
            raise ValueError("samples must be positive")


@dataclass(frozen=True)
class ResourceBudget:
    message_remaining: int
    environment_remaining: int

    def __post_init__(self) -> None:
        if min(self.message_remaining, self.environment_remaining) <= 0:
            raise ValueError("remaining budgets must be positive")


@dataclass(frozen=True)
class ActionCost:
    messages_per_update: int
    environment_per_update: int

    def __post_init__(self) -> None:
        if min(self.messages_per_update, self.environment_per_update) <= 0:
            raise ValueError("action costs must be positive")


@dataclass(frozen=True)
class LearningProgressDecision:
    selected_q: int
    horizons: dict[int, int]
    certified_risks: dict[int, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AdditiveDriftCertificate:
    """Family-wise lower certificate for one-step Lyapunov progress."""

    progress_lower: float
    confidence: float
    samples: int
    effective_samples: float
    mixing_bias: float

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0,1)")
        if self.samples <= 0 or not 0.0 < self.effective_samples <= self.samples:
            raise ValueError("effective samples must lie in (0,samples]")
        if self.mixing_bias < 0.0:
            raise ValueError("mixing_bias must be nonnegative")


@dataclass(frozen=True)
class AdditiveDriftDecision:
    selected_q: int
    horizons: dict[int, int]
    certified_progress: dict[int, float]
    used_fallback: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MixingBlockingPlan:
    """Registered separated-sample plan for a beta-mixing trace.

    ``beta_at_stride`` is an externally proved upper bound on the beta-mixing
    coefficient at the registered stride.  The plan spends part of the
    family-wise error probability on Berbee coupling and the remainder on a
    one-sided Hoeffding bound.  It therefore cannot be replaced by an
    unverified scalar ``effective_samples`` value.
    """

    raw_samples: int
    burn_in: int
    stride: int
    selected_indices: tuple[int, ...]
    beta_at_stride: float
    coupling_failure: float
    concentration_failure: float
    radius: float
    confidence: float
    family_size: int

    @property
    def selected_samples(self) -> int:
        return len(self.selected_indices)


@dataclass(frozen=True)
class ValidationCharge:
    """Exact charge of one disjoint validation block."""

    messages: int
    environment_ticks: int

    def __post_init__(self) -> None:
        if min(self.messages, self.environment_ticks) <= 0:
            raise ValueError("validation charges must be positive")


def bounded_validation_risk(
    validation_return: np.ndarray | Sequence[float],
    *,
    center: float,
    scale: float,
) -> np.ndarray:
    """Map a disjoint validation return to a bounded Lyapunov observation.

    This transform is deterministic, monotone, and fixed before outcomes.
    It is not a certificate by itself: confidence requires the separated
    validation construction represented by :class:`MixingBlockingPlan`.
    """

    values = np.asarray(validation_return, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("validation returns must be finite")
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("center must be finite and scale must be positive")
    return 0.5 - np.arctan((values - center) / scale) / np.pi


def make_beta_mixing_blocking_plan(
    *,
    raw_samples: int,
    burn_in: int,
    stride: int,
    beta_at_stride: float,
    confidence: float,
    family_size: int,
) -> MixingBlockingPlan:
    """Create a family-wise valid separated-sample concentration plan.

    For ``m`` selected observations, Berbee coupling contributes at most
    ``(m-1) beta(stride)`` failure probability.  The remaining per-candidate
    error probability is used in the one-sided Hoeffding radius for variables
    in ``[-1,1]``.  The function rejects plans whose claimed mixing bound is
    too weak for the requested confidence instead of silently inventing an
    effective sample size.
    """

    if raw_samples <= 0 or burn_in < 0 or stride <= 0 or burn_in >= raw_samples:
        raise ValueError("invalid raw sample, burn-in, or stride configuration")
    if not 0.0 <= beta_at_stride <= 1.0:
        raise ValueError("beta_at_stride must lie in [0,1]")
    if not 0.0 < confidence < 1.0 or family_size <= 0:
        raise ValueError("confidence and family_size are invalid")
    indices = tuple(range(burn_in, raw_samples, stride))
    if len(indices) < 2:
        raise ValueError("blocking plan must retain at least two samples")
    per_candidate_failure = (1.0 - confidence) / family_size
    coupling_failure = (len(indices) - 1) * beta_at_stride
    concentration_failure = per_candidate_failure - coupling_failure
    if concentration_failure <= 0.0:
        raise ValueError(
            "mixing coupling exhausts the family-wise confidence budget"
        )
    radius = float(
        np.sqrt(2.0 * np.log(1.0 / concentration_failure) / len(indices))
    )
    return MixingBlockingPlan(
        raw_samples=raw_samples,
        burn_in=burn_in,
        stride=stride,
        selected_indices=indices,
        beta_at_stride=float(beta_at_stride),
        coupling_failure=float(coupling_failure),
        concentration_failure=float(concentration_failure),
        radius=radius,
        confidence=confidence,
        family_size=family_size,
    )


def fit_blocked_additive_drift_certificate(
    before: np.ndarray,
    after: np.ndarray,
    *,
    plan: MixingBlockingPlan,
    mean_bias: float = 0.0,
    transfer_penalty: float = 0.0,
) -> AdditiveDriftCertificate:
    """Fit a strict certificate from the samples fixed by ``plan``.

    ``mean_bias`` covers a separately proved initialization/nonstationarity
    allowance. ``transfer_penalty`` is the registered upper bound between the
    calibration drift and the post-calibration epoch (for example ``L R``
    under a proved Lipschitz drift and trust-region radius).  Both are exposed
    and subtracted; neither may be learned from the same outcomes.
    """

    before = np.asarray(before, dtype=float).reshape(-1)
    after = np.asarray(after, dtype=float).reshape(-1)
    if before.shape != after.shape or before.size != plan.raw_samples:
        raise ValueError("trace length must match the registered blocking plan")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Lyapunov observations must be finite")
    if np.any((before < 0.0) | (before > 1.0)) or np.any(
        (after < 0.0) | (after > 1.0)
    ):
        raise ValueError("Lyapunov observations must lie in [0,1]")
    if mean_bias < 0.0 or transfer_penalty < 0.0:
        raise ValueError("bias and transfer penalties must be nonnegative")
    index = np.asarray(plan.selected_indices, dtype=int)
    empirical_progress = float(np.mean(before[index] - after[index]))
    progress_lower = empirical_progress - plan.radius - mean_bias - transfer_penalty
    return AdditiveDriftCertificate(
        progress_lower=float(progress_lower),
        confidence=plan.confidence,
        samples=plan.raw_samples,
        effective_samples=float(plan.selected_samples),
        mixing_bias=float(mean_bias + transfer_penalty),
    )


def trust_region_transfer_penalty(
    *, drift_lipschitz_upper: float, policy_radius: float
) -> float:
    """Return the explicit drift-transfer remainder ``L * R``."""

    if drift_lipschitz_upper < 0.0 or policy_radius < 0.0:
        raise ValueError("Lipschitz bound and policy radius must be nonnegative")
    return float(drift_lipschitz_upper * policy_radius)


def charge_validation_block(
    *, workers: int, horizon: int, messages_per_worker_tick: int = 1
) -> ValidationCharge:
    """Charge every validation transition and worker message exactly once."""

    if min(workers, horizon, messages_per_worker_tick) <= 0:
        raise ValueError("workers, horizon, and message multiplier must be positive")
    return ValidationCharge(
        messages=workers * horizon * messages_per_worker_tick,
        environment_ticks=workers * horizon,
    )


def fit_bounded_additive_drift_certificate(
    before: np.ndarray,
    after: np.ndarray,
    *,
    confidence: float,
    family_size: int,
    effective_samples: float | None = None,
    mixing_bias: float = 0.0,
) -> AdditiveDriftCertificate:
    """Lower-certify mean progress for a bounded Lyapunov observation.

    The observations must lie in ``[0,1]``, so paired progress lies in
    ``[-1,1]``.  ``effective_samples`` and ``mixing_bias`` are explicit inputs
    supplied by the registered Markov/dependence certificate; they are never
    inferred as if the pairs were i.i.d.
    """

    before = np.asarray(before, dtype=float).reshape(-1)
    after = np.asarray(after, dtype=float).reshape(-1)
    if before.shape != after.shape or before.size < 2:
        raise ValueError("before and after must contain at least two pairs")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Lyapunov observations must be finite")
    if np.any((before < 0.0) | (before > 1.0)) or np.any(
        (after < 0.0) | (after > 1.0)
    ):
        raise ValueError("Lyapunov observations must lie in [0,1]")
    if not 0.0 < confidence < 1.0 or family_size <= 0:
        raise ValueError("confidence and family_size are invalid")
    if mixing_bias < 0.0:
        raise ValueError("mixing_bias must be nonnegative")
    n_eff = float(before.size if effective_samples is None else effective_samples)
    if not 0.0 < n_eff <= before.size:
        raise ValueError("effective_samples must lie in (0,n]")
    per_candidate_delta = (1.0 - confidence) / family_size
    radius = np.sqrt(2.0 * np.log(1.0 / per_candidate_delta) / n_eff)
    progress_lower = float(np.mean(before - after) - radius - mixing_bias)
    return AdditiveDriftCertificate(
        progress_lower=progress_lower,
        confidence=confidence,
        samples=int(before.size),
        effective_samples=n_eff,
        mixing_bias=float(mixing_bias),
    )


def select_additive_drift_action(
    *,
    budget: ResourceBudget,
    costs: Mapping[int, ActionCost],
    certificates: Mapping[int, AdditiveDriftCertificate],
    fallback_q: int,
) -> AdditiveDriftDecision:
    """Maximize certified finite-budget Lyapunov progress.

    The score is the exact feasible update horizon multiplied by the positive
    part of the simultaneous lower drift certificate.  If no action has a
    positive certificate, the registered fallback is returned.
    """

    catalogue = sorted(set(costs) & set(certificates))
    if not catalogue or set(costs) != set(certificates):
        raise ValueError("costs and certificates must share one nonempty catalogue")
    if fallback_q not in catalogue:
        raise ValueError("fallback_q must belong to the catalogue")
    horizons = {q: feasible_horizon(budget, costs[q]) for q in catalogue}
    if any(value <= 0 for value in horizons.values()):
        raise ValueError("every action must admit at least one update")
    scores = {
        q: float(horizons[q] * max(certificates[q].progress_lower, 0.0))
        for q in catalogue
    }
    if max(scores.values()) <= 0.0:
        selected_q = fallback_q
        used_fallback = True
    else:
        selected_q = min(catalogue, key=lambda q: (-scores[q], q))
        used_fallback = False
    return AdditiveDriftDecision(
        selected_q=selected_q,
        horizons=horizons,
        certified_progress=scores,
        used_fallback=used_fallback,
    )


def fit_affine_drift_certificate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    confidence: float,
    noise_scale: float,
    parameter_radius: float,
    ridge: float = 1.0,
    markov_bias: float = 0.0,
) -> DriftCertificate:
    """Fit an affine drift model with a self-normalized upper rectangle.

    The caller supplies a conditionally sub-Gaussian noise scale after any
    blocking/mixing correction and an additive Markov-bias allowance.  The
    latter expands both coordinates and is deliberately explicit rather than
    being hidden inside an i.i.d. standard error.
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("x and y must contain at least three paired observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("drift observations must be finite")
    if np.any(x < 0.0) or np.any(y < 0.0):
        raise ValueError("Lyapunov observations must be nonnegative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if min(noise_scale, parameter_radius, markov_bias) < 0.0 or ridge <= 0.0:
        raise ValueError("scales and bias must be nonnegative; ridge positive")

    design = np.column_stack([x, np.ones_like(x)])
    gram = design.T @ design + ridge * np.eye(2)
    inverse = np.linalg.inv(gram)
    estimate = inverse @ design.T @ y
    delta = 1.0 - confidence
    log_det_ratio = np.linalg.slogdet(gram)[1] - 2.0 * np.log(ridge)
    beta = noise_scale * np.sqrt(
        max(log_det_ratio + 2.0 * np.log(1.0 / delta), 0.0)
    ) + np.sqrt(ridge) * parameter_radius
    coordinate_radii = beta * np.sqrt(np.diag(inverse)) + markov_bias
    upper = estimate + coordinate_radii
    a_upper = float(upper[0])
    c_upper = float(max(upper[1], 0.0))
    if not 0.0 <= a_upper < 1.0:
        raise ValueError("calibration cannot certify a stable contraction factor")
    return DriftCertificate(
        a_upper=a_upper,
        c_upper=c_upper,
        confidence=confidence,
        samples=int(x.size),
    )


def feasible_horizon(budget: ResourceBudget, cost: ActionCost) -> int:
    """Return the exact dual-budget learning horizon."""

    return min(
        budget.message_remaining // cost.messages_per_update,
        budget.environment_remaining // cost.environment_per_update,
    )


def affine_terminal_risk(
    initial_lyapunov: float, horizon: int, certificate: DriftCertificate
) -> float:
    """Evaluate the closed-form affine-drift terminal certificate."""

    if initial_lyapunov < 0.0:
        raise ValueError("initial_lyapunov must be nonnegative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    a = certificate.a_upper
    return float(
        (a**horizon) * initial_lyapunov
        + certificate.c_upper * (1.0 - a**horizon) / (1.0 - a)
    )


def select_learning_progress_action(
    *,
    initial_lyapunov: float,
    budget: ResourceBudget,
    costs: Mapping[int, ActionCost],
    certificates: Mapping[int, DriftCertificate],
) -> LearningProgressDecision:
    """Minimize certified terminal risk over the complete public catalogue."""

    catalogue = sorted(set(costs) & set(certificates))
    if not catalogue or set(costs) != set(certificates):
        raise ValueError("costs and certificates must share one nonempty catalogue")
    horizons: dict[int, int] = {}
    risks: dict[int, float] = {}
    for q in catalogue:
        if q <= 0:
            raise ValueError("participation levels must be positive")
        horizons[q] = feasible_horizon(budget, costs[q])
        if horizons[q] <= 0:
            raise ValueError(f"action q={q} has no feasible learning update")
        risks[q] = affine_terminal_risk(
            initial_lyapunov, horizons[q], certificates[q]
        )
    selected_q = min(catalogue, key=lambda q: (risks[q], q))
    return LearningProgressDecision(
        selected_q=selected_q,
        horizons=horizons,
        certified_risks=risks,
    )
