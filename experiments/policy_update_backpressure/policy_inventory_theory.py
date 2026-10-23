"""Outcome-free algebra for Lyapunov policy-inventory control.

The module contains exact fixed-variance Gaussian importance-ratio identities
and the scalar drift minimizer used by the proposed asynchronous-MARL theory.
It does not generate a scientific efficacy population.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray


def gaussian_joint_log_second_moment(
    current: Array, birth: Array, variances: Array
) -> float:
    """Return ``log E_birth[(pi_current/pi_birth)^2]`` exactly.

    The policies are independent Gaussian agent blocks with identical fixed
    diagonal variances at birth and evaluation time.
    """

    current = np.asarray(current, dtype=float)
    birth = np.asarray(birth, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if current.shape != birth.shape or current.shape != variances.shape:
        raise ValueError("policy vectors and variances must have equal shape")
    if (variances <= 0).any():
        raise ValueError("variances must be positive")
    return float(np.sum((current-birth)**2/variances))


def importance_variance_inflation(log_second_moment: float) -> float:
    """Return the chi-square inflation ``E[w^2]-1``."""

    if log_second_moment < 0:
        raise ValueError("a log second moment cannot be negative")
    return float(math.expm1(log_second_moment))


def rms_importance_gradient_radius(
    *, sample_norm_bound: float, batch_size: int, log_second_moment: float
) -> float:
    """Second-moment radius for a batch-mean importance-weighted gradient.

    If the unweighted trajectory-gradient contribution has norm at most ``C``,
    the mean-square estimation error is at most ``C^2 E[w^2] / B``.  The
    returned value is the corresponding root-mean-square radius.  It is not a
    high-probability confidence interval.
    """

    if sample_norm_bound < 0 or batch_size <= 0 or log_second_moment < 0:
        raise ValueError("invalid second-moment radius inputs")
    return float(
        sample_norm_bound
        * math.exp(0.5*log_second_moment)
        / math.sqrt(batch_size)
    )


@dataclass(frozen=True)
class CompletedInventoryProposal:
    direction_norm: float
    error_radius: float
    block_smoothness: float
    log_second_moment: float
    max_step: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.direction_norm,
            self.error_radius,
            self.block_smoothness,
            self.log_second_moment,
            self.max_step,
            self.weight,
        )
        if min(values) < 0 or self.block_smoothness == 0:
            raise ValueError("invalid completed inventory proposal")


@dataclass(frozen=True)
class PendingInventory:
    """Exact quadratic log-second-moment response to a scalar block step.

    ``linear`` and ``quadratic`` define
    ``z(alpha) = z(0) + linear*alpha + quadratic*alpha^2``.
    For a Gaussian block with displacement ``Delta``, update direction ``g``
    and variance ``sigma^2``, they equal ``2 <Delta,g>/sigma^2`` and
    ``||g||^2/sigma^2`` respectively.
    """

    log_second_moment: float
    linear: float
    quadratic: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.log_second_moment < 0 or self.quadratic < 0 or self.weight < 0:
            raise ValueError("invalid pending inventory")

    def post_step_log_second_moment(self, step: float) -> float:
        if step < 0:
            raise ValueError("step must be nonnegative")
        value = (
            self.log_second_moment
            + self.linear*step
            + self.quadratic*step*step
        )
        # Exact Gaussian geometry is nonnegative.  Tiny negative values can
        # arise from floating-point cancellation when a step moves to birth.
        if value < -1e-12:
            raise ValueError("declared coefficients violate nonnegative geometry")
        return float(max(value, 0.0))


def certified_gain(proposal: CompletedInventoryProposal, step: float) -> float:
    """One-event improvement on a declared norm-error confidence event."""

    if not 0 <= step <= proposal.max_step:
        raise ValueError("step outside public interval")
    signal = proposal.direction_norm
    return float(
        step*(signal*signal-proposal.error_radius*signal)
        - 0.5*proposal.block_smoothness*step*step*signal*signal
    )


def inventory_lyapunov_drift(
    proposal: CompletedInventoryProposal,
    pending: tuple[PendingInventory, ...],
    step: float,
    *,
    potential_weight: float,
) -> float:
    """One-event physical inventory-risk drift envelope."""

    if potential_weight <= 0:
        raise ValueError("potential weight must be positive")
    drift = -proposal.weight*importance_variance_inflation(
        proposal.log_second_moment
    )
    drift -= potential_weight*certified_gain(proposal, step)
    for item in pending:
        before = importance_variance_inflation(item.log_second_moment)
        after = importance_variance_inflation(
            item.post_step_log_second_moment(step)
        )
        drift += item.weight*(after-before)
    return float(drift)


def inventory_drift_derivative(
    proposal: CompletedInventoryProposal,
    pending: tuple[PendingInventory, ...],
    step: float,
    *,
    potential_weight: float,
) -> float:
    if potential_weight <= 0 or not 0 <= step <= proposal.max_step:
        raise ValueError("invalid derivative query")
    signal = proposal.direction_norm
    derivative = -potential_weight*(
        signal*signal-proposal.error_radius*signal
        - proposal.block_smoothness*step*signal*signal
    )
    for item in pending:
        z = item.post_step_log_second_moment(step)
        derivative += (
            item.weight*math.exp(z)*(item.linear+2.0*item.quadratic*step)
        )
    return float(derivative)


def inventory_drift_second_derivative(
    proposal: CompletedInventoryProposal,
    pending: tuple[PendingInventory, ...],
    step: float,
    *,
    potential_weight: float,
) -> float:
    if potential_weight <= 0 or not 0 <= step <= proposal.max_step:
        raise ValueError("invalid curvature query")
    signal = proposal.direction_norm
    curvature = (
        potential_weight*proposal.block_smoothness*signal*signal
    )
    for item in pending:
        z = item.post_step_log_second_moment(step)
        slope = item.linear+2.0*item.quadratic*step
        curvature += item.weight*math.exp(z)*(
            slope*slope+2.0*item.quadratic
        )
    return float(curvature)


def inventory_optimal_step(
    proposal: CompletedInventoryProposal,
    pending: tuple[PendingInventory, ...],
    *,
    potential_weight: float,
    tolerance: float = 1e-12,
    max_iterations: int = 160,
) -> float:
    """Minimize the convex scalar drift by safeguarded bisection."""

    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("invalid scalar-solver controls")
    cap = proposal.max_step
    if cap == 0:
        return 0.0
    at_zero = inventory_drift_derivative(
        proposal, pending, 0.0, potential_weight=potential_weight
    )
    if at_zero >= 0:
        return 0.0
    at_cap = inventory_drift_derivative(
        proposal, pending, cap, potential_weight=potential_weight
    )
    if at_cap <= 0:
        return float(cap)
    low, high = 0.0, float(cap)
    for _ in range(max_iterations):
        middle = 0.5*(low+high)
        derivative = inventory_drift_derivative(
            proposal, pending, middle, potential_weight=potential_weight
        )
        if derivative <= 0:
            low = middle
        else:
            high = middle
        if high-low <= tolerance:
            break
    return float(0.5*(low+high))
