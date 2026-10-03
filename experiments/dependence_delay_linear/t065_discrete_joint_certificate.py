"""Outcome-free certificates for the T-065 online joint controller.

The module contains two ingredients that must be valid before looking at any
controller outcome:

* a paired-block residual identity for observable signal/noise separation;
* a common quadratic Lyapunov certificate for a delayed affine recursion over
  a whole interval of online gains.

The SDP is solved offline.  Online decisions only consume the resulting
matrix and scalar constants; no online conic optimization is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class ResidualStatistics:
    """Paired-block estimators evaluated at one predictable parameter value."""

    signal: float
    noise: float


@dataclass(frozen=True)
class CommonLyapunovCertificate:
    """Offline common-quadratic certificate for a gain interval."""

    matrix: np.ndarray
    margin: float
    eta_min: float
    eta_max: float
    delay: int
    solver: str
    status: str


def paired_residual_statistics(
    first_block_mean: np.ndarray, second_block_mean: np.ndarray
) -> ResidualStatistics:
    """Return the cross-product signal and half-difference noise statistics.

    If ``first = F + eps_1`` and ``second = F + eps_2`` are conditionally
    independent with zero-mean errors of equal covariance, then

    E[first' second | history] = ||F||^2,
    E[||first-second||^2 / 2 | history] = tr(Cov(eps_1)).
    """

    first = np.asarray(first_block_mean, dtype=float)
    second = np.asarray(second_block_mean, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("paired residual means must be same-length vectors")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("paired residual means must be finite")
    difference = first - second
    return ResidualStatistics(
        signal=float(first @ second),
        noise=float(0.5 * (difference @ difference)),
    )


def ar1_block_mean_variance_factor(correlation: float, length: int) -> float:
    """Variance of a stationary unit-variance AR(1) block mean.

    The exact factor is
    ``L^-2 [L + 2 sum_{k=1}^{L-1} (L-k) correlation^k]``.
    """

    if not math.isfinite(correlation) or not -1.0 < correlation < 1.0:
        raise ValueError("correlation must lie strictly between -1 and 1")
    if int(length) != length or length < 1:
        raise ValueError("length must be a positive integer")
    lag_sum = sum(
        (length - lag) * correlation**lag for lag in range(1, length)
    )
    return float((length + 2.0 * lag_sum) / (length * length))


def delayed_companion_matrix(
    drift: np.ndarray, gain: float, delay: int
) -> np.ndarray:
    """Build the lifted matrix for e[t+1] = e[t] - gain A e[t-delay]."""

    matrix = np.asarray(drift, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("drift must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("drift must be finite")
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("gain must be finite and nonnegative")
    if int(delay) != delay or delay < 0:
        raise ValueError("delay must be a nonnegative integer")

    dimension = matrix.shape[0]
    if delay == 0:
        return np.eye(dimension) - gain * matrix

    lifted = np.zeros(((delay + 1) * dimension, (delay + 1) * dimension))
    lifted[:dimension, :dimension] = np.eye(dimension)
    lifted[:dimension, delay * dimension :] = -gain * matrix
    for block in range(1, delay + 1):
        row = slice(block * dimension, (block + 1) * dimension)
        column = slice((block - 1) * dimension, block * dimension)
        lifted[row, column] = np.eye(dimension)
    return lifted


def common_quadratic_certificate(
    drift: np.ndarray,
    *,
    eta_min: float,
    eta_max: float,
    delay: int,
    solver: str = "CLARABEL",
    positivity_floor: float = 1e-8,
) -> CommonLyapunovCertificate:
    """Find P with C(eta)' P C(eta) - P <= -margin I on an interval.

    ``C(eta)`` is affine in eta and ``C -> C' P C`` is matrix convex for
    ``P >= 0``.  Consequently, endpoint constraints certify every eta in the
    closed interval; a grid is unnecessary for the proof and is used only for
    numerical auditing.
    """

    matrix = np.asarray(drift, dtype=float)
    if not math.isfinite(eta_min) or not math.isfinite(eta_max):
        raise ValueError("gain endpoints must be finite")
    if eta_min < 0.0 or eta_min > eta_max:
        raise ValueError("gain interval must satisfy 0 <= eta_min <= eta_max")
    endpoint_matrices = [
        delayed_companion_matrix(matrix, eta, delay)
        for eta in sorted({float(eta_min), float(eta_max)})
    ]
    lifted_dimension = endpoint_matrices[0].shape[0]
    certificate = cp.Variable((lifted_dimension, lifted_dimension), symmetric=True)
    margin = cp.Variable()
    identity = np.eye(lifted_dimension)
    constraints = [
        certificate >> positivity_floor * identity,
        cp.trace(certificate) == 1.0,
        margin >= 0.0,
    ]
    constraints.extend(
        certificate - companion.T @ certificate @ companion >> margin * identity
        for companion in endpoint_matrices
    )
    problem = cp.Problem(cp.Maximize(margin), constraints)
    problem.solve(solver=solver)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"common Lyapunov SDP failed with status {problem.status}")
    if certificate.value is None or margin.value is None:
        raise RuntimeError("common Lyapunov SDP returned no certificate")
    value = 0.5 * (certificate.value + certificate.value.T)
    return CommonLyapunovCertificate(
        matrix=np.asarray(value, dtype=float),
        margin=float(margin.value),
        eta_min=float(eta_min),
        eta_max=float(eta_max),
        delay=int(delay),
        solver=solver,
        status=str(problem.status),
    )


def audit_certificate(
    drift: np.ndarray,
    certificate: CommonLyapunovCertificate,
    *,
    grid_size: int = 101,
) -> dict[str, float]:
    """Numerically audit positivity and the worst drift eigenvalue on a grid."""

    if int(grid_size) != grid_size or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least two")
    p_matrix = np.asarray(certificate.matrix, dtype=float)
    smallest_p = float(np.linalg.eigvalsh(p_matrix).min())
    worst_drift = -math.inf
    worst_radius = -math.inf
    for gain in np.linspace(certificate.eta_min, certificate.eta_max, grid_size):
        companion = delayed_companion_matrix(drift, float(gain), certificate.delay)
        difference = companion.T @ p_matrix @ companion - p_matrix
        worst_drift = max(worst_drift, float(np.linalg.eigvalsh(difference).max()))
        worst_radius = max(
            worst_radius, float(np.max(np.abs(np.linalg.eigvals(companion))))
        )
    return {
        "smallest_p_eigenvalue": smallest_p,
        "worst_drift_eigenvalue": worst_drift,
        "worst_spectral_radius": worst_radius,
    }
