"""Deterministic algebra gates for clocked asynchronous policy learning.

These utilities verify finite-dimensional identities used by the prospective
Lyapunov--Krasovskii analysis.  They are not an RL algorithm or efficacy
experiment.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def quadratic_potential(theta: Array, curvature: Array) -> float:
    """Return ``-0.5 * theta.T @ curvature @ theta``.

    ``curvature`` is expected to be symmetric positive semidefinite, making the
    returned potential concave.  Validation is deliberately explicit because
    silent use of a nonsymmetric matrix would invalidate the gradient identity.
    """

    theta = np.asarray(theta, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    if curvature.shape != (theta.size, theta.size):
        raise ValueError("curvature shape does not match theta")
    if not np.allclose(curvature, curvature.T, atol=1e-12, rtol=0.0):
        raise ValueError("curvature must be symmetric")
    if float(np.min(np.linalg.eigvalsh(curvature))) < -1e-12:
        raise ValueError("curvature must be positive semidefinite")
    return -0.5 * float(theta @ curvature @ theta)


def quadratic_block_gradient(
    theta: Array, curvature: Array, block: int
) -> float:
    """Exact scalar-block gradient of :func:`quadratic_potential`."""

    theta = np.asarray(theta, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    if block < 0 or block >= theta.size:
        raise IndexError("block outside theta")
    return -float(curvature[block] @ theta)


def interaction_weighted_drift_bound(
    cross_lipschitz: Array,
    current: Array,
    birth: Array,
    block: int,
) -> float:
    """Block-gradient mismatch bound from coordinate-wise policy drift.

    If ``|grad_i(x)-grad_i(y)| <= sum_j L[i,j] |x_j-y_j|``, this is the
    right-hand side.  Nonnegative entries are required; their provenance is a
    theorem obligation rather than something inferred by this helper.
    """

    matrix = np.asarray(cross_lipschitz, dtype=float)
    current = np.asarray(current, dtype=float)
    birth = np.asarray(birth, dtype=float)
    if current.shape != birth.shape:
        raise ValueError("current and birth shapes differ")
    if matrix.shape != (current.size, current.size):
        raise ValueError("cross_lipschitz shape does not match parameters")
    if (matrix < 0.0).any():
        raise ValueError("cross_lipschitz entries must be nonnegative")
    if block < 0 or block >= current.size:
        raise IndexError("block outside parameter vector")
    return float(matrix[block] @ np.abs(current-birth))


def quadratic_smooth_gain_lower_bound(
    current: Array,
    stale: Array,
    curvature: Array,
    block: int,
    step_size: float,
) -> tuple[float, float]:
    """Return actual and smoothness-certified gain for one stale block step.

    The update uses the exact block gradient at ``stale`` while the objective is
    evaluated at ``current``.  For a concave quadratic, the block smoothness
    constant is the diagonal curvature entry and the standard lower bound is
    exact.
    """

    if step_size < 0.0:
        raise ValueError("step_size must be nonnegative")
    current = np.asarray(current, dtype=float)
    stale = np.asarray(stale, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    stale_gradient = quadratic_block_gradient(stale, curvature, block)
    current_gradient = quadratic_block_gradient(current, curvature, block)
    updated = current.copy()
    updated[block] += step_size*stale_gradient
    actual = quadratic_potential(updated, curvature)-quadratic_potential(
        current, curvature
    )
    smoothness = float(curvature[block, block])
    certified = (
        step_size*current_gradient*stale_gradient
        -0.5*smoothness*(step_size*stale_gradient)**2
    )
    return actual, certified


def krasovskii_history_drift(
    previous_step_energy: Array, new_step_energy: float
) -> tuple[float, float]:
    """Return exact drift and telescoping expression for a delay history.

    For a fixed delay window of length ``D`` define

    ``H_k = sum_(r=k-D)^(k-1) (r-(k-D)+1) ||s_r||^2``.

    Supplying the ``D`` past squared step norms from oldest to newest gives
    ``H_(k+1)-H_k = D||s_k||^2-sum_(r=k-D)^(k-1)||s_r||^2``.
    """

    energy = np.asarray(previous_step_energy, dtype=float)
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("previous_step_energy must be a nonempty vector")
    if (energy < 0.0).any() or new_step_energy < 0.0:
        raise ValueError("step energies must be nonnegative")
    weights = np.arange(1, energy.size+1, dtype=float)
    old_value = float(weights @ energy)
    shifted = np.concatenate((energy[1:], np.asarray([new_step_energy])))
    new_value = float(weights @ shifted)
    direct = new_value-old_value
    telescope = float(energy.size*new_step_energy-np.sum(energy))
    return direct, telescope


def first_arrival_noise_mean(*, endogenous: bool) -> float:
    """Enumerate a two-worker arrival-order selection-bias counterexample.

    Each worker has independent Rademacher gradient noise.  Under endogenous
    completion, positive noise finishes at time zero and negative noise at time
    one.  The server applies the first arrival and breaks ties by worker index.
    Under exogenous completion, worker zero always arrives first.  Every worker
    is marginally unbiased in both cases, but endogenous arrival ordering is
    informative.
    """

    selected: list[float] = []
    for noise_0, noise_1 in product((-1.0, 1.0), repeat=2):
        if endogenous:
            completion = (
                0 if noise_0 > 0.0 else 1,
                0 if noise_1 > 0.0 else 1,
            )
        else:
            completion = (0, 1)
        first = min(range(2), key=lambda worker: (completion[worker], worker))
        selected.append((noise_0, noise_1)[first])
    return float(np.mean(selected))

