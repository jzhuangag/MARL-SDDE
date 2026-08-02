"""Exact finite-horizon risk of Polyak--Ruppert averaged delayed linear SA."""

from __future__ import annotations

import numpy as np

from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    delayed_vector_companion,
)


def exact_pr_averaged_vector_risk(
    *,
    initial_history: np.ndarray,
    drift: np.ndarray,
    step_size: float,
    delay: int,
    updates: int,
    burn_in: int,
    lag_covariances: np.ndarray | None,
    risk_matrix: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Risk of the average of iterates ``e[burn_in+1],...,e[updates]``.

    The recursion is ``e[t+1]=e[t]-eta*A*e[t-D]+eta*xi[t]``.  This function
    is exact for any zero-mean innovation process specified by its finite lag
    covariance matrices; Gaussianity is unnecessary.
    """

    matrix = np.asarray(drift, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("drift must be square")
    dimension = matrix.shape[0]
    history = np.asarray(initial_history, dtype=float)
    if history.shape != (delay + 1, dimension):
        raise ValueError("initial_history must have shape (delay+1, dimension)")
    if updates < 1 or burn_in < 0 or burn_in >= updates:
        raise ValueError("require updates >= 1 and 0 <= burn_in < updates")
    weight = np.eye(dimension) if risk_matrix is None else np.asarray(risk_matrix, dtype=float)
    if weight.shape != (dimension, dimension) or not np.allclose(weight, weight.T):
        raise ValueError("risk_matrix must be symmetric with drift dimension")

    companion = delayed_vector_companion(matrix, step_size, delay)
    selector = np.zeros((dimension, dimension * (delay + 1)))
    selector[:, :dimension] = np.eye(dimension)
    injector = selector.T
    lifted_initial = history.reshape(-1)
    averaged_count = updates - burn_in
    mean = np.zeros(dimension)
    for time in range(burn_in + 1, updates + 1):
        mean += selector @ np.linalg.matrix_power(companion, time) @ lifted_initial
    mean /= averaged_count

    impulses: list[np.ndarray] = []
    for innovation_time in range(updates):
        impulse = np.zeros((dimension, dimension))
        first_affected = max(burn_in + 1, innovation_time + 1)
        for iterate_time in range(first_affected, updates + 1):
            impulse += (
                step_size
                * selector
                @ np.linalg.matrix_power(
                    companion, iterate_time - 1 - innovation_time
                )
                @ injector
            )
        impulses.append(impulse / averaged_count)

    covariance = np.zeros((dimension, dimension))
    if lag_covariances is None:
        raise ValueError("lag_covariances are required")
    lags = np.asarray(lag_covariances, dtype=float)
    expected_shape = (2 * updates - 1, dimension, dimension)
    if lags.shape != expected_shape:
        raise ValueError(f"lag_covariances must have shape {expected_shape}")
    center = updates - 1
    for left in range(updates):
        for right in range(updates):
            covariance += (
                impulses[left]
                @ lags[center + left - right]
                @ impulses[right].T
            )
    covariance = (covariance + covariance.T) / 2.0
    bias_risk = float(mean @ weight @ mean)
    noise_risk = float(np.trace(weight @ covariance))
    return {
        "mean": mean,
        "covariance": covariance,
        "bias_risk": bias_risk,
        "noise_risk": noise_risk,
        "risk": bias_risk + noise_risk,
        "averaged_count": averaged_count,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(companion)))),
    }


def pr_message_proxy(*, q: int, rho: float, overhead: float) -> float:
    """Asymptotic PR noise proxy up to q-independent constants.

    With message budget B, N(q)=B/(overhead+q) and aggregate long-run
    covariance factor ``rho+(1-rho)/q``.  Hence the leading risk is
    proportional to the returned value divided by B.
    """

    if q < 1 or overhead < 0.0 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid q, overhead, or rho")
    return float((rho + (1.0 - rho) / q) * (overhead + q))


def discrete_pr_proxy_optimum(
    *, catalogue: list[int] | tuple[int, ...], rho: float, overhead: float
) -> int:
    if not catalogue or any(q < 1 for q in catalogue):
        raise ValueError("catalogue must contain positive participation counts")
    return min(catalogue, key=lambda q: (pr_message_proxy(q=q, rho=rho, overhead=overhead), q))
