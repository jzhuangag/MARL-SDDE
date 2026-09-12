"""Exact iid delayed-moment identities used as the T-030 base case.

This module deliberately does not claim that iid factorization holds for a
Markov sample coupled to the parameter history.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def variance_factor(q: int, rho: float) -> float:
    if q < 1 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid q or rho")
    return float(rho + (1.0 - rho) / q)


def delayed_iid_diagonal_step(
    *,
    mean_current: np.ndarray,
    mean_delayed: np.ndarray,
    moment_current: np.ndarray,
    moment_delayed_current: np.ndarray,
    moment_delayed: np.ndarray,
    mean_jacobian: np.ndarray,
    jacobian_operator: Callable[[np.ndarray], np.ndarray],
    noise_second_moment: np.ndarray,
    jacobian_noise_cross: Callable[[np.ndarray], np.ndarray],
    alpha: float,
    q: int,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact iid mean and new diagonal second-moment block."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    vq = variance_factor(q, rho)
    a = np.asarray(mean_jacobian, dtype=float)
    m_t = np.asarray(mean_current, dtype=float)
    m_d = np.asarray(mean_delayed, dtype=float)
    m_tt = np.asarray(moment_current, dtype=float)
    m_dt = np.asarray(moment_delayed_current, dtype=float)
    m_dd = np.asarray(moment_delayed, dtype=float)
    next_mean = m_t - alpha * a.dot(m_d)
    stochastic_quadratic = (
        vq * jacobian_operator(m_dd)
        + (1.0 - vq) * a.dot(m_dd).dot(a.T)
    )
    cross = jacobian_noise_cross(m_d)
    next_moment = (
        m_tt
        - alpha * (a.dot(m_dt) + m_dt.T.dot(a.T))
        + alpha * alpha * stochastic_quadratic
        - alpha * alpha * vq * (cross + cross.T)
        + alpha * alpha * vq * noise_second_moment
    )
    return next_mean, next_moment


def delayed_iid_cross_step(
    moment_current_older: np.ndarray,
    moment_delayed_older: np.ndarray,
    mean_jacobian: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return E[e_(t+1)e_k^T] for an older lifted block k."""

    return np.asarray(moment_current_older) - alpha * np.asarray(
        mean_jacobian
    ).dot(np.asarray(moment_delayed_older))
