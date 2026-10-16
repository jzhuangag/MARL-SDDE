"""Exact Gaussian lower bound for correlation-limited agent speedup."""

from typing import Dict, Iterable

import numpy as np


def observation_covariance(
    q: int,
    common_variance: float,
    private_variance: float,
) -> np.ndarray:
    """Return the q-agent covariance for one Gaussian observation round."""

    if q < 1:
        raise ValueError("q must be positive")
    if common_variance < 0.0 or private_variance <= 0.0:
        raise ValueError("variances must satisfy common >= 0 and private > 0")
    return (
        private_variance * np.eye(q)
        + common_variance * np.ones((q, q))
    )


def fisher_information(
    q: int,
    common_variance: float,
    private_variance: float,
) -> float:
    """Return per-round Fisher information for the shared mean."""

    if q < 1:
        raise ValueError("q must be positive")
    if common_variance < 0.0 or private_variance <= 0.0:
        raise ValueError("variances must satisfy common >= 0 and private > 0")
    return float(q / (private_variance + q * common_variance))


def minimax_risk(
    rounds: int,
    q: int,
    common_variance: float,
    private_variance: float,
) -> float:
    """Return the exact fixed-q minimax squared-error risk."""

    if rounds < 1:
        raise ValueError("rounds must be positive")
    return float(
        (common_variance + private_variance / q) / rounds
    )


def effective_speedup(q: int, rho: float) -> float:
    """Return exact speedup relative to one agent at total correlation rho."""

    if q < 1:
        raise ValueError("q must be positive")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    return float(q / (1.0 + (q - 1.0) * rho))


def information_per_cost(
    q: int,
    overhead: float,
    common_variance: float,
    private_variance: float,
) -> float:
    """Return Fisher information per unit resource cost."""

    if overhead < 0.0:
        raise ValueError("overhead must be nonnegative")
    return fisher_information(
        q, common_variance, private_variance
    ) / (overhead + q)


def continuous_optimal_participation(
    maximum_agents: int,
    overhead: float,
    common_variance: float,
    private_variance: float,
) -> float:
    """Return the clipped continuous optimum of the lower-bound objective."""

    if maximum_agents < 1:
        raise ValueError("maximum_agents must be positive")
    if overhead < 0.0:
        raise ValueError("overhead must be nonnegative")
    if common_variance < 0.0 or private_variance <= 0.0:
        raise ValueError("variances must satisfy common >= 0 and private > 0")
    if common_variance == 0.0:
        return float(maximum_agents)
    unconstrained = np.sqrt(
        overhead * private_variance / common_variance
    )
    return float(np.clip(unconstrained, 1.0, maximum_agents))


def optimal_integer_participation(
    candidate_agents: Iterable[int],
    overhead: float,
    common_variance: float,
    private_variance: float,
) -> Dict[str, float]:
    """Return the candidate with maximum information per resource cost."""

    candidates = sorted({int(q) for q in candidate_agents})
    if not candidates or candidates[0] < 1:
        raise ValueError("candidate_agents must be positive")
    rows = [
        (
            information_per_cost(
                q, overhead, common_variance, private_variance
            ),
            q,
        )
        for q in candidates
    ]
    efficiency, q_star = max(rows, key=lambda item: (item[0], -item[1]))
    return {
        "q": int(q_star),
        "information_per_cost": float(efficiency),
    }


def adaptive_budget_lower_bound(
    budget: float,
    candidate_agents: Iterable[int],
    overhead: float,
    common_variance: float,
    private_variance: float,
) -> float:
    """Return the minimax lower bound for predictable adaptive participation."""

    if budget <= 0.0:
        raise ValueError("budget must be positive")
    best = optimal_integer_participation(
        candidate_agents,
        overhead,
        common_variance,
        private_variance,
    )
    return float(1.0 / (budget * best["information_per_cost"]))
