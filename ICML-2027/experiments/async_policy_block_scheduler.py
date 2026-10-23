"""Exact small-team optimizer for a certified drift-plus-penalty surrogate.

This module only solves the finite-dimensional action problem. It does not
estimate policy gradients, confidence radii, curvature certificates, or prove
that a critic surrogate lower-bounds MARL return. Those are separate research
obligations; see internal/async_policy_block_scheduler_candidate.md.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ScheduleAction:
    agents: tuple[int, ...]
    step_sizes: tuple[float, ...]
    certified_gain_bound: float
    drift_score: float
    message_charge: float
    environment_charge: float


def _validate_inputs(gradient_norms, confidence_radii, curvature_majorant,
                     message_costs, environment_costs, alpha_max):
    norms = np.asarray(gradient_norms, dtype=float)
    radii = np.asarray(confidence_radii, dtype=float)
    curvature = np.asarray(curvature_majorant, dtype=float)
    message = np.asarray(message_costs, dtype=float)
    environment = np.asarray(environment_costs, dtype=float)
    n = norms.size
    if n == 0 or radii.shape != (n,) or message.shape != (n,) or environment.shape != (n,):
        raise ValueError("all per-agent vectors must have the same nonzero length")
    if curvature.shape != (n, n):
        raise ValueError("curvature_majorant must be an n by n matrix")
    arrays = (norms, radii, curvature, message, environment, np.asarray(alpha_max, dtype=float))
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("all inputs must be finite")
    if np.any(norms < 0) or np.any(radii < 0) or np.any(message < 0) or np.any(environment < 0):
        raise ValueError("norms, radii, and costs must be nonnegative")
    if not np.allclose(curvature, curvature.T, rtol=1e-10, atol=1e-12):
        raise ValueError("curvature_majorant must be symmetric")
    if np.any(curvature < -1e-12):
        raise ValueError("curvature_majorant must be entrywise nonnegative")
    if np.linalg.eigvalsh(curvature).min() < -1e-10:
        raise ValueError("curvature_majorant must be positive semidefinite")
    caps = np.broadcast_to(np.asarray(alpha_max, dtype=float), (n,)).copy()
    if np.any(caps <= 0):
        raise ValueError("alpha_max must be positive")
    return norms, radii, curvature, message, environment, caps


def _subsets(n):
    # Include the empty action first; lexicographic order is the deterministic tie-break.
    yield ()
    for size in range(1, n + 1):
        yield from combinations(range(n), size)


def _solve_subset(indices, norms, radii, curvature, caps, value_weight,
                  message_pressure, environment_pressure, message, environment):
    if not indices:
        return ScheduleAction((), (), 0.0, 0.0, 0.0, 0.0)
    idx = np.asarray(indices, dtype=int)
    g = norms[idx]
    certified_linear = g * g - radii[idx] * g
    # The Taylor penalty is 1/2 * alpha' H alpha, where H is PSD because
    # it is a principal submatrix of diag(g) L diag(g).
    hessian = np.outer(g, g) * curvature[np.ix_(idx, idx)]
    hessian = 0.5 * (hessian + hessian.T)
    caps_s = caps[idx]

    def objective(alpha):
        return -(value_weight * (certified_linear @ alpha - 0.5 * alpha @ hessian @ alpha))

    def jacobian(alpha):
        return -value_weight * (certified_linear - hessian @ alpha)

    result = minimize(objective, np.zeros(len(idx)), jac=jacobian,
                      method="L-BFGS-B", bounds=[(0.0, float(c)) for c in caps_s],
                      options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000,
                               "maxls": 50})
    if not result.success and np.linalg.norm(result.jac, ord=np.inf) > 1e-7:
        raise RuntimeError(f"convex step-size subproblem did not converge: {result.message}")
    alpha = np.clip(result.x, 0.0, caps_s)
    gain = float(certified_linear @ alpha - 0.5 * alpha @ hessian @ alpha)
    mcharge = float(message[idx].sum())
    echarge = float(environment[idx].sum())
    score = (value_weight * gain - message_pressure * mcharge
             - environment_pressure * echarge)
    return ScheduleAction(tuple(int(i) for i in indices),
                          tuple(float(x) for x in alpha), gain, float(score),
                          mcharge, echarge)


def select_policy_blocks(*, gradient_norms, confidence_radii, curvature_majorant,
                         message_costs, environment_costs, value_weight=1.0,
                         message_pressure=0.0, environment_pressure=0.0,
                         alpha_max=1.0):
    """Maximize the certified surrogate exactly over subsets, numerically over steps.

    For each subset, the continuous objective is a concave quadratic over a
    box because the supplied curvature majorant is PSD. The outer subset
    optimization is exhaustive, so the selected discrete action is globally
    optimal for this finite surrogate (not necessarily for true return).
    Complexity is exponential in the number of agents; this implementation is
    intended for small-team qualification and is not a scalability claim.
    """
    (norms, radii, curvature, message, environment, caps) = _validate_inputs(
        gradient_norms, confidence_radii, curvature_majorant, message_costs,
        environment_costs, alpha_max)
    pressures = np.asarray([value_weight, message_pressure, environment_pressure], dtype=float)
    if not np.all(np.isfinite(pressures)) or value_weight < 0 or message_pressure < 0 or environment_pressure < 0:
        raise ValueError("value weight and virtual-queue pressures must be finite and nonnegative")
    candidates = [_solve_subset(s, norms, radii, curvature, caps, value_weight,
                                message_pressure, environment_pressure, message, environment)
                 for s in _subsets(len(norms))]
    # Stable tie-break: highest score, then fewer packets, then lexicographic agents.
    return min(candidates, key=lambda a: (-a.drift_score, len(a.agents), a.agents))
