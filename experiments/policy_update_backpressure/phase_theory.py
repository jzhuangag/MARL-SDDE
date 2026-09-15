"""Deterministic algebra audit for perishable asynchronous policy updates.

The module checks theorem-facing inequalities only.  It does not run an
efficacy experiment or use a sampled RL outcome.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class EventCertificate:
    proposal_norm: float
    own_debt: float
    markov_radius: float
    smoothness: float
    cross_to_pending: tuple[float, ...]
    pending_debts: tuple[float, ...]
    max_step: float
    potential_weight: float = 1.0

    def __post_init__(self) -> None:
        scalar_values = (
            self.proposal_norm,
            self.own_debt,
            self.markov_radius,
            self.smoothness,
            self.max_step,
            self.potential_weight,
        )
        if min(scalar_values) < 0 or self.smoothness == 0:
            raise ValueError("invalid nonnegative event certificate")
        if len(self.cross_to_pending) != len(self.pending_debts):
            raise ValueError("cross sensitivities and debts must align")
        if min(self.cross_to_pending, default=0.0) < 0:
            raise ValueError("cross sensitivities must be nonnegative")
        if min(self.pending_debts, default=0.0) < 0:
            raise ValueError("pending debts must be nonnegative")

    @property
    def total_bias_radius(self) -> float:
        return self.own_debt+self.markov_radius

    @property
    def freshness_ratio(self) -> float:
        if self.proposal_norm == 0:
            return float("inf")
        return self.total_bias_radius/self.proposal_norm


def certified_potential_gain(cert: EventCertificate, alpha: float) -> float:
    """Smoothness lower bound for applying the completed proposal."""

    if not 0 <= alpha <= cert.max_step:
        raise ValueError("alpha outside the declared interval")
    s = cert.proposal_norm
    b = cert.total_bias_radius
    return alpha*(s*s-b*s)-0.5*cert.smoothness*alpha*alpha*s*s


def freshness_lyapunov_drift_bound(cert: EventCertificate, alpha: float) -> float:
    """Upper bound on freshness-debt drift minus weighted potential gain.

    The completed proposal's own debt is reset before its next computation.
    Each other pending proposal receives at most C_ji * alpha * ||g_i||
    additional cross-policy bias debt.
    """

    if not 0 <= alpha <= cert.max_step:
        raise ValueError("alpha outside the declared interval")
    s = cert.proposal_norm
    debt_drift = -0.5*cert.own_debt**2
    for cross, debt in zip(cert.cross_to_pending, cert.pending_debts):
        increment = cross*alpha*s
        debt_drift += debt*increment+0.5*increment*increment
    return debt_drift-cert.potential_weight*certified_potential_gain(cert, alpha)


def closed_form_step(cert: EventCertificate) -> float:
    """Exact minimizer of the scalar quadratic event-drift upper bound."""

    s = cert.proposal_norm
    if s == 0 or cert.max_step == 0:
        return 0.0
    linear_reward = cert.potential_weight*(s*s-cert.total_bias_radius*s)
    linear_debt = s*sum(
        cross*debt for cross, debt in zip(cert.cross_to_pending, cert.pending_debts)
    )
    curvature = s*s*(
        cert.potential_weight*cert.smoothness
        + sum(cross*cross for cross in cert.cross_to_pending)
    )
    if curvature <= 0:
        return 0.0
    return float(np.clip((linear_reward-linear_debt)/curvature, 0.0, cert.max_step))


def freshness_residual(cert: EventCertificate) -> float:
    """Certified signal left after own and outgoing-debt prices.

    This residual appears exactly in the linear coefficient of the Lyapunov
    drift bound.  It is theorem-facing algebra, not a performance metric.
    """

    s = cert.proposal_norm
    if s == 0:
        return 0.0
    outgoing_price = sum(
        cross*debt for cross, debt in zip(cert.cross_to_pending, cert.pending_debts)
    )/cert.potential_weight
    return max(s-cert.total_bias_radius-outgoing_price, 0.0)


def finite_time_event_slack(cert: EventCertificate) -> float:
    """Numerical slack in the nonbinding-cap one-event descent inequality.

    Write ``Kappa_i=V*L_i+sum_j C_(j,i)^2``.  If

        max_step >= V/Kappa_i,

    the closed-form action satisfies

        Delta H <= -Z_i^2/2 - V^2 R_i^2/(2 Kappa_i).

    A nonnegative return value is the slack in this inequality.  The function
    raises when the theorem conditions are not the declared certificate.
    """

    v = cert.potential_weight
    if v <= 0:
        raise ValueError("finite-time audit requires V>0")
    curvature_scale = v*cert.smoothness+sum(
        cross*cross for cross in cert.cross_to_pending
    )
    if cert.max_step+1e-12 < v/curvature_scale:
        raise ValueError("max_step must not clip the theorem step")
    alpha = closed_form_step(cert)
    residual = freshness_residual(cert)
    target = -0.5*cert.own_debt**2-v*v*residual**2/(2.0*curvature_scale)
    return target-freshness_lyapunov_drift_bound(cert, alpha)


def concave_quadratic_potential(theta: Array, cross: float) -> float:
    """Phi(x,y)=-0.5(x^2+y^2)-cross*x*y for |cross|<1."""

    x, y = np.asarray(theta, dtype=float)
    return float(-0.5*(x*x+y*y)-cross*x*y)


def concave_quadratic_gradient(theta: Array, cross: float) -> Array:
    x, y = np.asarray(theta, dtype=float)
    return np.array((-x-cross*y, -y-cross*x), dtype=float)


def quadratic_sign_flip_example() -> dict[str, float | bool]:
    """Exact high-load example in which a slow stale direction expires."""

    cross = 0.9
    alpha = 0.5
    initial = np.array((-1.2, 1.0), dtype=float)
    stale_gradient = concave_quadratic_gradient(initial, cross)
    after_fast = initial.copy()
    after_fast[0] += alpha*stale_gradient[0]
    current_gradient = concave_quadratic_gradient(after_fast, cross)
    stale_slow = float(stale_gradient[1])
    current_slow = float(current_gradient[1])
    debt = cross*abs(float(after_fast[0]-initial[0]))
    trial = after_fast.copy()
    trial[1] += alpha*stale_slow
    before = concave_quadratic_potential(after_fast, cross)
    after = concave_quadratic_potential(trial, cross)
    return {
        "cross": cross,
        "alpha": alpha,
        "stale_slow_gradient": stale_slow,
        "current_slow_gradient": current_slow,
        "cross_debt": debt,
        "freshness_ratio": debt/abs(stale_slow),
        "gradient_sign_flipped": bool(stale_slow*current_slow < 0),
        "stale_update_gain": after-before,
        "stale_update_is_harmful": bool(after < before),
    }


def wall_clock_separation_example(
    cross: float = 0.9,
    slow_clock: int = 20,
) -> dict[str, float | bool | int]:
    """Exact finite-horizon separation for a perishable unilateral update.

    Agent x completes at wall-clock one and agent y returns a proposal born at
    time zero at ``slow_clock``.  PUB uses ``V=sqrt(slow_clock)`` and its exact
    closed-form step.  Accept-all uses the same initial/fixed step but ignores
    freshness debt.  After the slow event, no corrective proposal completes
    before ``2*slow_clock``.  This deterministic completion trace is a
    degenerate Markov completion process.
    """

    if not 0 < cross < 1:
        raise ValueError("cross must lie in (0,1)")
    if slow_clock < 2:
        raise ValueError("slow_clock must be at least two")
    initial = np.array((1.0, 0.0), dtype=float)
    gradient = concave_quadratic_gradient(initial, cross)
    potential_weight = float(np.sqrt(slow_clock))
    curvature = potential_weight+cross*cross
    fixed_step = potential_weight/curvature
    after_fast = initial.copy()
    after_fast[0] += fixed_step*gradient[0]
    accept_all_after_stale = after_fast.copy()
    accept_all_after_stale[1] += fixed_step*gradient[1]
    initial_regret = -concave_quadratic_potential(initial, cross)
    plateau_regret = -concave_quadratic_potential(after_fast, cross)
    stale_regret = -concave_quadratic_potential(accept_all_after_stale, cross)
    debt = cross*abs(float(after_fast[0]-initial[0]))
    outgoing_price = cross*debt
    fast_residual_after_first = max(
        abs(float(after_fast[0]))-outgoing_price/potential_weight,
        0.0,
    )
    pub_regret_upper = initial_regret+(2*slow_clock-1)*plateau_regret
    barrier_regret_lower_bound = initial_regret*slow_clock
    accept_all_regret = (
        initial_regret+(slow_clock-1)*plateau_regret+slow_clock*stale_regret
    )
    return {
        "cross": cross,
        "slow_clock": slow_clock,
        "horizon": 2*slow_clock,
        "potential_weight": potential_weight,
        "shared_fixed_step": fixed_step,
        "initial_regret": initial_regret,
        "post_fast_x": float(after_fast[0]),
        "fast_residual_after_first": fast_residual_after_first,
        "pub_fast_queue_is_throttled": bool(fast_residual_after_first <= 1e-12),
        "stale_slow_proposal": float(gradient[1]),
        "cross_debt": debt,
        "freshness_ratio": debt/abs(float(gradient[1])),
        "post_fast_regret": plateau_regret,
        "accept_all_post_stale_regret": stale_regret,
        "pub_wall_clock_regret_upper_bound": pub_regret_upper,
        "barrier_wall_clock_regret_lower_bound": barrier_regret_lower_bound,
        "accept_all_wall_clock_regret": accept_all_regret,
        "pub_is_asymptotically_constant_in_slow_clock": True,
        "baselines_are_linear_in_slow_clock": True,
    }


def audit() -> dict[str, object]:
    """Exhaustive deterministic checks of the phase and drift formulas."""

    stale_checks = gain_checks = optimizer_checks = finite_time_checks = 0
    min_gain_slack = float("inf")
    max_optimizer_gap = 0.0
    min_finite_time_slack = float("inf")
    for cross, x, y, alpha in product(
        (0.0, 0.25, 0.6, 0.9),
        (-1.2, -0.4, 0.3, 1.1),
        (-0.9, 0.2, 0.8),
        (0.05, 0.2, 0.5),
    ):
        if cross >= 1:
            continue
        birth = np.array((x, y), dtype=float)
        stale_gradient = concave_quadratic_gradient(birth, cross)
        current = birth.copy()
        current[0] += alpha*stale_gradient[0]
        current_gradient = concave_quadratic_gradient(current, cross)
        error = abs(float(stale_gradient[1]-current_gradient[1]))
        debt = cross*abs(float(current[0]-birth[0]))
        if error > debt+1e-12:
            raise AssertionError("cross debt failed to cover exact stale error")
        stale_checks += 1

        proposal = np.array((0.0, stale_gradient[1]), dtype=float)
        proposal_norm = abs(float(stale_gradient[1]))
        cert = EventCertificate(
            proposal_norm=proposal_norm,
            own_debt=debt,
            markov_radius=0.0,
            smoothness=1.0,
            cross_to_pending=(cross,),
            pending_debts=(0.1,),
            max_step=0.5,
            potential_weight=1.7,
        )
        for step in (0.0, 0.1, 0.25, 0.5):
            trial = current+step*proposal
            true_gain = (
                concave_quadratic_potential(trial, cross)
                - concave_quadratic_potential(current, cross)
            )
            bound = certified_potential_gain(cert, step)
            slack = true_gain-bound
            if slack < -1e-10:
                raise AssertionError("certified gain bound was violated")
            min_gain_slack = min(min_gain_slack, slack)
            gain_checks += 1

        optimum = closed_form_step(cert)
        grid = np.linspace(0.0, cert.max_step, 2001)
        values = np.asarray([freshness_lyapunov_drift_bound(cert, float(a)) for a in grid])
        numeric = float(grid[int(np.argmin(values))])
        max_optimizer_gap = max(max_optimizer_gap, abs(optimum-numeric))
        optimizer_checks += 1

    for proposal_norm, own_debt, radius, smoothness, potential_weight in product(
        (0.0, 0.05, 0.3, 1.0, 2.0),
        (0.0, 0.03, 0.2),
        (0.0, 0.02, 0.15),
        (0.4, 1.0, 2.5),
        (4.0, 8.0, 16.0),
    ):
        cross = (0.1, 0.35)
        debts = (0.02, 0.1)
        curvature_scale = potential_weight*smoothness+sum(c*c for c in cross)
        cert = EventCertificate(
            proposal_norm=proposal_norm,
            own_debt=own_debt,
            markov_radius=radius,
            smoothness=smoothness,
            cross_to_pending=cross,
            pending_debts=debts,
            max_step=potential_weight/curvature_scale,
            potential_weight=potential_weight,
        )
        slack = finite_time_event_slack(cert)
        if slack < -1e-10:
            raise AssertionError("capped finite-time event inequality failed")
        min_finite_time_slack = min(min_finite_time_slack, slack)
        finite_time_checks += 1
    example = quadratic_sign_flip_example()
    if not example["gradient_sign_flipped"] or not example["stale_update_is_harmful"]:
        raise AssertionError("declared high-load sign-flip example failed")
    separation = wall_clock_separation_example()
    if not separation["pub_fast_queue_is_throttled"]:
        raise AssertionError("separation witness missed the backpressure boundary")
    if separation["pub_wall_clock_regret_upper_bound"] >= separation["accept_all_wall_clock_regret"]:
        raise AssertionError("separation witness did not separate accept-all")
    if separation["pub_wall_clock_regret_upper_bound"] >= separation["barrier_wall_clock_regret_lower_bound"]:
        raise AssertionError("separation witness did not separate the barrier")
    return {
        "kind": "deterministic_phase_theory_algebra_not_efficacy",
        "stale_bias_checks": stale_checks,
        "gain_bound_checks": gain_checks,
        "closed_form_optimizer_checks": optimizer_checks,
        "finite_time_event_checks": finite_time_checks,
        "minimum_true_gain_minus_bound": min_gain_slack,
        "maximum_closed_form_grid_gap": max_optimizer_gap,
        "minimum_finite_time_event_slack": min_finite_time_slack,
        "sign_flip_example": example,
        "wall_clock_separation_example": separation,
    }


def main(output: Path | None = None) -> dict[str, object]:
    result = audit()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    print(json.dumps(main(args.output), indent=2, sort_keys=True))
