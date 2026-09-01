"""Outcome-free algebra for cross-agent gradient transport.

This module checks deterministic Taylor, radius and Lyapunov-envelope formulas.
It does not run a scientific efficacy population or consume pilot outcomes.
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
class TransportCertificate:
    proposal_norm: float
    transport_radius: float
    block_smoothness: float
    max_step: float

    def __post_init__(self) -> None:
        values = (
            self.proposal_norm,
            self.transport_radius,
            self.block_smoothness,
            self.max_step,
        )
        if min(values) < 0 or self.block_smoothness == 0:
            raise ValueError("invalid transport certificate")


@dataclass(frozen=True)
class PendingTransportDebt:
    radius: float
    path_norm: float
    hvp_radius: float
    hessian_lipschitz: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.radius,
            self.path_norm,
            self.hvp_radius,
            self.hessian_lipschitz,
            self.weight,
        )
        if min(values) < 0:
            raise ValueError("pending transport debt must be nonnegative")


def raw_staleness_radius(
    gradient_radius: float, gradient_lipschitz: float, path_norm: float
) -> float:
    """First-order stale-gradient radius."""

    if min(gradient_radius, gradient_lipschitz, path_norm) < 0:
        raise ValueError("radius inputs must be nonnegative")
    return gradient_radius+gradient_lipschitz*path_norm


def transported_radius(
    gradient_radius: float,
    hvp_radius: float,
    hessian_lipschitz: float,
    path_norm: float,
) -> float:
    """Taylor-transport radius under gradient/HVP confidence events."""

    if min(gradient_radius, hvp_radius, hessian_lipschitz, path_norm) < 0:
        raise ValueError("radius inputs must be nonnegative")
    return (
        gradient_radius
        + hvp_radius*path_norm
        + 0.5*hessian_lipschitz*path_norm*path_norm
    )


def transported_gradient(
    gradient_at_birth: Array, hessian_at_birth: Array, displacement: Array
) -> Array:
    """One-HVP Taylor transport of a joint gradient."""

    gradient_at_birth = np.asarray(gradient_at_birth, dtype=float)
    hessian_at_birth = np.asarray(hessian_at_birth, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    if hessian_at_birth.shape != (gradient_at_birth.size, displacement.size):
        raise ValueError("gradient, Hessian and displacement dimensions do not align")
    return gradient_at_birth+hessian_at_birth@displacement


def certified_gain(cert: TransportCertificate, step: float) -> float:
    """Lower bound on objective gain from a transported block proposal."""

    if not 0 <= step <= cert.max_step:
        raise ValueError("step outside certificate interval")
    signal = cert.proposal_norm
    radius = cert.transport_radius
    return (
        step*(signal*signal-radius*signal)
        - 0.5*cert.block_smoothness*step*step*signal*signal
    )


def gain_optimal_step(cert: TransportCertificate) -> float:
    """Exact maximizer of the transported certified-gain quadratic."""

    signal = cert.proposal_norm
    if signal == 0:
        return 0.0
    unconstrained = (1.0-cert.transport_radius/signal)/cert.block_smoothness
    return float(np.clip(unconstrained, 0.0, cert.max_step))


def pending_radius_increment_envelope(
    debt: PendingTransportDebt, step: float, proposal_norm: float
) -> float:
    """Upper envelope for another proposal's new Taylor remainder radius."""

    if min(step, proposal_norm) < 0:
        raise ValueError("step and proposal norm must be nonnegative")
    path_increment = step*proposal_norm
    return (
        (debt.hvp_radius+debt.hessian_lipschitz*debt.path_norm)*path_increment
        + 0.5*debt.hessian_lipschitz*path_increment*path_increment
    )


def transport_lyapunov_drift_bound(
    cert: TransportCertificate,
    pending: tuple[PendingTransportDebt, ...],
    step: float,
    *,
    potential_weight: float,
    own_weight: float = 1.0,
) -> float:
    """One-event drift envelope for transport debt minus learning progress."""

    if potential_weight <= 0 or own_weight < 0:
        raise ValueError("invalid Lyapunov weights")
    drift = -0.5*own_weight*cert.transport_radius**2
    drift -= potential_weight*certified_gain(cert, step)
    for debt in pending:
        increment = pending_radius_increment_envelope(
            debt, step, cert.proposal_norm
        )
        drift += 0.5*debt.weight*((debt.radius+increment)**2-debt.radius**2)
    return float(drift)


def _transport_drift_derivative(
    cert: TransportCertificate,
    pending: tuple[PendingTransportDebt, ...],
    step: float,
    potential_weight: float,
) -> float:
    signal = cert.proposal_norm
    derivative = -potential_weight*(
        signal*signal-cert.transport_radius*signal
        - cert.block_smoothness*step*signal*signal
    )
    for debt in pending:
        linear = (
            debt.hvp_radius+debt.hessian_lipschitz*debt.path_norm
        )*signal
        quadratic = 0.5*debt.hessian_lipschitz*signal*signal
        increment = linear*step+quadratic*step*step
        derivative += debt.weight*(debt.radius+increment)*(
            linear+2.0*quadratic*step
        )
    return float(derivative)


def lyapunov_optimal_step(
    cert: TransportCertificate,
    pending: tuple[PendingTransportDebt, ...],
    *,
    potential_weight: float,
    iterations: int = 80,
) -> float:
    """Exact-to-tolerance minimizer of the scalar convex drift envelope.

    The derivative is monotone on the nonnegative interval because every
    pending-radius coefficient is nonnegative.  Bisection is therefore a
    continuous optimization, not a catalogue scan.
    """

    if potential_weight <= 0 or iterations <= 0:
        raise ValueError("invalid optimization controls")
    if cert.proposal_norm == 0 or cert.max_step == 0:
        return 0.0
    derivative_zero = _transport_drift_derivative(
        cert, pending, 0.0, potential_weight
    )
    if derivative_zero >= 0:
        return 0.0
    derivative_cap = _transport_drift_derivative(
        cert, pending, cert.max_step, potential_weight
    )
    if derivative_cap <= 0:
        return cert.max_step
    lower = 0.0
    upper = cert.max_step
    for _ in range(iterations):
        midpoint = 0.5*(lower+upper)
        derivative = _transport_drift_derivative(
            cert, pending, midpoint, potential_weight
        )
        if derivative <= 0:
            lower = midpoint
        else:
            upper = midpoint
    return float(0.5*(lower+upper))


def equicoupled_matrix(n_agents: int, coupling: float) -> Array:
    if n_agents < 2 or not 0 <= coupling < 1:
        raise ValueError("invalid equicoupled system")
    matrix = np.full((n_agents, n_agents), coupling, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def quadratic_gradient(theta: Array, matrix: Array) -> Array:
    return -np.asarray(matrix, dtype=float)@np.asarray(theta, dtype=float)


def quadratic_hessian(matrix: Array) -> Array:
    return -np.asarray(matrix, dtype=float)


def quartic_gradient(theta: Array, matrix: Array, quartic: float) -> Array:
    theta = np.asarray(theta, dtype=float)
    return -np.asarray(matrix, dtype=float)@theta-quartic*theta**3


def quartic_hessian(theta: Array, matrix: Array, quartic: float) -> Array:
    theta = np.asarray(theta, dtype=float)
    return -np.asarray(matrix, dtype=float)-3.0*quartic*np.diag(theta**2)


def audit() -> dict[str, object]:
    exact_quadratic_checks = 0
    nonlinear_radius_checks = 0
    transport_improvement_checks = 0
    optimizer_checks = 0
    maximum_quadratic_error = 0.0
    minimum_nonlinear_slack = float("inf")
    maximum_optimizer_gap = 0.0

    rng = np.random.default_rng(20260901)
    for n_agents, coupling in product((3, 5), (0.05, 0.45, 0.85)):
        matrix = equicoupled_matrix(n_agents, coupling)
        for _ in range(64):
            birth = rng.uniform(-0.8, 0.8, size=n_agents)
            displacement = rng.uniform(-0.2, 0.2, size=n_agents)
            current = birth+displacement
            transported = transported_gradient(
                quadratic_gradient(birth, matrix),
                quadratic_hessian(matrix),
                displacement,
            )
            error = float(np.linalg.norm(
                quadratic_gradient(current, matrix)-transported
            ))
            maximum_quadratic_error = max(maximum_quadratic_error, error)
            if error > 1e-12:
                raise AssertionError("quadratic gradient transport was not exact")
            exact_quadratic_checks += 1

    quartic = 0.08
    domain_bound = 1.0
    hessian_lipschitz = 6.0*quartic*domain_bound
    for n_agents, coupling in product((3, 5), (0.05, 0.45, 0.85)):
        matrix = equicoupled_matrix(n_agents, coupling)
        gradient_lipschitz = float(
            np.linalg.norm(matrix, ord=2)+3.0*quartic*domain_bound**2
        )
        for _ in range(96):
            birth = rng.uniform(-0.7, 0.7, size=n_agents)
            displacement = rng.uniform(-0.08, 0.08, size=n_agents)
            current = birth+displacement
            if np.max(np.abs(current)) > domain_bound:
                raise AssertionError("nonlinear audit left the declared domain")
            transported = transported_gradient(
                quartic_gradient(birth, matrix, quartic),
                quartic_hessian(birth, matrix, quartic),
                displacement,
            )
            error = float(np.linalg.norm(
                quartic_gradient(current, matrix, quartic)-transported
            ))
            path_norm = float(np.linalg.norm(displacement))
            radius = transported_radius(0.0, 0.0, hessian_lipschitz, path_norm)
            slack = radius-error
            if slack < -1e-12:
                raise AssertionError("Taylor transport radius was violated")
            minimum_nonlinear_slack = min(minimum_nonlinear_slack, slack)
            raw = raw_staleness_radius(0.0, gradient_lipschitz, path_norm)
            if radius < raw:
                transport_improvement_checks += 1
            nonlinear_radius_checks += 1

    for signal, radius, smoothness, max_step, potential_weight in product(
        (0.1, 0.5, 1.5),
        (0.0, 0.04, 0.2),
        (0.5, 1.0, 2.0),
        (0.25, 0.75, 1.0),
        (1.0, 4.0),
    ):
        cert = TransportCertificate(signal, radius, smoothness, max_step)
        pending = (
            PendingTransportDebt(0.03, 0.1, 0.02, 0.1),
            PendingTransportDebt(0.08, 0.4, 0.04, 0.3, weight=1.5),
        )
        optimum = lyapunov_optimal_step(
            cert, pending, potential_weight=potential_weight
        )
        grid = np.linspace(0.0, max_step, 20_001)
        values = np.asarray([
            transport_lyapunov_drift_bound(
                cert, pending, float(step), potential_weight=potential_weight
            )
            for step in grid
        ])
        numeric = float(grid[int(np.argmin(values))])
        maximum_optimizer_gap = max(maximum_optimizer_gap, abs(optimum-numeric))
        optimizer_checks += 1

    if transport_improvement_checks != nonlinear_radius_checks:
        raise AssertionError("transport did not improve every declared local radius")
    return {
        "kind": "cross_agent_transport_algebra_not_efficacy",
        "exact_quadratic_checks": exact_quadratic_checks,
        "maximum_quadratic_transport_error": maximum_quadratic_error,
        "nonlinear_radius_checks": nonlinear_radius_checks,
        "minimum_nonlinear_radius_slack": minimum_nonlinear_slack,
        "transport_radius_improvement_checks": transport_improvement_checks,
        "lyapunov_optimizer_checks": optimizer_checks,
        "maximum_optimizer_grid_gap": maximum_optimizer_gap,
        "scientific_population_generated": False,
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
