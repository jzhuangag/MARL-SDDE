"""Observable calibration and exact finite-budget delayed TD risk.

Only calibration observations and public dynamics/costs enter the selector.
Ground-truth signal and covariance parameters are evaluator-only arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import chi2, t


@dataclass(frozen=True)
class Calibration:
    target: float
    target_low: float
    target_high: float
    average_variance: float
    average_low: float
    average_high: float
    private_variance: float
    private_low: float
    private_high: float
    workers: int
    observations: int
    delta: float

    def variance(self, q: int) -> tuple[float, float, float]:
        if not 1 <= q <= self.workers:
            raise ValueError("q must belong to the observed worker range")
        factor = 1.0 / q - 1.0 / self.workers
        return (
            self.average_variance + factor * self.private_variance,
            self.average_low + factor * self.private_low,
            self.average_high + factor * self.private_high,
        )

    def initial_risk(self, initial: float = 0.0) -> tuple[float, float, float]:
        distances = (initial - self.target_low, initial - self.target_high)
        lower = 0.0 if self.target_low <= initial <= self.target_high else min(x*x for x in distances)
        return ((initial-self.target)**2, lower, max(x*x for x in distances))


def observe_calibration(
    rng: np.random.Generator, *, target: float, rho: float,
    persistence: float, workers: int, observations: int,
) -> np.ndarray:
    """Evaluator creates an AR(1) reward stream; selection sees only its values."""
    if not 0 <= rho <= 1 or not 0 <= persistence < 1:
        raise ValueError("invalid dependence parameters")
    noise = np.sqrt(rho)*rng.normal(size=(observations+1, 1))
    noise = noise + np.sqrt(1-rho)*rng.normal(size=(observations+1, workers))
    for k in range(1, observations+1):
        noise[k] = persistence*noise[k-1] + np.sqrt(1-persistence**2)*noise[k]
    return target + noise


def fit_calibration(values: np.ndarray, *, persistence: float, delta: float) -> Calibration:
    """Exact Gaussian finite-sample intervals, not a normal approximation.

    Public known persistence whitens the trajectory into independent innovations.
    Bonferroni combines a Student target interval and two chi-square variance
    intervals. Spatial contrasts and the spatial average are orthogonal.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 5 or values.shape[1] < 2:
        raise ValueError("need at least four innovations and two workers")
    if not np.isfinite(values).all() or not 0 <= persistence < 1 or not 0 < delta < 1:
        raise ValueError("invalid observations, persistence, or confidence")
    n, qmax = values.shape[0]-1, values.shape[1]
    innovations = (values[1:]-persistence*values[:-1])/np.sqrt(1-persistence**2)
    means = innovations.mean(axis=1)
    scale = np.sqrt((1-persistence)/(1+persistence))
    mean_var = float(means.var(ddof=1))
    private = float(np.square(innovations-means[:, None]).sum()/(n*(qmax-1)))
    alpha = delta/3.0
    target = float(means.mean()/scale)
    radius = float(t.ppf(1-alpha/2, n-1)*np.sqrt(mean_var/n)/scale)

    def variance_interval(estimate: float, df: int) -> tuple[float, float]:
        return (float(df*estimate/chi2.ppf(1-alpha/2, df)),
                float(df*estimate/chi2.ppf(alpha/2, df)))

    avg_low, avg_high = variance_interval(mean_var, n-1)
    private_low, private_high = variance_interval(private, n*(qmax-1))
    return Calibration(target, target-radius, target+radius, mean_var,
                       avg_low, avg_high, private, private_low, private_high,
                       qmax, n, delta)


@dataclass(frozen=True)
class Action:
    q: int
    spacing: int
    step: float
    updates: int
    bias_factor: float
    noise_factor: float
    messages: int
    actor_transitions: int
    parallel_ticks: int
    spectral_radius: float


def risk_coefficients(step: float, persistence: float, delay: int, updates: int) -> tuple[float, float, float]:
    """Exact lifted-state mean and covariance, including correlated innovations.

    e[k+1] = e[k] - step*e[k-delay] - step*z[k].
    Histories initially equal e[0]; z[0] has stationary unit variance.
    """
    if delay < 0 or updates < 0 or step <= 0 or not 0 <= persistence < 1:
        raise ValueError("invalid recursion")
    d = delay+2
    transition = np.zeros((d, d))
    transition[0, 0] = 1.0
    transition[0, delay] -= step
    transition[0, -1] = -step
    if delay:
        transition[1:delay+1, :delay] = np.eye(delay)
    transition[-1, -1] = persistence
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if spectral_radius >= 1-1e-10:
        raise ValueError("uncertified unstable action")
    forcing = np.zeros_like(transition)
    forcing[-1, -1] = 1-persistence**2
    stationary = solve_discrete_lyapunov(transition, forcing)
    initial_covariance = np.zeros_like(transition)
    initial_covariance[-1, -1] = 1.0
    power = np.linalg.matrix_power(transition, int(updates))
    initial_mean = np.ones(d)
    initial_mean[-1] = 0.0
    bias = float((power@initial_mean)[0]**2)
    covariance = stationary + power@(initial_covariance-stationary)@power.T
    noise = float(covariance[0, 0])
    if noise < -1e-9:
        raise ArithmeticError("negative exact covariance")
    return bias, max(0.0, noise), spectral_radius


def calibration_cost(workers: int, observations: int, overhead: int) -> tuple[int, int, int]:
    # Includes the initial stationary observation needed for whitening.
    samples = observations+1
    return samples*(overhead+workers), samples*workers, samples


def action_table(
    *, qs: Iterable[int], spacings: Iterable[int], steps: Iterable[float],
    persistence: float, delay: int, message_budget: int, actor_budget: int,
    overhead: int, probe_messages: int = 0, probe_actors: int = 0,
    probe_ticks: int = 0,
) -> list[Action]:
    actions = []
    for q in qs:
        for spacing in spacings:
            updates = min((message_budget-probe_messages)//(overhead+q),
                          (actor_budget-probe_actors)//(q*spacing))
            if updates <= 0:
                continue
            for step in steps:
                try:
                    bias, noise, radius = risk_coefficients(step, persistence**spacing, delay, updates)
                except ValueError:
                    continue
                actions.append(Action(q, spacing, step, updates, bias, noise,
                                      probe_messages+updates*(overhead+q),
                                      probe_actors+updates*q*spacing,
                                      probe_ticks+updates*spacing, radius))
    if not actions:
        raise ValueError("budgets and stability leave no action")
    return actions


def forecast(action: Action, calibration: Calibration) -> tuple[float, float, float]:
    signal, signal_low, signal_high = calibration.initial_risk()
    variance, variance_low, variance_high = calibration.variance(action.q)
    return (action.bias_factor*signal+action.noise_factor*variance,
            action.bias_factor*signal_low+action.noise_factor*variance_low,
            action.bias_factor*signal_high+action.noise_factor*variance_high)


def choose(actions: list[Action], calibration: Calibration, mode: str = "finite_horizon") -> Action:
    def score(action: Action) -> tuple[float, int, int, float]:
        if mode == "finite_horizon":
            value = forecast(action, calibration)[0]
        elif mode == "myopic":
            # Actual one-step risk with a constant initial history.
            value = (1-action.step)**2*calibration.initial_risk()[0] + action.step**2*calibration.variance(action.q)[0]
        elif mode == "correlation_only":
            value = calibration.variance(action.q)[0]/action.updates
        else:
            raise ValueError("unknown selection mode")
        return value, action.q, action.spacing, action.step
    if mode == "correlation_only":
        selected_q = min(actions, key=score).q
        # Give this comparator the same signal-aware step/spacing adaptation.
        return min((a for a in actions if a.q == selected_q), key=lambda a: (forecast(a, calibration)[0], a.spacing, a.step))
    return min(actions, key=score)


def true_risk(action: Action, *, target: float, rho: float) -> float:
    return action.bias_factor*target**2 + action.noise_factor*(rho+(1-rho)/action.q)


def selection_bound(actions: list[Action], calibration: Calibration, selected: Action) -> float:
    """True oracle regret is bounded by selected upper minus minimum lower."""
    return max(0.0, forecast(selected, calibration)[2] - min(forecast(a, calibration)[1] for a in actions))


def covers(calibration: Calibration, *, target: float, rho: float) -> bool:
    average = rho+(1-rho)/calibration.workers
    tolerance = 1e-12
    return (calibration.target_low <= target <= calibration.target_high
            and calibration.average_low-tolerance <= average <= calibration.average_high+tolerance
            and calibration.private_low-tolerance <= 1-rho <= calibration.private_high+tolerance)


def simulate_errors(
    rng: np.random.Generator, action: Action, *, target: float, rho: float,
    persistence: float, delay: int, replicates: int, checkpoints: np.ndarray,
) -> np.ndarray:
    """Fresh training streams; no calibration observation is reused in learning."""
    errors = np.full((delay+1, replicates), -target, dtype=float)
    variance = rho+(1-rho)/action.q
    noise = np.sqrt(variance)*rng.normal(size=replicates)
    retained_rate = persistence**action.spacing
    result = []
    requested = {int(k) for k in checkpoints}
    if 0 in requested:
        result.append(errors[0].copy())
    for k in range(1, action.updates+1):
        new_error = errors[0]-action.step*(errors[delay]+noise)
        if delay:
            errors[1:] = errors[:-1]
        errors[0] = new_error
        noise = retained_rate*noise + np.sqrt(variance*(1-retained_rate**2))*rng.normal(size=replicates)
        if k in requested:
            result.append(errors[0].copy())
    return np.stack(result)
