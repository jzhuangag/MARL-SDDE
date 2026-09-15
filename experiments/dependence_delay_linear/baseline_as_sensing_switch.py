"""Baseline-as-sensing, anytime likelihood switch for the Gaussian model.

This module is deliberately a small theorem-facing prototype.  It implements
the binary, finite-commit policy in
``docs/baseline_as_sensing_anytime_switch_20260905.md``; it is neither an
asynchronous-MARL runner nor a performance experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from adaptive_change_of_measure import (
    AdaptiveAction,
    GaussianFilter,
    adaptive_log_likelihood_ratio,
    scalar_normal_log_density,
)


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def anytime_threshold(*, switch_loss_upper: float, safety_budget: float) -> float:
    """Return ``log(W0 / epsilon)`` or infinity for exact no-harm.

    The theorem requires ``0 < epsilon < W0`` for a finite nontrivial
    threshold.  ``epsilon=0`` deliberately disables switching rather than
    pretending that a finite Gaussian likelihood threshold can ensure exact
    no-harm.
    """

    loss = float(switch_loss_upper)
    epsilon = float(safety_budget)
    if not math.isfinite(loss) or loss <= 0.0:
        raise ValueError("switch_loss_upper must be finite and positive")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("safety_budget must be finite and nonnegative")
    if epsilon == 0.0:
        return math.inf
    if epsilon >= loss:
        raise ValueError("finite anytime threshold requires safety_budget < switch_loss_upper")
    return math.log(loss / epsilon)


@dataclass(frozen=True)
class PacketLikelihood:
    """One sequential likelihood calculation and its irreversible decision."""

    packet_index: int
    increment: float
    cumulative_log_likelihood_ratio: float
    crossed: bool
    switched: bool


class BaselineAsSensingSwitch:
    """Use baseline packets as evidence and make at most one regime-one switch.

    The likelihood ratio is ``log p_1/p_0``.  The first packet is evaluated
    from the stationary prior; later packets propagate both public filters by
    the realized action's stride before evaluating their Gaussian innovations.
    """

    def __init__(
        self,
        *,
        theta_low: float,
        theta_high: float,
        mixing: float,
        threshold: float,
        cutoff_packets: int,
    ) -> None:
        low = _finite_nonnegative(theta_low, "theta_low")
        high = _finite_nonnegative(theta_high, "theta_high")
        rho = float(mixing)
        level = float(threshold)
        if low == high:
            raise ValueError("theta hypotheses must be distinct")
        if not math.isfinite(rho) or not 0.0 <= rho < 1.0:
            raise ValueError("mixing must be finite in [0, 1)")
        if not (math.isfinite(level) or math.isinf(level)) or level < 0.0:
            raise ValueError("threshold must be nonnegative and finite or infinity")
        if not isinstance(cutoff_packets, int) or isinstance(cutoff_packets, bool) or cutoff_packets < 0:
            raise ValueError("cutoff_packets must be a nonnegative integer")
        self._theta_low = low
        self._theta_high = high
        self._mixing = rho
        self._threshold = level
        self._cutoff_packets = cutoff_packets
        self._low_filter = GaussianFilter(low, rho)
        self._high_filter = GaussianFilter(high, rho)
        self._cumulative = 0.0
        self._packets = 0
        self._switched_at: int | None = None

    @property
    def cumulative_log_likelihood_ratio(self) -> float:
        return self._cumulative

    @property
    def switched_at(self) -> int | None:
        return self._switched_at

    @property
    def switched(self) -> bool:
        return self._switched_at is not None

    def observe_baseline_packet(
        self, *, observation: float, action: AdaptiveAction
    ) -> PacketLikelihood:
        """Process one charged baseline packet before the switch cutoff.

        The caller owns the actual baseline learning update.  This method only
        performs the two scalar Kalman likelihood calculations and never
        changes the action or charges a new sample.
        """

        if self._packets >= self._cutoff_packets:
            raise RuntimeError("no baseline packet may be processed after cutoff")
        value = float(observation)
        if not math.isfinite(value):
            raise ValueError("observation must be finite")
        if not isinstance(action, AdaptiveAction):
            raise TypeError("action must be an AdaptiveAction")
        if self._packets:
            self._low_filter.propagate(action.b)
            self._high_filter.propagate(action.b)
        low_residual, low_variance = self._low_filter.innovation(value, action.q)
        high_residual, high_variance = self._high_filter.innovation(value, action.q)
        increment = scalar_normal_log_density(
            high_residual, high_variance
        ) - scalar_normal_log_density(low_residual, low_variance)
        self._low_filter.update(value, action.q)
        self._high_filter.update(value, action.q)
        self._packets += 1
        self._cumulative += increment
        crossed = self._cumulative >= self._threshold
        if crossed and self._switched_at is None:
            self._switched_at = self._packets
        return PacketLikelihood(
            packet_index=self._packets,
            increment=increment,
            cumulative_log_likelihood_ratio=self._cumulative,
            crossed=crossed,
            switched=self.switched,
        )


@dataclass(frozen=True)
class HighRegimeRiskBound:
    """The finite-cutoff Chernoff/tail-sum upper bound from Theorem 2."""

    detection_scale: int | float
    expected_detection_time_upper: float
    cutoff_miss_probability_upper: float
    regime_one_regret_upper: float


def high_regime_chernoff_risk_bound(
    *,
    threshold: float,
    chernoff_s: float,
    information_rate: float,
    initialization_constant: float,
    cutoff_packets: int,
    delay_and_inflight_loss: float,
    per_packet_opportunity_loss: float,
    no_switch_loss: float,
) -> HighRegimeRiskBound:
    """Compute the conservative Theorem-2 risk upper bound.

    Probability terms are capped at one, which only strengthens the displayed
    Chernoff bound when its exponential prefactor is larger than one.
    """

    h = float(threshold)
    s = float(chernoff_s)
    rate = float(information_rate)
    init = _finite_nonnegative(initialization_constant, "initialization_constant")
    delay = _finite_nonnegative(delay_and_inflight_loss, "delay_and_inflight_loss")
    omega = _finite_nonnegative(per_packet_opportunity_loss, "per_packet_opportunity_loss")
    no_switch = _finite_nonnegative(no_switch_loss, "no_switch_loss")
    if not (math.isfinite(h) or math.isinf(h)) or h < 0.0:
        raise ValueError("threshold must be nonnegative and finite or infinity")
    if not math.isfinite(s) or not 0.0 < s < 1.0:
        raise ValueError("chernoff_s must be in (0, 1)")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("information_rate must be finite and positive")
    if not isinstance(cutoff_packets, int) or isinstance(cutoff_packets, bool) or cutoff_packets < 0:
        raise ValueError("cutoff_packets must be a nonnegative integer")
    if math.isinf(h):
        return HighRegimeRiskBound(
            detection_scale=math.inf,
            expected_detection_time_upper=math.inf,
            cutoff_miss_probability_upper=1.0,
            regime_one_regret_upper=math.inf if omega > 0.0 else delay + no_switch,
        )
    detection_scale = math.ceil((s * h + init) / rate)
    expected_time = detection_scale + 1.0 / (1.0 - math.exp(-rate))
    exponent = s * h + init - cutoff_packets * rate
    miss = min(1.0, math.exp(exponent))
    regret = delay + omega * expected_time + no_switch * miss
    return HighRegimeRiskBound(
        detection_scale=detection_scale,
        expected_detection_time_upper=expected_time,
        cutoff_miss_probability_upper=miss,
        regime_one_regret_upper=regret,
    )


def bretagnolle_huber_safety_slack_lower_bound(
    *,
    low_regime_wrong_deployment_gap: float,
    high_regime_wrong_deployment_gap: float,
    maximum_kl: float,
    high_regime_regret: float,
) -> float:
    """Return Corollary-1's finite-budget expected-safety lower bound.

    This is ``g0 [0.5 exp(-Kmax) - r/g1]_+``.  It is a lower bound on
    necessary expected low-regime slack, not a guarantee of the switch above.
    """

    g0 = float(low_regime_wrong_deployment_gap)
    g1 = float(high_regime_wrong_deployment_gap)
    kl = _finite_nonnegative(maximum_kl, "maximum_kl")
    regret = _finite_nonnegative(high_regime_regret, "high_regime_regret")
    if not math.isfinite(g0) or g0 <= 0.0:
        raise ValueError("low_regime_wrong_deployment_gap must be finite and positive")
    if not math.isfinite(g1) or g1 <= 0.0:
        raise ValueError("high_regime_wrong_deployment_gap must be finite and positive")
    return g0 * max(0.0, 0.5 * math.exp(-kl) - regret / g1)


def exact_cumulative_log_likelihood_ratio(
    *, observations: list[float], actions: list[AdaptiveAction], theta_low: float,
    theta_high: float, mixing: float,
) -> float:
    """Expose the reference batch identity used by prototype tests."""

    return -adaptive_log_likelihood_ratio(
        observations, actions, theta_low, theta_high, mixing
    )
