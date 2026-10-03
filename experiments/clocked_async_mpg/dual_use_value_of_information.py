"""Causal one-step value of information for a dual-use optimistic oracle."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .dual_use_fingerprint import (
    BinaryGeometryBelief,
    expected_binary_log_gain,
    predict_binary_geometry,
    update_binary_geometry,
)


@dataclass(frozen=True)
class DualUseLookaheadDecision:
    use_optimism: bool
    immediate_expected_log_gain: float
    no_call_objective: float
    call_objective: float
    no_observation_next_value: float
    observation_next_value: float
    same_debt_information_value: float
    resource_debt_if_no_call: float
    resource_debt_if_call: float


def _myopic_relative_value(
    rotation_probability: float,
    *,
    potential_log_gain: float,
    rotational_log_gain: float,
    resource_debt: float,
    lyapunov_tradeoff: float,
) -> float:
    gain = expected_binary_log_gain(
        BinaryGeometryBelief(rotation_probability),
        potential_log_gain=potential_log_gain,
        rotational_log_gain=rotational_log_gain,
    )
    return min(0.0, -gain + resource_debt / lyapunov_tradeoff)


def binary_gaussian_expectation_after_observation(
    belief: BinaryGeometryBelief,
    *,
    observation_standard_deviation: float,
    value: Callable[[BinaryGeometryBelief], float],
    quadrature_order: int = 17,
) -> float:
    """Integrate a posterior value under the declared binary Gaussian model.

    Potential and rotation scores have means -1 and +1.  Gauss--Hermite
    quadrature is deterministic, so this routine introduces no simulation
    noise into a controller decision.
    """

    probability = belief.rotation_probability
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("belief probability must lie in [0, 1]")
    if (
        not math.isfinite(observation_standard_deviation)
        or observation_standard_deviation <= 0.0
    ):
        raise ValueError("observation standard deviation must be positive")
    if quadrature_order < 3:
        raise ValueError("quadrature order must be at least three")
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    total = 0.0
    for phase_probability, mean in ((1.0 - probability, -1.0), (probability, 1.0)):
        if phase_probability == 0.0:
            continue
        phase_value = 0.0
        for node, weight in zip(nodes, weights):
            observed = mean + math.sqrt(2.0) * observation_standard_deviation * node
            posterior = update_binary_geometry(
                belief,
                observed_score=float(observed),
                observation_standard_deviation=observation_standard_deviation,
            )
            phase_value += float(weight) * float(value(posterior))
        total += phase_probability * phase_value / math.sqrt(math.pi)
    return float(total)


def binary_gaussian_posterior_hinge(
    belief: BinaryGeometryBelief,
    *,
    observation_standard_deviation: float,
    intercept: float,
    posterior_slope: float,
) -> float:
    """Exactly integrate ``min(0, intercept-slope*posterior)``.

    The posterior is monotone in the Gaussian score.  Moreover,
    ``E[p(Y) 1_A] = P(H=1, A)`` for every observation event ``A``.  A posterior
    hinge therefore reduces to two univariate Gaussian tail probabilities;
    no quadrature or belief grid is required.
    """

    probability = belief.rotation_probability
    values = (probability, observation_standard_deviation, intercept, posterior_slope)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("posterior-hinge inputs must be finite")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("belief probability must lie in [0, 1]")
    if observation_standard_deviation <= 0.0:
        raise ValueError("observation standard deviation must be positive")
    if posterior_slope == 0.0 or probability in (0.0, 1.0):
        return min(0.0, intercept - posterior_slope * probability)

    threshold = intercept / posterior_slope
    if posterior_slope > 0.0:
        if threshold <= 0.0:
            return float(intercept - posterior_slope * probability)
        if threshold >= 1.0:
            return 0.0
        upper_event = True
    else:
        if threshold <= 0.0:
            return 0.0
        if threshold >= 1.0:
            return float(intercept - posterior_slope * probability)
        upper_event = False

    logit_threshold = math.log(threshold / (1.0 - threshold))
    logit_prior = math.log(probability / (1.0 - probability))
    score_threshold = (
        0.5
        * observation_standard_deviation**2
        * (logit_threshold - logit_prior)
    )

    def normal_cdf(value: float) -> float:
        return 0.5 * math.erfc(-value / math.sqrt(2.0))

    z_potential = (score_threshold + 1.0) / observation_standard_deviation
    z_rotation = (score_threshold - 1.0) / observation_standard_deviation
    if upper_event:
        phase_zero_probability = 1.0 - normal_cdf(z_potential)
        phase_one_probability = 1.0 - normal_cdf(z_rotation)
    else:
        phase_zero_probability = normal_cdf(z_potential)
        phase_one_probability = normal_cdf(z_rotation)
    event_probability = (
        (1.0 - probability) * phase_zero_probability
        + probability * phase_one_probability
    )
    posterior_truncated_moment = probability * phase_one_probability
    return float(
        intercept * event_probability
        - posterior_slope * posterior_truncated_moment
    )


def choose_dual_use_lookahead(
    belief: BinaryGeometryBelief,
    *,
    potential_to_rotation: float,
    rotation_to_potential: float,
    potential_log_gain: float,
    rotational_log_gain: float,
    observation_standard_deviation: float,
    resource_debt: float,
    average_optimism_budget: float,
    lyapunov_tradeoff: float,
    hard_feasible: bool = True,
) -> DualUseLookaheadDecision:
    """Choose a paid optimistic query using control and information value.

    The current call is priced by its immediate expected log-drift benefit and
    by its resource debt.  Because that same call reveals a fingerprint for
    the next event, the decision also includes one exact belief-step of
    lookahead under the declared binary Gaussian emission model.  The current
    action never uses the observation it may generate.
    """

    scalars = (
        potential_log_gain,
        rotational_log_gain,
        resource_debt,
        average_optimism_budget,
        lyapunov_tradeoff,
    )
    if any(not math.isfinite(value) for value in scalars):
        raise ValueError("controller inputs must be finite")
    if resource_debt < 0.0 or lyapunov_tradeoff <= 0.0:
        raise ValueError("debt must be nonnegative and tradeoff positive")
    if not 0.0 <= average_optimism_budget <= 1.0:
        raise ValueError("average optimism budget must lie in [0, 1]")

    immediate_gain = expected_binary_log_gain(
        belief,
        potential_log_gain=potential_log_gain,
        rotational_log_gain=rotational_log_gain,
    )
    debt_no_call = max(0.0, resource_debt - average_optimism_budget)
    debt_call = max(0.0, resource_debt + 1.0 - average_optimism_budget)

    transition_intercept = potential_to_rotation
    transition_slope = 1.0 - potential_to_rotation - rotation_to_potential
    gain_intercept = (
        potential_log_gain
        + (rotational_log_gain - potential_log_gain) * transition_intercept
    )
    gain_slope = (
        (rotational_log_gain - potential_log_gain) * transition_slope
    )

    def next_value(posterior: BinaryGeometryBelief, debt: float) -> float:
        predicted = predict_binary_geometry(
            posterior,
            potential_to_rotation=potential_to_rotation,
            rotation_to_potential=rotation_to_potential,
        )
        return _myopic_relative_value(
            predicted.rotation_probability,
            potential_log_gain=potential_log_gain,
            rotational_log_gain=rotational_log_gain,
            resource_debt=debt,
            lyapunov_tradeoff=lyapunov_tradeoff,
        )

    no_observation_next = next_value(belief, debt_no_call)
    observation_next = binary_gaussian_posterior_hinge(
        belief,
        observation_standard_deviation=observation_standard_deviation,
        intercept=debt_call / lyapunov_tradeoff - gain_intercept,
        posterior_slope=gain_slope,
    )
    same_debt_observation = binary_gaussian_posterior_hinge(
        belief,
        observation_standard_deviation=observation_standard_deviation,
        intercept=debt_no_call / lyapunov_tradeoff - gain_intercept,
        posterior_slope=gain_slope,
    )
    information_value = no_observation_next - same_debt_observation
    if information_value < -1e-10:
        raise RuntimeError("Bayesian information value must be nonnegative")
    information_value = max(0.0, information_value)

    no_call_objective = no_observation_next
    call_objective = (
        -immediate_gain
        + resource_debt / lyapunov_tradeoff
        + observation_next
    )
    use_optimism = bool(hard_feasible and call_objective < no_call_objective)
    return DualUseLookaheadDecision(
        use_optimism=use_optimism,
        immediate_expected_log_gain=immediate_gain,
        no_call_objective=no_call_objective,
        call_objective=call_objective,
        no_observation_next_value=no_observation_next,
        observation_next_value=observation_next,
        same_debt_information_value=information_value,
        resource_debt_if_no_call=debt_no_call,
        resource_debt_if_call=debt_call,
    )
