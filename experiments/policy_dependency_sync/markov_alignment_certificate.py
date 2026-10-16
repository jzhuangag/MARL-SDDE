"""Finite-state Markov certificates for launch-time alignment decisions.

The joint Lyapunov controller needs a lower bound on the conditional
alignment of a packet that will be generated after a cache action.  This
module separates two cases:

* a model-based finite-horizon value, which preserves the current Markov
  state and is therefore suitable for genuinely state-adaptive decisions;
* a fixed-event holdout lower confidence bound, which is valid for an
  arbitrary initial state but bridges through stationary laws and can be
  conservative.

The holdout score functions and candidate kernels must be fixed before the
reference path is generated.  This predictability condition is deliberate:
reusing a path both to fit an unrestricted score and to certify that score is
not covered by the bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
from typing import Sequence

import numpy as np

from experiments.dependence_delay_linear.t042_poisson_td_remainder import (
    solve_centered_poisson,
    stationary_distribution,
)
from experiments.dependence_delay_linear.t036_affine_markov_moments import (
    validate_markov_transition,
)


@dataclass(frozen=True)
class AlignmentLowerCertificate:
    lower_bound: float
    sample_mean: float
    reference_concentration_penalty: float
    target_transient_penalty: float
    stationary_shift_penalty: float
    approximation_penalty: float
    event_failure_probability: float


@dataclass(frozen=True)
class RobustConditionalAlignmentCertificate:
    lower_bound: float
    identified: bool
    robust_value_by_state: tuple[float, ...]
    score_radius_by_state: tuple[float, ...]
    transition_l1_radius_by_state: tuple[float, ...]
    event_failure_probability: float


def summable_event_failure_probability(
    *, total_failure_probability: float, event_index: int
) -> float:
    """Allocate ``delta`` over infinitely many decision events.

    The allocation is ``6 delta / (pi^2 (p+1)^2)`` and therefore sums to
    ``delta``.  A union bound can consequently make all fixed-event
    certificates simultaneous without calling a fixed-event statement
    "time-uniform".
    """

    if not 0.0 < total_failure_probability < 1.0:
        raise ValueError("total failure probability must lie in (0,1)")
    if event_index < 0:
        raise ValueError("event index must be nonnegative")
    return float(
        6.0
        * total_failure_probability
        / (pi**2 * float(event_index + 1) ** 2)
    )


def dobrushin_contraction(transition: np.ndarray) -> float:
    """Return the exact total-variation contraction coefficient of a kernel."""

    probability = validate_markov_transition(np.asarray(transition, dtype=float))
    pairwise = 0.5 * np.abs(
        probability[:, None, :] - probability[None, :, :]
    ).sum(axis=2)
    return float(np.max(pairwise))


def scalar_poisson_span(transition: np.ndarray, score_by_state: Sequence[float]) -> float:
    """Compute the canonical Poisson-solution span for a scalar score."""

    probability = validate_markov_transition(np.asarray(transition, dtype=float))
    score = np.asarray(score_by_state, dtype=float)
    if score.shape != (probability.shape[0],) or not np.all(np.isfinite(score)):
        raise ValueError("one finite score is required per Markov state")
    stationary = stationary_distribution(probability)
    centered = score - float(stationary @ score)
    solution = solve_centered_poisson(
        transition=probability,
        field=centered,
        stationary=stationary,
    )["solution"]
    return float(np.max(solution) - np.min(solution))


def finite_horizon_markov_alignment(
    *,
    transition: np.ndarray,
    score_by_state: Sequence[float],
    initial_state: int,
    horizon: int,
) -> float:
    """Exact expected average score of a future Markov packet."""

    probability = validate_markov_transition(np.asarray(transition, dtype=float))
    score = np.asarray(score_by_state, dtype=float)
    states = probability.shape[0]
    if score.shape != (states,) or not np.all(np.isfinite(score)):
        raise ValueError("one finite score is required per Markov state")
    if not 0 <= initial_state < states or horizon <= 0:
        raise ValueError("invalid initial state or horizon")
    law = np.zeros(states, dtype=float)
    law[initial_state] = 1.0
    total = 0.0
    for _ in range(horizon):
        total += float(law @ score)
        law = law @ probability
    return float(total / horizon)


def stationary_total_variation(
    reference_transition: np.ndarray, target_transition: np.ndarray
) -> float:
    """Exact TV distance between two finite-state stationary laws."""

    reference = stationary_distribution(reference_transition)
    target = stationary_distribution(target_transition)
    return float(0.5 * np.abs(reference - target).sum())


def stationary_shift_upper_from_kernel_shift(
    *, kernel_shift_total_variation: float, target_contraction_upper: float
) -> float:
    """Perturbation bound ``TV(pi_0,pi_a)<=eps/(1-rho_a)``."""

    if not 0.0 <= kernel_shift_total_variation <= 1.0:
        raise ValueError("kernel shift must lie in [0,1]")
    if not 0.0 <= target_contraction_upper < 1.0:
        raise ValueError("target contraction upper bound must lie in [0,1)")
    return float(
        min(
            1.0,
            kernel_shift_total_variation / (1.0 - target_contraction_upper),
        )
    )


def poisson_holdout_alignment_lower_bound(
    *,
    score_samples: Sequence[float],
    reference_poisson_span: float,
    target_poisson_span: float,
    target_score_oscillation: float,
    stationary_shift_total_variation_upper: float,
    future_horizon: int,
    action_count: int,
    event_failure_probability: float,
    approximation_error: float = 0.0,
) -> AlignmentLowerCertificate:
    """Lower-bound a future packet's conditional expected alignment.

    A completed reference path provides ``score_samples``.  The reference
    Poisson span controls its Markov concentration from any initial state.
    Stationary-law shift and the target Poisson span then bridge to a future
    packet generated by the candidate kernel from its observed launch state.
    The statement is simultaneous over ``action_count`` candidates at this
    event by an ordinary union bound.
    """

    samples = np.asarray(score_samples, dtype=float)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("score_samples must be a nonempty finite vector")
    nonnegative = (
        reference_poisson_span,
        target_poisson_span,
        target_score_oscillation,
        stationary_shift_total_variation_upper,
        approximation_error,
    )
    if min(nonnegative) < 0.0:
        raise ValueError("certificate radii must be nonnegative")
    if stationary_shift_total_variation_upper > 1.0:
        raise ValueError("stationary shift upper bound cannot exceed one")
    if future_horizon <= 0 or action_count <= 0:
        raise ValueError("future horizon and action count must be positive")
    if not 0.0 < event_failure_probability < 1.0:
        raise ValueError("event failure probability must lie in (0,1)")

    sample_count = int(samples.size)
    logarithm = log(float(action_count) / event_failure_probability)
    reference_penalty = float(
        reference_poisson_span
        * (1.0 / sample_count + sqrt(logarithm / (2.0 * sample_count)))
    )
    target_penalty = float(target_poisson_span / future_horizon)
    shift_penalty = float(
        target_score_oscillation * stationary_shift_total_variation_upper
    )
    approximation_penalty = float(approximation_error)
    sample_mean = float(samples.mean())
    lower = (
        sample_mean
        - reference_penalty
        - target_penalty
        - shift_penalty
        - approximation_penalty
    )
    return AlignmentLowerCertificate(
        lower_bound=float(lower),
        sample_mean=sample_mean,
        reference_concentration_penalty=reference_penalty,
        target_transient_penalty=target_penalty,
        stationary_shift_penalty=shift_penalty,
        approximation_penalty=approximation_penalty,
        event_failure_probability=float(event_failure_probability),
    )


def exact_finite_state_holdout_lower_bound(
    *,
    reference_transition: np.ndarray,
    target_transition: np.ndarray,
    score_by_state: Sequence[float],
    reference_state_path: Sequence[int],
    future_horizon: int,
    action_count: int,
    event_failure_probability: float,
    approximation_error: float = 0.0,
) -> AlignmentLowerCertificate:
    """Build the certificate using exact finite-state Poisson constants."""

    reference = validate_markov_transition(
        np.asarray(reference_transition, dtype=float)
    )
    target = validate_markov_transition(np.asarray(target_transition, dtype=float))
    if reference.shape != target.shape:
        raise ValueError("reference and target kernels must have matching shapes")
    score = np.asarray(score_by_state, dtype=float)
    states = reference.shape[0]
    path = np.asarray(reference_state_path, dtype=int)
    if score.shape != (states,) or not np.all(np.isfinite(score)):
        raise ValueError("one finite score is required per state")
    if path.ndim != 1 or path.size == 0 or np.any((path < 0) | (path >= states)):
        raise ValueError("reference path must contain valid states")
    return poisson_holdout_alignment_lower_bound(
        score_samples=score[path],
        reference_poisson_span=scalar_poisson_span(reference, score),
        target_poisson_span=scalar_poisson_span(target, score),
        target_score_oscillation=float(np.max(score) - np.min(score)),
        stationary_shift_total_variation_upper=stationary_total_variation(
            reference, target
        ),
        future_horizon=future_horizon,
        action_count=action_count,
        event_failure_probability=event_failure_probability,
        approximation_error=approximation_error,
    )


def dobrushin_holdout_alignment_lower_bound(
    *,
    score_samples: Sequence[float],
    score_oscillation_upper: float,
    reference_contraction_upper: float,
    target_contraction_upper: float,
    kernel_shift_total_variation_upper: float,
    future_horizon: int,
    action_count: int,
    event_failure_probability: float,
    approximation_error: float = 0.0,
) -> AlignmentLowerCertificate:
    """Model-free algebra given public contraction and kernel-shift bounds."""

    if score_oscillation_upper < 0.0:
        raise ValueError("score oscillation upper bound must be nonnegative")
    if not 0.0 <= reference_contraction_upper < 1.0:
        raise ValueError("reference contraction upper bound must lie in [0,1)")
    if not 0.0 <= target_contraction_upper < 1.0:
        raise ValueError("target contraction upper bound must lie in [0,1)")
    stationary_shift = stationary_shift_upper_from_kernel_shift(
        kernel_shift_total_variation=kernel_shift_total_variation_upper,
        target_contraction_upper=target_contraction_upper,
    )
    return poisson_holdout_alignment_lower_bound(
        score_samples=score_samples,
        reference_poisson_span=(
            score_oscillation_upper / (1.0 - reference_contraction_upper)
        ),
        target_poisson_span=(
            score_oscillation_upper / (1.0 - target_contraction_upper)
        ),
        target_score_oscillation=score_oscillation_upper,
        stationary_shift_total_variation_upper=stationary_shift,
        future_horizon=future_horizon,
        action_count=action_count,
        event_failure_probability=event_failure_probability,
        approximation_error=approximation_error,
    )


def worst_case_simplex_expectation_l1(
    *, nominal_probability: Sequence[float], values: Sequence[float], l1_radius: float
) -> float:
    """Minimize a linear value over a simplex intersected with an L1 ball.

    Probability mass is moved greedily from the largest-value states to the
    smallest-value states.  This is the exact transportation solution for a
    scalar objective and costs ``O(S log S)`` because of the sort.
    """

    nominal = np.asarray(nominal_probability, dtype=float)
    value = np.asarray(values, dtype=float)
    if (
        nominal.ndim != 1
        or value.shape != nominal.shape
        or nominal.size == 0
        or not np.all(np.isfinite(nominal))
        or not np.all(np.isfinite(value))
        or np.any(nominal < 0.0)
        or not np.isclose(nominal.sum(), 1.0, atol=1e-10)
    ):
        raise ValueError("nominal probabilities and values are invalid")
    if not 0.0 <= l1_radius <= 2.0:
        raise ValueError("an L1 probability radius must lie in [0,2]")

    probability = nominal.copy()
    remaining = min(float(l1_radius) / 2.0, 1.0)
    low_order = np.argsort(value, kind="stable")
    high_order = low_order[::-1]
    low_pointer = high_pointer = 0
    tolerance = 1e-15
    while remaining > tolerance:
        while (
            low_pointer < nominal.size
            and probability[low_order[low_pointer]] >= 1.0 - tolerance
        ):
            low_pointer += 1
        while (
            high_pointer < nominal.size
            and probability[high_order[high_pointer]] <= tolerance
        ):
            high_pointer += 1
        if low_pointer >= nominal.size or high_pointer >= nominal.size:
            break
        receiver = int(low_order[low_pointer])
        donor = int(high_order[high_pointer])
        if receiver == donor or value[donor] <= value[receiver] + tolerance:
            break
        moved = min(
            remaining,
            float(probability[donor]),
            float(1.0 - probability[receiver]),
        )
        probability[donor] -= moved
        probability[receiver] += moved
        remaining -= moved
    return float(probability @ value)


def anytime_bounded_mean_radius(
    *,
    sample_count: int,
    value_range: float,
    family_size: int,
    event_failure_probability: float,
) -> float:
    """Two-sided Hoeffding radius uniform over every positive sample count.

    Failure mass is split across the declared finite family and counts with
    weight ``1/(n(n+1))``.  The radius is intentionally elementary and can be
    replaced by a sharper confidence sequence without changing the robust
    dynamic-programming interface.
    """

    if sample_count <= 0 or family_size <= 0 or value_range < 0.0:
        raise ValueError("sample count/family must be positive and range nonnegative")
    if not 0.0 < event_failure_probability < 1.0:
        raise ValueError("event failure probability must lie in (0,1)")
    logarithm = log(
        2.0
        * float(family_size)
        * float(sample_count)
        * float(sample_count + 1)
        / event_failure_probability
    )
    return float(value_range * sqrt(logarithm / (2.0 * sample_count)))


def robust_finite_horizon_markov_alignment(
    *,
    nominal_transition: np.ndarray,
    score_lower_by_state: Sequence[float],
    transition_l1_radius_by_state: Sequence[float],
    initial_state: int,
    horizon: int,
) -> tuple[float, np.ndarray]:
    """Rectangular robust lower value for an average Markov packet score."""

    probability = validate_markov_transition(
        np.asarray(nominal_transition, dtype=float)
    )
    score_lower = np.asarray(score_lower_by_state, dtype=float)
    radii = np.asarray(transition_l1_radius_by_state, dtype=float)
    states = probability.shape[0]
    if (
        score_lower.shape != (states,)
        or radii.shape != (states,)
        or not np.all(np.isfinite(score_lower))
        or not np.all(np.isfinite(radii))
        or np.any((radii < 0.0) | (radii > 2.0))
    ):
        raise ValueError("score lower bounds or transition radii are invalid")
    if not 0 <= initial_state < states or horizon <= 0:
        raise ValueError("invalid initial state or horizon")

    value = np.zeros(states, dtype=float)
    for _ in range(horizon):
        continuation = np.asarray(
            [
                worst_case_simplex_expectation_l1(
                    nominal_probability=probability[state],
                    values=value,
                    l1_radius=float(radii[state]),
                )
                for state in range(states)
            ]
        )
        value = score_lower + continuation
    return float(value[initial_state] / horizon), value / horizon


def empirical_conditional_markov_alignment_lower_bound(
    *,
    transition_counts: np.ndarray,
    score_sum_by_state: Sequence[float],
    score_count_by_state: Sequence[int],
    score_range: float,
    initial_state: int,
    future_horizon: int,
    total_action_count: int,
    event_failure_probability: float,
    score_approximation_error: float = 0.0,
    transition_l1_drift_by_state: Sequence[float] | None = None,
) -> RobustConditionalAlignmentCertificate:
    """State-conditional finite-horizon lower bound from tabular feedback.

    Departures from a fixed state of a time-homogeneous controlled Markov
    chain have the same conditional transition law.  Coordinate Hoeffding
    bounds, unioned over states, successors, actions, and all positive visit
    counts, give row-wise L1 balls.  Bounded score observations use the same
    count-uniform construction.  Rectangular robust dynamic programming then
    lower-bounds the packet value from the current state.

    A zero transition or score count returns an unidentified ``-inf`` bound,
    causing the Lyapunov packet-weight minimizer to reject that uncertified
    action.  Exploration is therefore a separate, explicitly charged design
    question rather than being hidden in this safety certificate.
    """

    counts = np.asarray(transition_counts, dtype=float)
    score_sums = np.asarray(score_sum_by_state, dtype=float)
    score_counts = np.asarray(score_count_by_state, dtype=int)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("transition counts must be square")
    states = counts.shape[0]
    if (
        score_sums.shape != (states,)
        or score_counts.shape != (states,)
        or np.any(counts < 0.0)
        or np.any(score_counts < 0)
        or not np.all(np.isfinite(counts))
        or not np.all(np.isfinite(score_sums))
    ):
        raise ValueError("empirical count or score arrays are invalid")
    if score_range < 0.0 or score_approximation_error < 0.0:
        raise ValueError("score range and approximation error must be nonnegative")
    if not 0 <= initial_state < states or future_horizon <= 0:
        raise ValueError("invalid initial state or future horizon")
    if total_action_count <= 0 or not 0.0 < event_failure_probability < 1.0:
        raise ValueError("action count and event failure probability are invalid")
    drift = (
        np.zeros(states, dtype=float)
        if transition_l1_drift_by_state is None
        else np.asarray(transition_l1_drift_by_state, dtype=float)
    )
    if (
        drift.shape != (states,)
        or not np.all(np.isfinite(drift))
        or np.any((drift < 0.0) | (drift > 2.0))
    ):
        raise ValueError("transition drift bounds must lie in [0,2]")

    row_counts = counts.sum(axis=1)
    if np.any(row_counts <= 0.0) or np.any(score_counts <= 0):
        return RobustConditionalAlignmentCertificate(
            lower_bound=float("-inf"),
            identified=False,
            robust_value_by_state=tuple(float("-inf") for _ in range(states)),
            score_radius_by_state=tuple(float("inf") for _ in range(states)),
            transition_l1_radius_by_state=tuple(float("inf") for _ in range(states)),
            event_failure_probability=float(event_failure_probability),
        )

    score_family = total_action_count * states
    transition_family = total_action_count * states * states
    score_radii = np.asarray(
        [
            anytime_bounded_mean_radius(
                sample_count=int(count),
                value_range=score_range,
                family_size=score_family,
                event_failure_probability=event_failure_probability / 2.0,
            )
            for count in score_counts
        ]
    )
    coordinate_radii = np.asarray(
        [
            anytime_bounded_mean_radius(
                sample_count=int(count),
                value_range=1.0,
                family_size=transition_family,
                event_failure_probability=event_failure_probability / 2.0,
            )
            for count in row_counts
        ]
    )
    transition_radii = np.minimum(2.0, states * coordinate_radii + drift)
    empirical_transition = counts / row_counts[:, None]
    empirical_score = score_sums / score_counts
    score_lower = empirical_score - score_radii - score_approximation_error
    lower, values = robust_finite_horizon_markov_alignment(
        nominal_transition=empirical_transition,
        score_lower_by_state=score_lower,
        transition_l1_radius_by_state=transition_radii,
        initial_state=initial_state,
        horizon=future_horizon,
    )
    return RobustConditionalAlignmentCertificate(
        lower_bound=lower,
        identified=True,
        robust_value_by_state=tuple(float(item) for item in values),
        score_radius_by_state=tuple(float(item) for item in score_radii),
        transition_l1_radius_by_state=tuple(float(item) for item in transition_radii),
        event_failure_probability=float(event_failure_probability),
    )
