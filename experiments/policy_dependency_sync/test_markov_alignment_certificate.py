from __future__ import annotations

import itertools

import numpy as np
import pytest

from .markov_alignment_certificate import (
    anytime_bounded_mean_radius,
    dobrushin_contraction,
    dobrushin_holdout_alignment_lower_bound,
    empirical_conditional_markov_alignment_lower_bound,
    exact_finite_state_holdout_lower_bound,
    finite_horizon_markov_alignment,
    poisson_holdout_alignment_lower_bound,
    robust_finite_horizon_markov_alignment,
    scalar_poisson_span,
    stationary_shift_upper_from_kernel_shift,
    stationary_total_variation,
    summable_event_failure_probability,
    worst_case_simplex_expectation_l1,
)


def _path_probability(
    path: tuple[int, ...], transition: np.ndarray, initial_state: int
) -> float:
    if path[0] != initial_state:
        return 0.0
    probability = 1.0
    for source, target in zip(path, path[1:]):
        probability *= float(transition[source, target])
    return probability


def test_event_failure_schedule_is_summable() -> None:
    total = 0.07
    allocated = sum(
        summable_event_failure_probability(
            total_failure_probability=total, event_index=index
        )
        for index in range(100000)
    )
    assert allocated < total
    assert allocated == pytest.approx(total, rel=1e-5)


def test_dobrushin_and_poisson_span_for_symmetric_two_state_chain() -> None:
    transition = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    score = np.asarray([-0.25, 0.25])
    assert dobrushin_contraction(transition) == pytest.approx(0.8)
    assert scalar_poisson_span(transition, score) == pytest.approx(2.5)
    robust_span = np.ptp(score) / (1.0 - dobrushin_contraction(transition))
    assert scalar_poisson_span(transition, score) <= robust_span + 1e-12


def test_conditional_value_preserves_current_state() -> None:
    transition = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    score = np.asarray([0.25, -0.25])
    left = finite_horizon_markov_alignment(
        transition=transition, score_by_state=score, initial_state=0, horizon=4
    )
    right = finite_horizon_markov_alignment(
        transition=transition, score_by_state=score, initial_state=1, horizon=4
    )
    assert left > 0.0
    assert right == pytest.approx(-left)


def test_stationary_perturbation_bound_dominates_exact_tv() -> None:
    reference = np.asarray([[0.8, 0.2], [0.35, 0.65]])
    target = np.asarray([[0.75, 0.25], [0.25, 0.75]])
    kernel_shift = max(
        0.5 * np.abs(reference[row] - target[row]).sum() for row in range(2)
    )
    upper = stationary_shift_upper_from_kernel_shift(
        kernel_shift_total_variation=kernel_shift,
        target_contraction_upper=dobrushin_contraction(target),
    )
    assert stationary_total_variation(reference, target) <= upper + 1e-12


def test_exact_holdout_bound_has_fixed_event_coverage_by_enumeration() -> None:
    transition = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    score = np.asarray([-0.4, 0.6])
    sample_count = 8
    future_horizon = 5
    event_delta = 0.1
    violation_probability = 0.0
    for initial_state in range(2):
        for path in itertools.product(range(2), repeat=sample_count):
            probability = 0.5 * _path_probability(path, transition, initial_state)
            if probability == 0.0:
                continue
            certificate = exact_finite_state_holdout_lower_bound(
                reference_transition=transition,
                target_transition=transition,
                score_by_state=score,
                reference_state_path=path,
                future_horizon=future_horizon,
                action_count=2,
                event_failure_probability=event_delta,
            )
            target = finite_horizon_markov_alignment(
                transition=transition,
                score_by_state=score,
                initial_state=path[-1],
                horizon=future_horizon,
            )
            if certificate.lower_bound > target + 1e-12:
                violation_probability += probability
    assert violation_probability <= event_delta / 2.0 + 1e-12


def test_dobrushin_bound_is_no_less_conservative_than_exact_constants() -> None:
    transition = np.asarray([[0.7, 0.3], [0.2, 0.8]])
    score = np.asarray([-0.2, 0.8])
    path = np.asarray([0, 0, 1, 1, 1, 0, 1, 1])
    exact = exact_finite_state_holdout_lower_bound(
        reference_transition=transition,
        target_transition=transition,
        score_by_state=score,
        reference_state_path=path,
        future_horizon=6,
        action_count=3,
        event_failure_probability=0.05,
    )
    robust = dobrushin_holdout_alignment_lower_bound(
        score_samples=score[path],
        score_oscillation_upper=float(np.ptp(score)),
        reference_contraction_upper=dobrushin_contraction(transition),
        target_contraction_upper=dobrushin_contraction(transition),
        kernel_shift_total_variation_upper=0.0,
        future_horizon=6,
        action_count=3,
        event_failure_probability=0.05,
    )
    assert robust.lower_bound <= exact.lower_bound + 1e-12


def test_approximation_error_is_subtracted_exactly() -> None:
    base = poisson_holdout_alignment_lower_bound(
        score_samples=[0.4] * 20,
        reference_poisson_span=0.0,
        target_poisson_span=0.0,
        target_score_oscillation=0.0,
        stationary_shift_total_variation_upper=0.0,
        future_horizon=4,
        action_count=2,
        event_failure_probability=0.1,
    )
    perturbed = poisson_holdout_alignment_lower_bound(
        score_samples=[0.4] * 20,
        reference_poisson_span=0.0,
        target_poisson_span=0.0,
        target_score_oscillation=0.0,
        stationary_shift_total_variation_upper=0.0,
        future_horizon=4,
        action_count=2,
        event_failure_probability=0.1,
        approximation_error=0.13,
    )
    assert perturbed.lower_bound == pytest.approx(base.lower_bound - 0.13)


def test_simplex_l1_minimizer_matches_dense_binary_grid() -> None:
    nominal = np.asarray([0.65, 0.35])
    values = np.asarray([-0.4, 0.9])
    radius = 0.5
    exact = worst_case_simplex_expectation_l1(
        nominal_probability=nominal, values=values, l1_radius=radius
    )
    feasible = []
    for first in np.linspace(0.0, 1.0, 20001):
        candidate = np.asarray([first, 1.0 - first])
        if np.abs(candidate - nominal).sum() <= radius + 1e-12:
            feasible.append(float(candidate @ values))
    assert exact == pytest.approx(min(feasible), abs=1e-4)


def test_robust_dynamic_programming_is_below_every_checked_kernel() -> None:
    nominal = np.asarray([[0.75, 0.25], [0.3, 0.7]])
    score_lower = np.asarray([0.5, -0.2])
    radii = np.asarray([0.2, 0.3])
    lower, _ = robust_finite_horizon_markov_alignment(
        nominal_transition=nominal,
        score_lower_by_state=score_lower,
        transition_l1_radius_by_state=radii,
        initial_state=0,
        horizon=5,
    )
    for first_row_zero in np.linspace(0.65, 0.85, 21):
        for second_row_zero in np.linspace(0.15, 0.45, 31):
            transition = np.asarray(
                [
                    [first_row_zero, 1.0 - first_row_zero],
                    [second_row_zero, 1.0 - second_row_zero],
                ]
            )
            exact = finite_horizon_markov_alignment(
                transition=transition,
                score_by_state=score_lower,
                initial_state=0,
                horizon=5,
            )
            assert lower <= exact + 1e-12


def test_anytime_radius_contracts_and_empirical_state_value_is_identified() -> None:
    small = anytime_bounded_mean_radius(
        sample_count=100,
        value_range=1.0,
        family_size=4,
        event_failure_probability=0.05,
    )
    large = anytime_bounded_mean_radius(
        sample_count=10000,
        value_range=1.0,
        family_size=4,
        event_failure_probability=0.05,
    )
    assert large < small
    transition = np.asarray([[0.85, 0.15], [0.2, 0.8]])
    counts = np.rint(200000 * transition).astype(int)
    score = np.asarray([0.7, -0.1])
    score_counts = np.asarray([200000, 200000])
    certificate = empirical_conditional_markov_alignment_lower_bound(
        transition_counts=counts,
        score_sum_by_state=score * score_counts,
        score_count_by_state=score_counts,
        score_range=0.8,
        initial_state=0,
        future_horizon=4,
        total_action_count=2,
        event_failure_probability=0.05,
    )
    truth = finite_horizon_markov_alignment(
        transition=transition,
        score_by_state=score,
        initial_state=0,
        horizon=4,
    )
    assert certificate.identified
    assert certificate.lower_bound <= truth
    assert certificate.lower_bound > 0.0


def test_empirical_state_value_fails_closed_on_unvisited_state() -> None:
    certificate = empirical_conditional_markov_alignment_lower_bound(
        transition_counts=np.asarray([[10, 0], [0, 0]]),
        score_sum_by_state=[5.0, 0.0],
        score_count_by_state=[10, 0],
        score_range=1.0,
        initial_state=0,
        future_horizon=2,
        total_action_count=2,
        event_failure_probability=0.1,
    )
    assert not certificate.identified
    assert certificate.lower_bound == float("-inf")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_poisson_span": -1.0},
        {"stationary_shift_total_variation_upper": 1.1},
        {"future_horizon": 0},
        {"action_count": 0},
        {"event_failure_probability": 1.0},
    ],
)
def test_invalid_holdout_inputs_fail_closed(kwargs: dict[str, float]) -> None:
    arguments = dict(
        score_samples=[0.1, 0.2],
        reference_poisson_span=1.0,
        target_poisson_span=1.0,
        target_score_oscillation=1.0,
        stationary_shift_total_variation_upper=0.1,
        future_horizon=2,
        action_count=2,
        event_failure_probability=0.1,
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        poisson_holdout_alignment_lower_bound(**arguments)
