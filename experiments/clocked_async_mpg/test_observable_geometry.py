from __future__ import annotations

import math

import numpy as np
import pytest

from .clocked_optimism_phase import (
    expected_quadratic_multiplier,
    heterogeneous_clock_metric,
)
from .observable_geometry import (
    certified_operator_log_gain,
    coordinate_game_transitions,
)


def test_exact_rotation_certifies_positive_log_gain() -> None:
    operator = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    certificate = certified_operator_log_gain(
        operator,
        operator_error_radius=0.0,
        operator_norm_bound=1.0,
        metric=np.diag(heterogeneous_clock_metric(0.3)),
        arrival_probabilities=(0.3, 0.7),
        step=0.5,
    )
    assert certificate.certifies_positive_gain
    assert certificate.log_gain_lower == pytest.approx(
        math.log(certificate.plain_multiplier_estimate)
        - math.log(certificate.fresh_multiplier_estimate)
    )


def test_exact_potential_rejects_fresh_anchor() -> None:
    certificate = certified_operator_log_gain(
        np.eye(2),
        operator_error_radius=0.0,
        operator_norm_bound=1.0,
        metric=np.diag(heterogeneous_clock_metric(0.1)),
        arrival_probabilities=(0.1, 0.9),
        step=0.5,
    )
    assert not certificate.certifies_positive_gain
    assert certificate.log_gain_lower < 0.0


@pytest.mark.parametrize("seed", range(25))
def test_multiplier_intervals_cover_every_sampled_operator(seed: int) -> None:
    rng = np.random.default_rng(seed)
    estimate = rng.normal(size=(2, 2))
    estimate *= 0.7 / max(0.7, np.linalg.norm(estimate, ord=2))
    perturbation = rng.normal(size=(2, 2))
    perturbation *= 0.05 / np.linalg.norm(perturbation, ord=2)
    truth = estimate + perturbation
    norm_bound = max(
        np.linalg.norm(estimate, ord=2), np.linalg.norm(truth, ord=2)
    )
    metric = np.diag([3.0, 1.0])
    probabilities = (0.25, 0.75)
    certificate = certified_operator_log_gain(
        estimate,
        operator_error_radius=0.05,
        operator_norm_bound=norm_bound,
        metric=metric,
        arrival_probabilities=probabilities,
        step=0.2,
    )
    true_plain = expected_quadratic_multiplier(
        metric,
        coordinate_game_transitions(truth, step=0.2, use_fresh_anchor=False),
        probabilities,
    )
    true_fresh = expected_quadratic_multiplier(
        metric,
        coordinate_game_transitions(truth, step=0.2, use_fresh_anchor=True),
        probabilities,
    )
    assert true_plain >= certificate.plain_multiplier_lower - 1e-12
    assert true_plain <= (
        certificate.plain_multiplier_estimate
        + certificate.plain_multiplier_radius
        + 1e-12
    )
    assert true_fresh <= certificate.fresh_multiplier_upper + 1e-12
    assert true_fresh >= (
        certificate.fresh_multiplier_estimate
        - certificate.fresh_multiplier_radius
        - 1e-12
    )


def test_large_operator_uncertainty_fails_closed() -> None:
    certificate = certified_operator_log_gain(
        np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        operator_error_radius=1.0,
        operator_norm_bound=1.0,
        metric=np.eye(2),
        arrival_probabilities=(0.5, 0.5),
        step=0.5,
    )
    assert certificate.log_gain_lower == -math.inf
    assert not certificate.certifies_positive_gain


def test_observable_certificate_validates_declared_contract() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        certified_operator_log_gain(
            2.0 * np.eye(2),
            operator_error_radius=0.1,
            operator_norm_bound=1.0,
            metric=np.eye(2),
            arrival_probabilities=(0.5, 0.5),
            step=0.2,
        )
    with pytest.raises(ValueError, match="probability"):
        certified_operator_log_gain(
            np.eye(2),
            operator_error_radius=0.1,
            operator_norm_bound=1.0,
            metric=np.eye(2),
            arrival_probabilities=(1.0,),
            step=0.2,
        )
