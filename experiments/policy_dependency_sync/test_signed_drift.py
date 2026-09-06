import itertools

import numpy as np
import pytest

from experiments.policy_dependency_sync.signed_drift import (
    choose_edge_and_step,
    fixed_step_score,
    optimal_step,
)


def test_fixed_step_formula_equals_direct_quadratic_difference():
    rng = np.random.default_rng(42)
    for _ in range(100):
        matrix = rng.normal(size=(4, 4))
        matrix = matrix.T @ matrix + 0.5 * np.eye(4)
        error = rng.normal(size=4)
        owner = int(rng.integers(4))
        candidate_gradient = float(rng.normal())
        step = float(rng.uniform(0.0, 0.2))
        true_gradient = float(matrix[owner] @ error)
        updated = error.copy()
        updated[owner] -= step * candidate_gradient
        direct = 0.5 * updated @ matrix @ updated - 0.5 * error @ matrix @ error
        formula = fixed_step_score(
            true_gradient,
            candidate_gradient,
            float(matrix[owner, owner]),
            step,
        )
        assert formula == pytest.approx(direct, abs=1e-11)


def test_closed_form_step_matches_dense_scalar_search():
    for true_gradient, candidate_gradient in itertools.product(
        (-2.0, -0.4, 0.3, 1.7), (-1.5, -0.2, 0.0, 0.6, 2.0)
    ):
        curvature = 1.3
        maximum = 0.7
        step = optimal_step(true_gradient, candidate_gradient, curvature, maximum)
        grid = np.linspace(0.0, maximum, 100_001)
        values = (
            -grid * true_gradient * candidate_gradient
            + 0.5 * grid**2 * curvature * candidate_gradient**2
        )
        assert fixed_step_score(
            true_gradient, candidate_gradient, curvature, step
        ) <= float(values.min()) + 1e-10


def test_joint_choice_matches_explicit_candidate_enumeration():
    true_gradient = 0.8
    stale_gradient = -0.2
    changes = {1: 0.5, 2: 1.0, 3: -0.7}
    reset = {1: 0.01, 2: 0.03, 3: 0.0}
    prices = {1: 0.02, 2: 0.01, 3: 0.04}
    choice = choose_edge_and_step(
        true_gradient,
        stale_gradient,
        changes,
        curvature=1.4,
        maximum_step=0.5,
        reset_benefits=reset,
        queue_prices=prices,
    )
    explicit = []
    for donor in (None, 1, 2, 3):
        gradient = stale_gradient + (0.0 if donor is None else changes[donor])
        step = optimal_step(true_gradient, gradient, 1.4, 0.5)
        score = fixed_step_score(
            true_gradient,
            gradient,
            1.4,
            step,
            0.0 if donor is None else reset[donor],
            0.0 if donor is None else prices[donor],
        )
        explicit.append((score, donor, step, gradient))
    best = min(explicit, key=lambda item: (item[0], item[1] is not None, item[1] or -1))
    assert choice.score == pytest.approx(best[0])
    assert choice.donor == best[1]
    assert choice.step == pytest.approx(best[2])
    assert choice.candidate_gradient == pytest.approx(best[3])


def test_non_descent_candidate_gets_zero_step_without_other_benefit():
    assert optimal_step(1.0, -1.0, 2.0, 0.5) == 0.0


@pytest.mark.parametrize("curvature", [0.0, -1.0])
def test_invalid_curvature_is_rejected(curvature):
    with pytest.raises(ValueError):
        optimal_step(1.0, 1.0, curvature, 0.5)

