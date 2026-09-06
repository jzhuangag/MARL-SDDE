import itertools

import numpy as np
import pytest

from experiments.policy_dependency_sync.signed_drift import (
    choose_edge_and_step,
    delayed_cache_reset_benefit,
    fixed_step_score,
    outgoing_cache_debt_drift,
    optimal_step,
    receipt_optimal_step,
    robust_alignment_lower_bound,
    robust_candidate_norm_upper_bound,
    robust_optimal_step,
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


def test_delayed_cache_reset_benefit_equals_direct_energy_change():
    weight = 1.7
    current = 1.2
    cached = -0.4
    delivered = 0.8
    before = 0.5 * weight * (current - cached) ** 2
    after = 0.5 * weight * (current - delivered) ** 2
    assert delayed_cache_reset_benefit(
        weight, current, cached, delivered
    ) == pytest.approx(before - after)


def test_delayed_refresh_can_increase_cache_debt():
    assert delayed_cache_reset_benefit(1.0, 1.0, 0.9, -1.0) < 0.0


def test_outgoing_cache_drift_equals_direct_energy_change():
    weights = {1: 0.4, 2: 1.1, 3: 0.2}
    caches = {1: -0.7, 2: 0.3, 3: 1.4}
    current = 0.8
    gradient = -0.6
    step = 0.13
    updated = current - step * gradient
    before = 0.5 * sum(
        weights[item] * (current - caches[item]) ** 2 for item in weights
    )
    after = 0.5 * sum(
        weights[item] * (updated - caches[item]) ** 2 for item in weights
    )
    assert outgoing_cache_debt_drift(
        weights, current, caches, gradient, step
    ) == pytest.approx(after - before)


def test_receipt_step_matches_explicit_composite_quadratic_search():
    weights = {1: 0.4, 2: 0.7}
    caches = {1: -0.2, 2: 0.5}
    current = 0.9
    objective_gradient = 0.8
    packet_gradient = 0.6
    objective_smoothness = 1.3
    pending_linear = 0.12
    pending_curvature = 0.3
    maximum = 0.5
    step = receipt_optimal_step(
        objective_gradient,
        packet_gradient,
        objective_smoothness,
        weights,
        current,
        caches,
        maximum,
        pending_linear,
        pending_curvature,
    )
    grid = np.linspace(0.0, maximum, 100_001)
    objective = (
        -grid * objective_gradient * packet_gradient
        + 0.5 * objective_smoothness * grid**2 * packet_gradient**2
    )
    cache = np.asarray(
        [
            outgoing_cache_debt_drift(
                weights, current, caches, packet_gradient, float(alpha)
            )
            for alpha in grid
        ]
    )
    pending = (
        grid * pending_linear * abs(packet_gradient)
        + 0.5 * pending_curvature * grid**2 * packet_gradient**2
    )
    values = objective + cache + pending
    chosen = (
        -step * objective_gradient * packet_gradient
        + 0.5 * objective_smoothness * step**2 * packet_gradient**2
        + outgoing_cache_debt_drift(
            weights, current, caches, packet_gradient, step
        )
        + step * pending_linear * abs(packet_gradient)
        + 0.5 * pending_curvature * step**2 * packet_gradient**2
    )
    assert chosen <= float(values.min()) + 1e-10


def test_robust_bounds_cover_all_sampled_perturbations():
    rng = np.random.default_rng(731)
    estimate_a = np.asarray((0.8, -0.4, 0.3))
    estimate_g = np.asarray((0.2, 0.6, -0.5))
    radius_a = 0.25
    radius_g = 0.18
    alignment_lower = robust_alignment_lower_bound(
        estimate_a, estimate_g, radius_a, radius_g
    )
    norm_upper = robust_candidate_norm_upper_bound(estimate_g, radius_g)
    for _ in range(1_000):
        direction_a = rng.normal(size=3)
        direction_g = rng.normal(size=3)
        direction_a /= np.linalg.norm(direction_a)
        direction_g /= np.linalg.norm(direction_g)
        true_a = estimate_a + rng.random() * radius_a * direction_a
        true_g = estimate_g + rng.random() * radius_g * direction_g
        assert true_a @ true_g >= alignment_lower - 1e-12
        assert np.linalg.norm(true_g) <= norm_upper + 1e-12


def test_robust_step_minimizes_certified_scalar_upper_bound():
    alignment_lower = 0.7
    norm_upper = 1.3
    smoothness = 1.8
    maximum = 0.4
    step = robust_optimal_step(
        alignment_lower, norm_upper, smoothness, maximum
    )
    grid = np.linspace(0.0, maximum, 100_001)
    values = (
        -grid * alignment_lower
        + 0.5 * smoothness * grid**2 * norm_upper**2
    )
    value = (
        -step * alignment_lower
        + 0.5 * smoothness * step**2 * norm_upper**2
    )
    assert value <= float(values.min()) + 1e-10


def test_nonpositive_certified_alignment_rejects_learning_step():
    assert robust_optimal_step(-0.1, 1.0, 2.0, 0.5) == 0.0


@pytest.mark.parametrize("curvature", [0.0, -1.0])
def test_invalid_curvature_is_rejected(curvature):
    with pytest.raises(ValueError):
        optimal_step(1.0, 1.0, curvature, 0.5)
