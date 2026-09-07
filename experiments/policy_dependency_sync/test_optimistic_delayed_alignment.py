from __future__ import annotations

import numpy as np
import pytest

from .optimistic_delayed_alignment import (
    AlignmentConfidence,
    DelayedRidgeAlignment,
    choose_optimistic_joint_factor_action,
    delayed_leverage_sum_bound,
)


def test_fit_changes_only_at_receipt() -> None:
    model = DelayedRidgeAlignment(
        dimension=2,
        ridge=1.0,
        noise_subgaussian=0.1,
        parameter_norm_upper=2.0,
    )
    before = model.gram.copy()
    packet = model.launch(np.asarray([1.0, -0.5]))
    assert np.array_equal(model.gram, before)
    model.receive(packet, 0.75, approximation_bound=0.2)
    assert not np.array_equal(model.gram, before)
    assert np.allclose(model.inverse_gram, np.linalg.inv(model.gram))
    assert model.log_determinant_ratio == pytest.approx(
        np.linalg.slogdet(model.gram)[1] - np.linalg.slogdet(before)[1]
    )
    assert model.historical_bias_squared == pytest.approx(0.04)
    with pytest.raises(KeyError):
        model.receive(packet, 0.75)


def test_noiseless_ridge_confidence_contains_linear_value() -> None:
    model = DelayedRidgeAlignment(
        dimension=2,
        ridge=0.5,
        noise_subgaussian=0.0,
        parameter_norm_upper=float(np.sqrt(5.0)),
    )
    parameter = np.asarray([1.0, -2.0])
    for context in (np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])):
        packet = model.launch(context)
        model.receive(packet, float(context @ parameter))
    contexts = {
        "null": np.asarray([0.4, -0.2]),
        "edge": np.asarray([-0.3, 0.7]),
    }
    confidence = model.confidence(
        contexts,
        current_approximation_bound={"null": 0.0, "edge": 0.0},
        delta=0.05,
    )
    for action, context in contexts.items():
        truth = float(context @ parameter)
        assert abs(confidence[action].mean - truth) <= confidence[action].radius
        assert confidence[action].upper >= truth


def test_declared_current_and_historical_bias_expand_radius() -> None:
    clean = DelayedRidgeAlignment(
        dimension=1, ridge=1.0, noise_subgaussian=0.0, parameter_norm_upper=0.0
    )
    biased = DelayedRidgeAlignment(
        dimension=1, ridge=1.0, noise_subgaussian=0.0, parameter_norm_upper=0.0
    )
    clean_packet = clean.launch(np.asarray([1.0]))
    biased_packet = biased.launch(np.asarray([1.0]))
    clean.receive(clean_packet, 0.0)
    biased.receive(biased_packet, 0.0, approximation_bound=0.3)
    clean_value = clean.confidence(
        {0: np.asarray([1.0])}, current_approximation_bound={0: 0.0}, delta=0.1
    )[0]
    biased_value = biased.confidence(
        {0: np.asarray([1.0])}, current_approximation_bound={0: 0.2}, delta=0.1
    )[0]
    assert biased_value.radius - clean_value.radius == pytest.approx(
        0.3 / np.sqrt(2.0) + 0.2
    )


def test_physical_alignment_cap_limits_optimism_without_changing_radius() -> None:
    model = DelayedRidgeAlignment(
        dimension=1, ridge=1.0, noise_subgaussian=2.0, parameter_norm_upper=3.0
    )
    uncapped = model.confidence(
        {0: np.asarray([1.0])},
        current_approximation_bound={0: 0.5},
        delta=0.01,
    )[0]
    capped = model.confidence(
        {0: np.asarray([1.0])},
        current_approximation_bound={0: 0.5},
        delta=0.01,
        alignment_upper=0.75,
    )[0]
    assert uncapped.upper > 0.75
    assert capped.upper == pytest.approx(0.75)
    assert capped.radius == pytest.approx(uncapped.radius)


def test_optimistic_index_regret_uses_only_selected_radius() -> None:
    # Optimism deliberately selects action 1 although action 0 is truly best.
    # The resulting nonzero regret is still controlled by action 1's radius;
    # no radius for the unselected comparator appears.
    true_alignment = np.asarray([0.7, 0.4, -0.1])
    radius = np.asarray([0.01, 0.20, 0.20])
    estimate = np.asarray([0.69, 0.60, -0.20])
    assert np.all(np.abs(estimate - true_alignment) <= radius + 1e-12)
    upper = estimate + radius
    # A deliberately simple member of the Lyapunov-index family:
    # J(a, alpha)=-A(a) alpha + alpha^2/2, alpha in [0,1].
    alpha_by_action = np.clip(upper, 0.0, 1.0)
    optimistic_index = -upper * alpha_by_action + 0.5 * alpha_by_action**2
    selected = int(np.argmin(optimistic_index))
    assert selected == 1
    selected_true_index = (
        -true_alignment[selected] * alpha_by_action[selected]
        + 0.5 * alpha_by_action[selected] ** 2
    )
    for comparator in range(3):
        comparator_alpha = float(np.clip(true_alignment[comparator], 0.0, 1.0))
        comparator_true_index = (
            -true_alignment[comparator] * comparator_alpha
            + 0.5 * comparator_alpha**2
        )
        assert selected_true_index - comparator_true_index <= (
            2.0 * radius[selected] * alpha_by_action[selected] + 1e-12
        )


def test_delayed_leverage_bound_dominates_explicit_sequence() -> None:
    dimension = 2
    ridge = 1.0
    maximum_delay = 3
    contexts = [
        np.asarray([1.0, 0.0]) if step % 2 == 0 else np.asarray([0.0, 1.0])
        for step in range(40)
    ]
    observed_sum = 0.0
    for launch, context in enumerate(contexts):
        gram = ridge * np.eye(dimension)
        for earlier in range(max(0, launch - maximum_delay)):
            gram += np.outer(contexts[earlier], contexts[earlier])
        observed_sum += float(np.sqrt(context @ np.linalg.solve(gram, context)))
    bound = delayed_leverage_sum_bound(
        launches=len(contexts),
        maximum_feedback_delay=maximum_delay,
        dimension=dimension,
        context_norm_upper=1.0,
        ridge=ridge,
    )
    assert observed_sum <= bound + 1e-12


def test_optimistic_scores_drive_the_exact_joint_edge_weight_choice() -> None:
    confidence = {
        "null": AlignmentConfidence(0.1, 0.0, 0.1, 0.0),
        "edge": AlignmentConfidence(0.2, 0.5, 0.7, 0.4),
    }
    selected = choose_optimistic_joint_factor_action(
        confidence_by_action=confidence,
        reset_benefit_by_action={"null": 0.0, "edge": 0.0},
        communication_cost_by_action={"null": 0.0, "edge": 1.0},
        gradient_norm_upper_by_action={"null": 1.0, "edge": 1.0},
        communication_queue=0.01,
        learning_weight=1.0,
        learning_smoothness=1.0,
        receipt_motion_upper=0.0,
        receipt_cache_linear_upper=0.0,
        outgoing_cache_weight=0.0,
        maximum_packet_weight=1.0,
    )
    assert selected.action == "edge"
    assert selected.packet_weight == pytest.approx(0.7)


def test_invalid_confidence_inputs_fail_closed() -> None:
    model = DelayedRidgeAlignment(
        dimension=2, ridge=1.0, noise_subgaussian=1.0, parameter_norm_upper=1.0
    )
    with pytest.raises(ValueError):
        model.launch(np.asarray([1.0]))
    with pytest.raises(ValueError):
        model.confidence(
            {0: np.asarray([1.0, 0.0])},
            current_approximation_bound={1: 0.0},
            delta=0.1,
        )
    with pytest.raises(ValueError):
        delayed_leverage_sum_bound(
            launches=2,
            maximum_feedback_delay=1,
            dimension=2,
            context_norm_upper=2.0,
            ridge=1.0,
        )
