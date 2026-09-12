from __future__ import annotations

import numpy as np

from .pursuit_delayed_alignment_interface import (
    FEATURE_NAMES,
    collect_alignment_launches,
    episode_block_conformal_radius,
    encode_alignment_contexts,
    fit_ridge,
    predict_ridge,
    train_alignment_representation,
)


def test_pursuit_launch_receipt_interface_is_causal_and_charged() -> None:
    rows, diagnostics = collect_alignment_launches(
        seeds=(93991, 93992),
        cycles=20,
        n_evaders=12,
        rollout_horizon=1,
        counterfactual_replicates=2,
    )
    assert diagnostics["launches"] == len(rows) == 40
    assert diagnostics["delivered_packets"] == len(rows)
    assert diagnostics["maximum_candidates"] <= 5
    assert diagnostics["nonnull_launches"] > 0
    assert diagnostics["charged_policy_bytes"] == sum(
        row.policy_bytes for row in rows
    )
    assert all(row.receipt_event > row.event for row in rows)
    assert all(len(row.context) == len(FEATURE_NAMES) for row in rows)
    assert all(
        len(row.candidate_feedback)
        == len(row.candidate_mean_feedback)
        == len(row.candidate_raw_mean_alignment)
        == len(row.candidate_factor_mean_alignment)
        == len(row.candidate_probability_difference)
        == row.candidate_count
        for row in rows
    )
    assert all(len(row.owner_probability_direction) == 5 for row in rows)
    assert all(len(row.owner_observation) == 7 * 7 * 3 for row in rows)
    assert all(
        len(observation) == 7 * 7 * 3
        for row in rows
        for _, observation in row.candidate_donor_observation
    )
    assert all(
        abs(sum(delta)) < 1e-6
        for row in rows
        for _, delta in row.candidate_probability_difference
    )
    assert any(row.reference_available for row in rows)
    assert all(-1.0 <= row.feedback <= 1.0 for row in rows)
    assert all(0 <= row.launch_owner_action < 5 for row in rows)
    assert all(np.isfinite(row.discounted_reward) for row in rows)
    assert all(
        (row.donor is None and row.policy_bytes == 0)
        or (row.donor is not None and row.policy_bytes > 0)
        for row in rows
    )


def test_ridge_head_and_episode_block_radius() -> None:
    contexts = np.asarray(
        [
            [1.0, 0.0, -1.0],
            [1.0, 1.0, -0.5],
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 1.0],
        ]
    )
    response = np.asarray([-0.8, -0.05, 0.2, 0.95])
    weight, location, scale = fit_ridge(contexts, response, ridge=1e-8)
    prediction = predict_ridge(
        contexts,
        weight=weight,
        location=location,
        scale=scale,
    )
    assert np.max(np.abs(prediction - response)) < 1e-6

    residual = np.asarray([0.1, -0.3, 0.2, -0.4])
    seeds = np.asarray([1, 1, 2, 2])
    radius = episode_block_conformal_radius(
        residual,
        seeds,
        miscoverage=0.2,
    )
    assert radius == 0.4


def test_representation_is_frozen_before_linear_head() -> None:
    rng = np.random.default_rng(404)
    contexts = rng.normal(size=(96, len(FEATURE_NAMES)))
    contexts[:, 0] = 1.0
    responses = np.tanh(contexts[:, 2] * contexts[:, 4])
    model, location, scale, losses = train_alignment_representation(
        contexts,
        responses,
        seed=405,
        embedding_dimension=8,
        epochs=80,
    )
    encoded = encode_alignment_contexts(
        model,
        contexts,
        location=location,
        scale=scale,
    )
    assert encoded.shape == (96, 9)
    assert np.all(encoded[:, 0] == 1.0)
    assert losses["final_mse"] < losses["initial_mse"]
    assert all(not parameter.requires_grad for parameter in model.parameters())
