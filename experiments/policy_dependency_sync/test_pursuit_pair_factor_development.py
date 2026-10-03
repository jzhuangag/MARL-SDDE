from __future__ import annotations

import numpy as np

from .pursuit_pair_factor_development import (
    fit_pair_factor_ridge,
    heldout_pair_metrics,
    pair_factor_design_row,
    predict_pair_factor_ridge,
)


def test_pair_design_row_is_bilinear_factor_contraction() -> None:
    rng = np.random.default_rng(601)
    direction = rng.normal(size=5)
    difference = rng.normal(size=5)
    features = rng.normal(size=4)
    weight = rng.normal(size=(5, 5, 4))
    design = pair_factor_design_row(
        owner_probability_direction=direction,
        donor_probability_difference=difference,
        pair_features=features,
    )
    assert abs(design @ weight.ravel() - np.einsum(
        "i,ijk,j,k", direction, weight, difference, features
    )) < 1e-10


def test_pair_ridge_recovers_structured_effects() -> None:
    rng = np.random.default_rng(602)
    matrix = rng.normal(size=(180, 24))
    weight = rng.normal(size=24)
    response = matrix @ weight
    model = fit_pair_factor_ridge(matrix[:140], response[:140], ridge=1e-8)
    prediction = predict_pair_factor_ridge(model, matrix[140:])
    assert np.max(np.abs(prediction - response[140:])) < 1e-5


def test_heldout_metrics_include_null_in_best_action() -> None:
    truth = np.asarray([0.4, -0.2, -0.1, -0.3])
    prediction = np.asarray([0.3, -0.1, -0.2, -0.4])
    packet = np.asarray([1, 1, 2, 2])
    metrics = heldout_pair_metrics(
        truth=truth,
        prediction=prediction,
        packet_id=packet,
    )
    assert metrics["packets"] == 2
    assert metrics["best_action_accuracy"] == 1.0
    assert metrics["strict_oracle_packet_fraction"] == 0.5
