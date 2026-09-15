from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from .pursuit_delayed_alignment_interface import collect_alignment_launches
from .pursuit_profile_factor_critic import (
    ProfileConditionedFactorCritic,
    fit_factor_normalization,
    normalize_selected_batch,
    predict_alignment_candidates,
    selected_packet_batch,
    train_profile_factor_critic,
)


def test_selected_packet_view_is_counterfactual_label_blind() -> None:
    rows, _ = collect_alignment_launches(
        seeds=(94101,),
        cycles=10,
        n_evaders=12,
        rollout_horizon=1,
        counterfactual_replicates=1,
    )
    altered = [
        replace(
            row,
            candidate_feedback=((None, 999.0),),
            candidate_mean_feedback=((None, -999.0),),
            candidate_raw_mean_alignment=((None, 777.0),),
            candidate_factor_mean_alignment=((None, -777.0),),
        )
        for row in rows
    ]
    original_batch = selected_packet_batch(rows)
    altered_batch = selected_packet_batch(altered)
    for field in original_batch.__dataclass_fields__:
        assert np.array_equal(
            getattr(original_batch, field), getattr(altered_batch, field)
        )


def test_normalization_masks_padded_pairs() -> None:
    rows, _ = collect_alignment_launches(
        seeds=(94102,),
        cycles=10,
        n_evaders=12,
        rollout_horizon=1,
        audit_counterfactuals=False,
    )
    batch = selected_packet_batch(rows)
    normalization = fit_factor_normalization(batch)
    normalized = normalize_selected_batch(batch, normalization)
    assert np.all(normalized.pair_features[normalized.donor_mask == 0.0] == 0.0)
    assert np.all(np.isfinite(normalized.self_features))
    assert np.all(np.isfinite(normalized.pair_features))


def test_factor_critic_learns_selected_packet_teacher() -> None:
    rng = np.random.default_rng(94103)
    count = 160
    feature_dimension = 6
    self_features = rng.normal(size=(count, feature_dimension)).astype(np.float32)
    pair_features = rng.normal(size=(count, 2, feature_dimension)).astype(np.float32)
    profiles = rng.dirichlet(np.ones(5), size=(count, 2)).astype(np.float32)
    mask = np.ones((count, 2), dtype=np.float32)
    actions = rng.integers(0, 5, size=count, dtype=np.int64)
    teacher = ProfileConditionedFactorCritic(
        feature_dimension=feature_dimension,
        hidden_dimension=8,
    )
    with torch.no_grad():
        target = teacher(
            torch.as_tensor(self_features),
            torch.as_tensor(pair_features),
            torch.as_tensor(profiles),
            torch.as_tensor(mask),
            torch.as_tensor(actions),
        ).numpy()
    from .pursuit_profile_factor_critic import SelectedPacketBatch

    batch = SelectedPacketBatch(
        self_features=self_features,
        pair_features=pair_features,
        donor_profiles=profiles,
        donor_mask=mask,
        owner_action=actions,
        cost_return=target.astype(np.float32),
        seed=np.arange(count),
        packet_id=np.arange(count),
    )
    model, diagnostics = train_profile_factor_critic(
        batch,
        batch,
        seed=94104,
        hidden_dimension=24,
        epochs=500,
        learning_rate=0.01,
        patience=100,
    )
    prediction = model(
        torch.as_tensor(self_features),
        torch.as_tensor(pair_features),
        torch.as_tensor(profiles),
        torch.as_tensor(mask),
        torch.as_tensor(actions),
    ).numpy()
    assert np.mean((prediction - target) ** 2) < 2e-4
    assert diagnostics["best_validation_mse"] < 2e-4


def test_candidate_scoring_replaces_only_one_pair_profile() -> None:
    rows, _ = collect_alignment_launches(
        seeds=(94105,),
        cycles=12,
        n_evaders=12,
        rollout_horizon=1,
        counterfactual_replicates=1,
    )
    row = next(
        value
        for value in rows
        if value.reference_available and value.candidate_count > 1
    )
    feature_dimension = len(
        fit_factor_normalization(selected_packet_batch(rows)).self_location
    )
    model = ProfileConditionedFactorCritic(
        feature_dimension=feature_dimension,
        hidden_dimension=8,
    )
    normalization = fit_factor_normalization(selected_packet_batch(rows))
    score = predict_alignment_candidates(
        model=model,
        normalization=normalization,
        row=row,
    )
    assert set(score) == {candidate for candidate, _ in row.candidate_contexts}
    assert all(np.isfinite(value) for value in score.values())
