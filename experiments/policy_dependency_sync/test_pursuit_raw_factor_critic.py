from __future__ import annotations

from dataclasses import replace

import numpy as np

from .pursuit_delayed_alignment_interface import collect_alignment_launches
from .pursuit_raw_factor_critic import (
    predict_raw_alignment_candidates,
    raw_selected_packet_batch,
    train_raw_factor_critic,
)


def test_raw_selected_view_is_counterfactual_blind() -> None:
    rows, _ = collect_alignment_launches(
        seeds=(94490,),
        cycles=10,
        n_evaders=12,
        rollout_horizon=1,
        counterfactual_replicates=1,
    )
    altered = [
        replace(
            row,
            candidate_feedback=((None, 100.0),),
            candidate_mean_feedback=((None, -100.0),),
            candidate_raw_mean_alignment=((None, 50.0),),
            candidate_factor_mean_alignment=((None, -50.0),),
        )
        for row in rows
    ]
    left = raw_selected_packet_batch(rows)
    right = raw_selected_packet_batch(altered)
    for field in left.__dataclass_fields__:
        assert np.array_equal(getattr(left, field), getattr(right, field))


def test_raw_shapes_and_candidate_scan() -> None:
    rows, _ = collect_alignment_launches(
        seeds=(94491,),
        cycles=12,
        n_evaders=12,
        rollout_horizon=1,
        counterfactual_replicates=1,
    )
    batch = raw_selected_packet_batch(rows)
    assert batch.owner_observation.shape == (12, 3, 7, 7)
    assert batch.donor_observation.shape == (12, 4, 3, 7, 7)
    assert batch.donor_profile.shape == (12, 4, 5)
    model, scale, _ = train_raw_factor_critic(
        batch,
        batch,
        seed=94492,
        epochs=4,
        batch_size=6,
        patience=4,
    )
    row = next(
        value
        for value in rows
        if value.reference_available and value.candidate_count > 1
    )
    score = predict_raw_alignment_candidates(model=model, scale=scale, row=row)
    assert set(score) == {candidate for candidate, _ in row.candidate_contexts}
    assert all(np.isfinite(value) for value in score.values())
