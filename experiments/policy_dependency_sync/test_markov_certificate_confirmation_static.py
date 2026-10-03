from __future__ import annotations

from .run_markov_certificate_confirmation import (
    CONFIG as CONFIRMATION_CONFIG,
    canonical_config_hash,
)
from .run_markov_certificate_nonvacuity import CONFIG as DEVELOPMENT_CONFIG


def test_confirmation_configuration_is_frozen() -> None:
    assert CONFIRMATION_CONFIG["audit_id"] == "MCERT-001"
    assert len(CONFIRMATION_CONFIG["seeds"]) == 128
    assert canonical_config_hash() == "8fafb90456be3537228482411dddc63caab55bdd734c5f3088e334843f96b3d3"


def test_confirmation_seeds_are_disjoint_from_development() -> None:
    assert set(CONFIRMATION_CONFIG["seeds"]).isdisjoint(DEVELOPMENT_CONFIG["seeds"])


def test_scientific_model_and_gates_are_unchanged_from_development() -> None:
    keys = (
        "sample_sizes_per_positive_action",
        "transition",
        "score_by_action",
        "future_horizon",
        "total_action_count_including_null",
        "event_failure_probability",
        "gates",
    )
    assert all(CONFIRMATION_CONFIG[key] == DEVELOPMENT_CONFIG[key] for key in keys)
