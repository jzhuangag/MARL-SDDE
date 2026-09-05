from __future__ import annotations

import numpy as np

from .run_harl_freshness_development import (
    MODES,
    _birth_periodic_phase,
    _clip,
    _periodic_phase,
    _requests_birth_augmentation,
    _service_bases,
)


def test_modes_include_all_strong_barrier_baselines() -> None:
    assert {
        "lsff",
        "lsff_transition",
        "never_refresh",
        "always_refresh",
        "always_extra_birth",
    }.issubset(MODES)
    assert {f"periodic_phase_{index}" for index in range(4)}.issubset(MODES)
    assert {f"birth_periodic_phase_{index}" for index in range(4)}.issubset(
        MODES
    )


def test_step_clip_enforces_public_norm_bound() -> None:
    result = _clip(np.asarray([3.0, 4.0]), 0.1)
    assert np.linalg.norm(result) <= 0.1 + 1e-15


def test_periodic_phase_parser() -> None:
    assert _periodic_phase("periodic_phase_3") == 3
    assert _periodic_phase("lsff") is None
    assert _birth_periodic_phase("birth_periodic_phase_2") == 2
    assert _birth_periodic_phase("periodic_phase_2") is None


def test_birth_augmentation_schedule_is_predictable() -> None:
    assert _requests_birth_augmentation(
        "always_extra_birth", completion_index=3, period=4
    )
    assert _requests_birth_augmentation(
        "birth_periodic_phase_3", completion_index=3, period=4
    )
    assert not _requests_birth_augmentation(
        "birth_periodic_phase_3", completion_index=2, period=4
    )
    assert not _requests_birth_augmentation(
        "always_refresh", completion_index=3, period=4
    )


def test_service_profiles_scale_to_configured_agent_count() -> None:
    assert _service_bases("balanced", 6) == (1.0,) * 6
    heterogeneous = _service_bases("heterogeneous", 6)
    assert len(heterogeneous) == 6
    assert heterogeneous[0] == 1.0
    assert heterogeneous[-1] == 4.0
