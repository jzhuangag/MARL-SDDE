from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .run_lco_headroom import (
    EXPECTED_CONFIG_SHA256,
    _fixed_masks,
    _load_config,
    _phase_and_arrival_paths,
    _specifications,
    _transition_matrix,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "docs" / "lco_headroom_config.json"


def test_frozen_config_hash_and_scale() -> None:
    config = _load_config(CONFIG_PATH)
    assert EXPECTED_CONFIG_SHA256 == "58caceeea755d8a1057073eeae0cca9284abc0f4f8e139c695c7d834eb54f6b8"
    assert len(_specifications(config)) == 8640
    assert len(_specifications(config)) * config["horizon"] == 8847360


def test_fixed_masks_obey_each_budget() -> None:
    assert set(_fixed_masks(0.25, 4)) == {(), (0,), (1,), (2,), (3,)}
    assert all(len(mask) <= 2 for mask in _fixed_masks(0.5, 4))
    assert all(len(mask) <= 3 for mask in _fixed_masks(0.75, 4))


def test_stationary_phase_controls_are_exact() -> None:
    for fraction, expected in ((0.0, False), (1.0, True)):
        phases, agents, initial = _phase_and_arrival_paths(
            seed=1,
            horizon=100,
            persistence=0.8,
            rotation_fraction=fraction,
            first_agent_probability=0.3,
        )
        assert np.all(phases == expected)
        assert set(agents).issubset({0, 1})
        assert initial.shape == (2,)


def test_rotation_and_potential_transitions_match_closed_forms() -> None:
    state = np.asarray([2.0, 3.0])
    rotation_plain = _transition_matrix(
        phase_is_rotational=True, use_optimism=False, agent=0, step=0.2
    ) @ state
    rotation_eg = _transition_matrix(
        phase_is_rotational=True, use_optimism=True, agent=0, step=0.2
    ) @ state
    potential_plain = _transition_matrix(
        phase_is_rotational=False, use_optimism=False, agent=0, step=0.2
    ) @ state
    np.testing.assert_allclose(rotation_plain, [1.4, 3.0])
    np.testing.assert_allclose(rotation_eg, [1.32, 3.0])
    np.testing.assert_allclose(potential_plain, [1.6, 3.0])


def test_config_json_is_plain_parseable() -> None:
    assert json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["experiment"] == "LCO-H1"
