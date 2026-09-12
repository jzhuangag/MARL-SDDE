from __future__ import annotations

import json
from pathlib import Path

from .mpe_bridge_contract import METHODS, TASK_AGENTS, async_completed_packets, barrier_schedule, service_bases
from .run_two_clocks_mpe_bridge_pilot import _clip_step


CONFIG = Path(__file__).with_name("two_clocks_mpe_bridge_pilot_config.json")


def test_frozen_pilot_grid_and_fresh_seed_registry() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert tuple(config["methods"]) == METHODS
    assert tuple(config["profiles"]) == ("balanced", "heterogeneous")
    assert tuple(config["tasks"]) == tuple(TASK_AGENTS)
    assert len(config["pilot_seeds"]) == len(set(config["pilot_seeds"])) == 8
    assert config["expected_endpoint_rows"] == 8 * 2 * 2 * 4
    assert config["expected_curve_rows"] == config["expected_endpoint_rows"] * 5


def test_checkpoints_align_with_every_barrier_round() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for task, task_config in config["tasks"].items():
        horizon = task_config["service_horizon"]
        for profile in config["profiles"]:
            round_length = max(service_bases(task, profile))
            for fraction in config["checkpoint_fractions"]:
                assert abs((horizon * fraction) / round_length - round((horizon * fraction) / round_length)) < 1e-12


def test_static_workload_has_expected_clock_separation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    spread = config["tasks"]["simple_spread_v2"]
    bases = service_bases("simple_spread_v2", "heterogeneous")
    assert sum(async_completed_packets(bases, spread["service_horizon"])) == 242
    barrier = barrier_schedule(bases, spread["service_horizon"], spread["episode_length"])
    assert sum(barrier["completed_by_owner"]) == 224


def test_step_clipping_is_radial_and_bounded() -> None:
    import numpy as np

    step = _clip_step(np.asarray([3.0, 4.0]), learning_rate=1.0, maximum_norm=0.2)
    assert np.allclose(step, np.asarray([0.12, 0.16]))
