from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from .policy_inventory_headroom import (
    EVENTS,
    _action_objective,
    _base_step,
    _state,
    baseline_specs,
    declared_scenarios,
    design_payload,
    pending_count,
    validate_design,
)


def test_static_cardinality_and_unique_ids() -> None:
    validation = validate_design()
    assert validation["status"] == "static_design_valid_no_outcomes"
    assert design_payload()["scenarios"] == 72
    assert design_payload()["states"] == 72*EVENTS
    assert len(baseline_specs()) == 29
    assert len({x.scenario_id for x in declared_scenarios()}) == 72


def test_workload_counts_stay_physical() -> None:
    for scenario in declared_scenarios():
        counts = [pending_count(scenario, event) for event in range(EVENTS)]
        assert min(counts) >= 0
        assert max(counts) <= scenario.n_agents-1
        if scenario.workload in ("alternating", "bursty"):
            assert len(set(counts)) == 2


def test_declared_inventory_geometry_is_nonnegative_on_trust_interval() -> None:
    for scenario in declared_scenarios():
        for event in range(EVENTS):
            _, pending, _ = _state(scenario, event)
            for item in pending:
                for step in np.linspace(0.0, 1.0, 21):
                    assert item.post_step_log_second_moment(float(step)) >= 0


def test_zero_pending_objective_recovers_base_gain_step() -> None:
    zero_pending_states = 0
    for scenario in declared_scenarios():
        for event in range(EVENTS):
            proposal, pending, _ = _state(scenario, event)
            if pending:
                continue
            zero_pending_states += 1
            base = _base_step(proposal)
            grid = np.linspace(0.0, proposal.max_step, 10001)
            best = float(grid[int(np.argmin([
                _action_objective(proposal, pending, float(step))
                for step in grid
            ]))])
            assert best == pytest.approx(base, abs=1.1e-4)
    assert zero_pending_states > 0


def test_no_result_file_is_part_of_static_validation(tmp_path) -> None:
    validate_design()
    assert list(tmp_path.iterdir()) == []


def test_frozen_manifest_matches_runner_design() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root/"docs"/"policy_inventory_headroom_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    validation = validate_design()
    assert manifest["design"] == validation["design"]
    assert manifest["design_hash"] == validation["design_hash"]
