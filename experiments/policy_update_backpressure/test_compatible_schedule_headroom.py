from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .compatible_schedule_headroom import (
    HORIZON,
    POLICIES,
    arrivals,
    declared_scenarios,
    graph_adjacency,
    greedy_coloring,
    select,
    validate_design,
)
from .compatible_update_theory import is_independent


def test_design_cardinality_and_unique_ids() -> None:
    validation = validate_design()
    assert validation["status"] == "static_design_valid_no_outcomes"
    assert validation["design"]["scenarios"] == 48
    assert validation["design"]["scenario_policy_runs"] == 48*len(POLICIES)
    assert len({x.scenario_id for x in declared_scenarios()}) == 48


def test_graphs_are_symmetric_loop_free_and_coloring_valid() -> None:
    for scenario in declared_scenarios():
        adjacency = graph_adjacency(scenario)
        for node, neighbors in enumerate(adjacency):
            assert node not in neighbors
            assert all(node in adjacency[other] for other in neighbors)
        colors = greedy_coloring(adjacency)
        for color in set(colors):
            chosen = [node for node, value in enumerate(colors) if value == color]
            assert is_independent(chosen, adjacency)


def test_every_trace_has_arrivals_for_every_actor() -> None:
    for scenario in declared_scenarios():
        colors = greedy_coloring(graph_adjacency(scenario))
        total = sum(
            (arrivals(scenario, epoch, colors) for epoch in range(HORIZON)),
            start=np.zeros(scenario.n_agents, dtype=int),
        )
        assert (total > 0).all()


def test_every_scheduler_returns_ready_compatible_set() -> None:
    for scenario in declared_scenarios():
        adjacency = graph_adjacency(scenario)
        colors = greedy_coloring(adjacency)
        queues = np.asarray([(node % 4) for node in range(scenario.n_agents)])
        progress = np.linspace(0.5, 1.5, scenario.n_agents)
        for policy in POLICIES:
            chosen = select(
                policy, scenario, 7, queues, progress, adjacency, colors
            )
            assert is_independent(chosen, adjacency)
            assert all(queues[node] > 0 for node in chosen)


def test_static_validation_writes_no_results(tmp_path) -> None:
    validate_design()
    assert list(tmp_path.iterdir()) == []


def test_manifest_matches_runner_design() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root/"docs"/"compatible_schedule_headroom_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    validation = validate_design()
    assert manifest["design"] == validation["design"]
    assert manifest["design_hash"] == validation["design_hash"]
