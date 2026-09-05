"""Focused integrity checks for the deterministic baseline-sensing audit."""

from __future__ import annotations

import ast
import inspect
import math

from baseline_sensing_headroom_audit import _risk, _strong_fixed, evaluate_budget_cell
from run_adaptation_cost_pilot import Action
from t018_static_scan import scenario_grid


def test_source_has_no_results_input_or_trajectory_import() -> None:
    source = inspect.getsource(__import__("baseline_sensing_headroom_audit"))
    tree = ast.parse(source)
    path_literals = [
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert "t018_corrected_scan_results.json" in path_literals
    assert not any(value.endswith((".csv", ".parquet", ".npz")) for value in path_literals)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "exact_markov_terminal_mse" in imported
    assert not {"pandas", "numpy", "csv"} & imported


def test_oracle_normalization_and_strong_fixed_minimax() -> None:
    a = Action(2, 1)
    b = Action(4, 1)
    action, objective, normalized = _strong_fixed({a: (1.0, 8.0), b: (2.0, 2.0)})
    assert action == b
    assert objective == 2.0
    assert normalized == (2.0, 1.0)


def test_oracle_ratio_and_strong_fixed_not_worse_than_all_agent() -> None:
    scenario = scenario_grid()[0]
    record = evaluate_budget_cell(scenario, {"name": "test", "scale": 1000}, "finite")
    assert record is not None
    assert record["oracle_worst_normalized_ratio"] == 1.0
    assert float(record["strong_fixed_worst_normalized_ratio"]) <= float(
        record["all_agent_worst_normalized_ratio"]
    )


def test_finite_filtering_excludes_infeasible_actions() -> None:
    scenario = scenario_grid()[0]
    assert evaluate_budget_cell(scenario, {"name": "zero", "scale": 0}, "finite") is None
    assert math.isinf(_risk(0.05, scenario, 0, Action(2, 1)))
