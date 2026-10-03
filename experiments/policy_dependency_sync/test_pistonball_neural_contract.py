from __future__ import annotations

import inspect

from .pistonball_neural_contract import (
    run_neural_contract,
    validate_neural_contract,
)


def test_real_pistonball_neural_contract_passes() -> None:
    result = run_neural_contract(73001)
    assert all(validate_neural_contract(result).values())


def test_contract_is_deterministic() -> None:
    first = run_neural_contract(73002)
    second = run_neural_contract(73002)
    assert first == second


def test_decision_function_does_not_read_outcomes() -> None:
    source = inspect.getsource(run_neural_contract).lower()
    prohibited = (".rewards", ".reward", "terminations", "truncations", ".step(")
    assert all(token not in source for token in prohibited)


def test_contract_has_a_sparse_launch_cone() -> None:
    result = run_neural_contract(73003)
    assert 0 < len(result.eligible_donors) < 19
