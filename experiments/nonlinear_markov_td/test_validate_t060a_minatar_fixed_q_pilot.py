from __future__ import annotations

import json

from experiments.nonlinear_markov_td.validate_t060a_minatar_fixed_q_pilot import (
    maximum_numeric_difference,
)


def test_strict_difference_detects_nested_changes() -> None:
    first = {"a": [1.0, {"b": True}], "c": "same"}
    assert maximum_numeric_difference(first, json.loads(json.dumps(first))) == 0.0
    assert maximum_numeric_difference(first, {"a": [1.1, {"b": True}], "c": "same"}) > 0.09
    assert maximum_numeric_difference(first, {"a": [1.0], "c": "same"}) == float("inf")
