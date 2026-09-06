from __future__ import annotations

from .run_packet_debt_headroom_dev import evaluate


def test_development_scan_is_finite_and_resource_matched() -> None:
    result = evaluate()
    assert result["contexts"] == 180
    assert result["price_cells"] == 36
    assert not result["scientific_outcome_authorized"]
    for cell in result["cells"]:
        assert cell["dynamic_debt"] >= 0.0
        assert cell["fixed_envelope_debt"] >= cell["dynamic_debt"] - 1e-8
        assert cell["dynamic_budget"]["environment"] >= 2.0
        assert cell["dynamic_budget"]["message"] >= 0.0

