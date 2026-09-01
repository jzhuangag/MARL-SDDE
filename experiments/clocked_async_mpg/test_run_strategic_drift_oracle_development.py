from __future__ import annotations

from .run_strategic_drift_oracle_development import POLICIES, run


def test_small_oracle_comparison_has_complete_cells() -> None:
    payload = run(seeds=1, workers=1, namespace="strategic-oracle-runner-unit")
    assert payload["development_only"] is True
    assert payload["row_count"] == 16*len(POLICIES)
    assert len(payload["cells"]) == 16
    for population in ("all", "heterogeneous"):
        for comparator in POLICIES[1:]:
            assert payload["aggregate"][population][comparator]["coverage"] == 1.0
