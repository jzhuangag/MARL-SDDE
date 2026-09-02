from __future__ import annotations

import pytest

from .run_two_clocks_layer0_g0 import _assert_no_outcome_keys, validate_summary


def _valid_summary() -> dict[str, object]:
    return {
        "scope": "outcome-free Two Clocks Layer-0 G0",
        "scientific_outcome_generated": False,
        "invariants": {"ownership": True, "accounting": True},
        "accounting": {
            "completed_packets": 3,
            "applied_updates": 3,
            "cancelled_packets": 0,
        },
    }


def test_g0_summary_contract_accepts_only_all_pass_outcome_free_result() -> None:
    validate_summary(_valid_summary())


@pytest.mark.parametrize("key", ["reward", "return", "win_rate", "final_return"])
def test_g0_summary_contract_rejects_scientific_outcome_fields(key: str) -> None:
    summary = _valid_summary()
    summary["nested"] = [{key: 1.0}]
    with pytest.raises(RuntimeError):
        _assert_no_outcome_keys(summary)


def test_g0_summary_contract_rejects_failed_invariant() -> None:
    summary = _valid_summary()
    summary["invariants"] = {"ownership": True, "accounting": False}
    with pytest.raises(RuntimeError):
        validate_summary(summary)
