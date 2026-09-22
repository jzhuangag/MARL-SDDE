from __future__ import annotations

import pytest

from .run_two_clocks_mpe_bridge_g0 import _assert_outcome_free, validate_summary


def _summary() -> dict[str, object]:
    return {
        "scope": "outcome-free Two Clocks public-MPE bridge G0",
        "scientific_outcome_generated": False,
        "invariants": {"paired": True, "accounted": True},
    }


def test_g0_contract_accepts_only_all_pass_summary() -> None:
    validate_summary(_summary())


@pytest.mark.parametrize(
    "key", ["return_change", "mean_reward", "gradient_norm", "training_loss"]
)
def test_g0_contract_rejects_outcome_bearing_keys(key: str) -> None:
    summary = _summary()
    summary["nested"] = [{key: 1.0}]
    with pytest.raises(RuntimeError):
        _assert_outcome_free(summary)


def test_g0_contract_rejects_failed_invariant() -> None:
    summary = _summary()
    summary["invariants"] = {"paired": True, "accounted": False}
    with pytest.raises(RuntimeError):
        validate_summary(summary)
