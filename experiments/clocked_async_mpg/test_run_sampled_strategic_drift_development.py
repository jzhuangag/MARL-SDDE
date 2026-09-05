from __future__ import annotations

from .run_sampled_strategic_drift_development import POLICIES, summarize


def test_summarize_uses_paired_cell_ratios() -> None:
    rows = []
    for policy_index, policy in enumerate(POLICIES, start=1):
        rows.append(
            {
                "applied_updates": 10,
                "coupling": 0.1,
                "debt": 0.2,
                "final_normalized_gap": float(policy_index),
                "mean_scale": 0.5,
                "policy": policy,
                "rejected_updates": 2,
                "seed": 0,
                "service_ratio": 2.0,
                "time_to_target": float(policy_index),
                "transition_work_at_target": float(10*policy_index),
            }
        )
    payload = summarize(rows, seeds=1)
    assert len(payload["cells"]) == 1
    assert payload["cells"][0]["mean_rejection_rate"] == 0.2
    assert payload["aggregate"]["heterogeneous"]["raw_common"][
        "time_geometric_ratio"
    ] == 0.25
