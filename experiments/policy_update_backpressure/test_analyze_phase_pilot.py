from __future__ import annotations

import json
from pathlib import Path

from experiments.policy_update_backpressure.analyze_phase_pilot import GATES, analyze


def test_gate_values_are_frozen() -> None:
    assert GATES["g2_high_geometric_regret_ratio_max"] == 0.85
    assert GATES["g3_high_fraction_cells_five_percent_min"] == 0.60
    assert GATES["g4_transition_geometric_regret_ratio_max"] == 0.95
    assert GATES["g5_low_geometric_regret_ratio_max"] == 1.03
    assert GATES["g8_high_final_gradient_ratio_max"] == 1.05


def test_analyzer_selects_one_strong_static_per_cell(tmp_path: Path) -> None:
    rows = []
    phases = ("low", "transition", "high")
    for phase_index, phase in enumerate(phases):
        for n_agents in (3, 5):
            for seed in (1, 2):
                cell_id = f"{phase}-n{n_agents}"
                common = {
                    "cell_id": cell_id,
                    "phase": phase,
                    "n_agents": n_agents,
                    "seed": seed,
                    "final_potential": -0.1,
                    "acceptance_rate": 0.5,
                    "accepted": 2,
                    "rejected": 2,
                    "harmful": 0,
                    "high_load_fraction": 0.4,
                    "median_load": 0.9,
                    "controller_operations": (2*n_agents+8)*4,
                }
                rows.append({
                    **common,
                    "policy": "pub",
                    "normalized_regret": 0.7-0.1*phase_index,
                    "final_gradient_norm": 0.8,
                })
                rows.append({
                    **common,
                    "policy": "fixed_a",
                    "normalized_regret": 1.0,
                    "final_gradient_norm": 1.0,
                })
                rows.append({
                    **common,
                    "policy": "fixed_b",
                    "normalized_regret": 1.2,
                    "final_gradient_norm": 1.1,
                })
    path = tmp_path/"toy.json"
    path.write_text(json.dumps({
        "design": {"trajectories": len(rows)}, "rows": rows
    }), encoding="utf-8")
    result = analyze(path)
    assert len(result["cell_results"]) == 6
    assert all(row["best_static"] == "fixed_a" for row in result["cell_results"])
    assert result["rows"] == len(rows)
