import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_exp018a_pilot import PROJECTIONS, diagnose
from exp018a_direct_gradient_config import Q_LEVELS


def test_posthoc_diagnosis_is_non_authorizing(tmp_path: Path) -> None:
    rows = []
    for seed in range(8):
        base = np.linspace(-1.0, 1.0, len(PROJECTIONS)) + 0.1 * seed
        for q in Q_LEVELS:
            row = {
                "seed": seed,
                "task": "synthetic",
                "mixing": "synthetic",
                "checkpoint": "frozen",
                "rho": 0.0,
                "q": q,
            }
            row.update(
                {
                    projection: float(value / np.sqrt(q))
                    for projection, value in zip(PROJECTIONS, base)
                }
            )
            rows.append(row)
    path = tmp_path / "projections.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = diagnose(path)
    assert result["status"] == "post_hoc_diagnosis_not_a_gate"
    assert "authorization" in result["interpretation_boundary"].lower()


def test_result_document_json_parses() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in (
        "exp018a_pilot_summary.json",
        "exp018a_reproduction_audit.json",
        "t022_exp018a_failure_diagnosis.json",
    ):
        json.loads((root / "docs" / name).read_text(encoding="utf-8"))
