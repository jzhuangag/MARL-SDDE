from pathlib import Path

import numpy as np
import pandas as pd

from exp018a_direct_gradient_config import Q_LEVELS, variance_factor
from t023_exp018b_power_audit import PROJECTIONS, aggregate_errors, extract_array


def test_extract_and_aggregate_exact_scaling(tmp_path: Path) -> None:
    rows = []
    rng = np.random.RandomState(23)
    for seed in range(32):
        base = rng.normal(size=len(PROJECTIONS))
        for rho in (0.0, 0.5, 0.9):
            for q in Q_LEVELS:
                row = {
                    "seed": seed,
                    "task": "task",
                    "mixing": "mixing",
                    "checkpoint": "checkpoint",
                    "rho": rho,
                    "q": q,
                }
                scaled = base * np.sqrt(variance_factor(q, rho))
                row.update(dict(zip(PROJECTIONS, scaled)))
                rows.append(row)
    frame = pd.DataFrame(rows)
    array, strata, seeds = extract_array(frame)
    median, p90 = aggregate_errors(array, np.arange(len(seeds)))
    assert array.shape == (32, 1, 3, 4, len(PROJECTIONS))
    assert len(strata) == 1
    assert median < 1.0e-12
    assert p90 < 1.0e-12
