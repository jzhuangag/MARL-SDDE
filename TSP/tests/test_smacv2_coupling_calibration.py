import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load(
    "smac_cal_runner", ROOT / "experiments" / "run_smacv2_coupling_calibration.py"
)
ANALYZER = load(
    "smac_cal_analyzer",
    ROOT / "experiments" / "analyze_smacv2_coupling_calibration.py",
)


def test_dependence_summary_recovers_common_and_independent_streams():
    rng = np.random.default_rng(7)
    shared_base = rng.normal(size=(200, 1))
    shared = shared_base + 0.05 * rng.normal(size=(200, 8))
    independent = rng.normal(size=(200, 8))
    assert RUNNER.dependence_summary(shared)["median_pairwise_correlation"] > 0.99
    assert abs(RUNNER.dependence_summary(independent)["median_pairwise_correlation"]) < 0.1


def test_frozen_calibration_gates(tmp_path):
    paths = []
    for index, (regime, correlation, selected_q) in enumerate(
        (("independent", 0.1, 8), ("independent", 0.2, 8),
         ("shared", 0.9, 1), ("shared", 0.8, 1))
    ):
        row = {
            "coupling": regime,
            "policy_updates": 0,
            "return_or_win_rate_observed": False,
            "dependence": {"median_pairwise_correlation": correlation},
            "decision": {"selected_q": selected_q},
        }
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)
    result = ANALYZER.analyze(paths)
    assert result["pass"]
    assert result["separation"] == pytest.approx(0.7)
