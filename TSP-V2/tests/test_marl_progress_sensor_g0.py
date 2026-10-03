import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_marl_progress_sensor_g0.py"
)
SPEC = importlib.util.spec_from_file_location("run_marl_progress_sensor_g0", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_progress_statistics_reports_direction_and_uncertainty():
    values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    result = MODULE.progress_statistics(values)
    assert result["updates"] == 6
    assert result["late_minus_early"] == pytest.approx(3.0)
    assert result["slope"] == pytest.approx(1.0)
    assert result["standard_error"] > 0.0


def test_progress_statistics_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        MODULE.progress_statistics(np.asarray([0.0, 1.0, np.nan, 3.0]))

