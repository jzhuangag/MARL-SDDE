import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
SPEC = importlib.util.spec_from_file_location(
    "smac_headroom", ROOT / "experiments" / "analyze_mappo_smacv2_headroom.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_headroom_config_costs_and_seed_legality():
    config = json.loads(
        (ROOT / "experiments" / "marl_smacv2_headroom.json").read_text(
            encoding="utf-8"
        )
    )
    budget = config["budgets"]
    assert budget["fixed_q1_updates"] == min(
        budget["message_budget"] // 1000, budget["environment_budget"] // 200
    )
    assert budget["fixed_q8_updates"] == min(
        budget["message_budget"] // 2400, budget["environment_budget"] // 200
    )
    for seed in config["development_seeds"]:
        assert seed * 50_000 + 7 * 10_000 < 2**32


def test_relative_gain_is_scale_normalized():
    assert MODULE.relative_gain(11, 10) == pytest.approx(0.1)
    assert MODULE.relative_gain(9, 10) == pytest.approx(-0.1)
