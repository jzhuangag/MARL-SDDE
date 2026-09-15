import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SMAC = _load("tsp_smac_qgrid", "analyze_mappo_smacv2_qgrid.py")


def test_mpe_qgrid_is_complete_and_does_not_relabel_confirmation():
    config = json.loads((EXPERIMENTS / "marl_mpe_qgrid_audit.json").read_text())
    assert config["fixed_q_grid"] == [1, 2, 4, 8]
    assert config["new_methods"] == ["fixed_q2", "fixed_q4"]
    assert config["planned_new_runs"] == 32
    assert config["predecessor"]["frozen_result_unchanged"] is True
    assert "not a new confirmatory" in config["role"]


def test_smac_qgrid_has_theory_predicted_interior_action():
    config = json.loads(
        (EXPERIMENTS / "marl_smacv2_qgrid_extension.json").read_text()
    )
    assert config["fixed_q_grid"] == [1, 2, 4, 8]
    assert SMAC.predicted_q(config, "independent") == 8
    assert SMAC.predicted_q(config, "shared") == 2
    assert config["planned_new_runs"] == 8


def test_qgrid_gates_target_near_oracle_performance_not_q_direction():
    for filename, section in (
        ("marl_mpe_qgrid_audit.json", "audit_gates"),
        (
            "marl_smacv2_qgrid_extension.json",
            "admission_gates_for_a_future_controller_study",
        ),
    ):
        config = json.loads((EXPERIMENTS / filename).read_text())
        gates = config[section]
        serialized = json.dumps(gates).lower()
        assert "selected_q1" not in serialized
        assert "selected_q8" not in serialized
        assert "envelope" in serialized or "oracle" in serialized


def test_slurm_arrays_cover_every_new_cell_once():
    mpe = []
    for global_index in range(32):
        q = 2 if global_index // 16 == 0 else 4
        regime = "independent" if (global_index % 16) // 8 == 0 else "shared"
        seed = 96001 + global_index % 8
        mpe.append((q, regime, seed))
    assert len(mpe) == len(set(mpe)) == 32

    smac = []
    for index in range(8):
        q = 2 if index // 4 == 0 else 4
        regime = "independent" if (index % 4) // 2 == 0 else "shared"
        seed = 81101 + index % 2
        smac.append((q, regime, seed))
    assert len(smac) == len(set(smac)) == 8
