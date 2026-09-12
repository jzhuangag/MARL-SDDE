import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "audit_smacv2_coupling.py"
SPEC = importlib.util.spec_from_file_location("tsp_smacv2_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_static_smacv2_coupling_audit_passes() -> None:
    result = MODULE.audit(
        ROOT / "experiments" / "marl_smacv2_development.json"
    )
    assert result["marginal_seed_multisets_match"]
    assert result["independent_within_run_unique"]
    assert result["shared_within_run_synchronous"]
    assert result["categorical_marginal_match"]
    assert result["pass"]


def test_smoke_metadata_is_required_when_requested(tmp_path) -> None:
    result = MODULE.audit(
        ROOT / "experiments" / "marl_smacv2_development.json", tmp_path
    )
    assert result["smoke"]["complete_runs"] == 0
    assert not result["pass"]

    config = json.loads(
        (ROOT / "experiments" / "marl_smacv2_development.json").read_text(
            encoding="utf-8"
        )
    )
    for index in range(2):
        run = tmp_path / f"run-{index}"
        run.mkdir()
        (run / "tsp_bridge_metadata.json").write_text(
            json.dumps(
                {
                    "environment": "smacv2",
                    "map_name": "terran_10_vs_10",
                    "harl_commit": config["upstream"]["harl_commit"],
                    "upstream_modified": False,
                }
            ),
            encoding="utf-8",
        )
    result = MODULE.audit(
        ROOT / "experiments" / "marl_smacv2_development.json", tmp_path
    )
    assert result["smoke"]["complete_runs"] == 2
    assert result["pass"]
