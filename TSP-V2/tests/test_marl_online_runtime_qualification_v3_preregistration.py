import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualification_v3_lock_matches_implementation_and_scope():
    lock = json.loads(
        (ROOT / "internal" / "marl_online_runtime_qualification_v3_lock.json").read_text()
    )
    assert lock["controller_decisions"] > lock["window"]
    assert lock["performance_interpretation_forbidden"]
    assert lock["project_writes_forbidden"]
    assert lock["retry_on_failure_forbidden"]
    assert lock["runner_sha256"] == _sha256(
        ROOT / "experiments" / "run_marl_online_drift_queue_v3.py"
    )
    assert lock["controller_sha256"] == _sha256(
        ROOT / "experiments" / "audited_online_drift_queue_controller.py"
    )
    assert lock["analyzer_sha256"] == _sha256(
        ROOT / "experiments" / "analyze_marl_online_drift_queue_v2.py"
    )


def test_qualification_v3_script_is_scratch_only_and_uses_audited_runner():
    script = (
        ROOT / "slurm" / "marl_online_runtime_qualification_v3_a30.sbatch"
    ).read_text()
    assert "/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-QUAL-003" in script
    assert "/project/" not in script
    assert "run_marl_online_drift_queue_v3.py" in script
    assert "--controller-decisions 9" in script
    assert "--window 8" in script
    assert "--array=0-7%2" in script
