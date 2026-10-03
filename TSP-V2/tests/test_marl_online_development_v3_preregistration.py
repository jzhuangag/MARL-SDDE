import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_v3_lock_is_two_environment_and_fresh_seeded():
    lock = json.loads(
        (ROOT / "internal" / "marl_online_development_v3_lock.json").read_text()
    )
    assert len(lock["tasks"]) == 2
    assert lock["jobs"] == 40
    assert lock["training_seeds"] == [253001, 253002]
    assert lock["confirmation_seeds_registered"] is False
    assert lock["gates"]["both_environments_required"]
    assert lock["gates"]["each_task_mean_auc_gain_minimum"] == 0.01
    assert lock["runner_sha256"] == _sha256(
        ROOT / "experiments" / "run_marl_online_drift_queue_v3.py"
    )
    assert lock["controller_sha256"] == _sha256(
        ROOT / "experiments" / "audited_online_drift_queue_controller.py"
    )
    assert lock["analyzer_sha256"] == _sha256(
        ROOT / "experiments" / "analyze_marl_online_drift_queue_v2.py"
    )


def test_development_v3_script_freezes_full_strong_fixed_catalogue():
    script = (
        ROOT / "slurm" / "marl_online_development_v3_a30.sbatch"
    ).read_text()
    assert "MARL-SDDE-TSP-V2-ONLINE-DEV-003" in script
    assert "METHODS=(controller fixed_q1 fixed_q2 fixed_q4 fixed_q8)" in script
    assert "SEED=$((253001+SEED_INDEX))" in script
    assert "VALIDATION_SEED_BASE=$((283001+200*SEED_INDEX))" in script
    assert "EVALUATION_SEED=$((273001+SEED_INDEX))" in script
    assert "run_marl_online_drift_queue_v3.py" in script
    assert "/project/" not in script
    assert "${OFFSET:?OFFSET must be exported}" in script
