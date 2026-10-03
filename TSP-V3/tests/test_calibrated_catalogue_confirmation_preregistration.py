import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments" / "marl_mamujoco_calibrated_confirmation.json"
SBATCH_PATH = ROOT / "slurm" / "marl_mamujoco_calibrated_confirmation_a30.sbatch"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_confirmation_lattice_and_fresh_seeds_are_frozen() -> None:
    cfg = config()
    assert cfg["role"].startswith("fresh-seed confirmation")
    assert cfg["methods"] == [
        "lyapunov_probe_commit",
        "fixed_q1",
        "fixed_q2",
        "fixed_q4",
        "fixed_q8",
    ]
    assert cfg["planned_runs"]["total"] == 80
    training = set(cfg["seed_registry"]["confirmation_training"])
    probe = set(cfg["seed_registry"]["confirmation_probe"])
    assert len(training) == len(probe) == 8
    assert training.isdisjoint(probe)
    assert training.isdisjoint({125001, 125002, 125003, 125004})
    assert probe.isdisjoint({136001, 136002, 136003, 136004})


def test_controller_and_comparators_are_complete() -> None:
    cfg = config()
    assert cfg["controller"]["candidate_q"] == [2, 8]
    assert cfg["controller"]["benchmark_return_used_by_controller"] is False
    assert cfg["controller"]["probe_updates_model"] is False
    assert cfg["inference"]["frozen_strong_single_fixed_q"] == 8
    assert cfg["mandatory_confirmation_gates"][
        "mixture_vs_q8_positive_pairs_min"
    ] == 7
    assert cfg["mandatory_confirmation_gates"]["mixture_vs_q8_sign_p_max"] == 0.05


def test_slurm_mapping_matches_frozen_lattice() -> None:
    text = SBATCH_PATH.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-79%4" in text
    assert "METHODS=(lyapunov_probe_commit fixed_q1 fixed_q2 fixed_q4 fixed_q8)" in text
    assert "SEED=$((138001+SEED_INDEX))" in text
    assert "PROBE_SEED=$((139001+SEED_INDEX))" in text
    assert "TSP-V3-MARL-MAMUJOCO-CAL-CONF-001" in text
    assert "--candidate-q 2 8" in text
    assert "/project" not in text
