import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_g0b_uses_new_seeds_and_frozen_scientific_interface():
    config = json.loads(
        (ROOT / "experiments" / "progress_sensor_g0b.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["experiment_id"] == "TSP-V2-MARL-PROGRESS-G0B"
    assert config["candidate_q"] == [1, 2, 4, 8]
    assert config["micro_updates"] == 24
    assert {seed for task in config["tasks"] for seed in task["seeds"]} == {
        140201,
        140202,
    }
    assert set(config["tasks"][0]["seeds"]).isdisjoint({140101, 140102})


def test_g0b_task_specific_runtimes_and_output_root_are_explicit():
    batch = (ROOT / "slurm" / "progress_sensor_g0b_a30.sbatch").read_text(
        encoding="utf-8"
    )
    assert "MARL-SDDE-TSP-V2-PROGRESS-G0B" in batch
    assert "MPE_BASE=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001" in batch
    assert 'PYTHON="$MPE_BASE/venv/bin/python"' in batch
    assert "MARL-SDDE-TwoClocks-20260902" in batch
    assert "--candidate-q 1 2 4 8" in batch
    assert "--micro-updates 24" in batch
    assert "--experiment-id TSP-V2-MARL-PROGRESS-G0B" in batch
