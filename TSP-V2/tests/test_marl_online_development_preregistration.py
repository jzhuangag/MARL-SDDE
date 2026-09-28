import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_frozen_configuration_and_hashes_are_consistent():
    config_path = ROOT / "experiments" / "marl_online_development.json"
    config = json.loads(config_path.read_text())
    assert _sha(config_path) == "BFCEC379B2C1D3E91A0A98FC04B031EA565B86819863A4C840DD169FC3C61C94"
    assert config["implementation_commit"] == "654b7b6"
    assert config["methods"] == [
        "controller",
        "fixed_q1",
        "fixed_q2",
        "fixed_q4",
        "fixed_q8",
    ]
    assert len(config["tasks"]) * len(config["couplings"]) * len(config["methods"]) * len(config["seeds"]) == 40
    assert config["mandatory_development_gates"]["complete_unique_runs"] == 40


def test_frozen_executables_match_preregistered_hashes():
    expected = {
        ROOT / "experiments" / "run_marl_online_drift_queue.py": "5979B34AC234BE73D7FE15956BD5F40DCE9149E55CBD0777E33462118AF07259",
        ROOT / "experiments" / "analyze_marl_online_drift_queue.py": "CDBB757D3A03BEE6FBC4CCAB3F29E5A6096E56D3CDCA51B91138F17140EF70A3",
        ROOT / "experiments" / "online_drift_queue_controller.py": "F61A5BC1E1E92D366D8396C223003EAA9E3152FA4307BADB970381326CE9A2FD",
        ROOT / "slurm" / "marl_online_development_a30.sbatch": "CEC7B92785451B242D30E1EF0B53A31111E506672A7F3FCD4FBC35079C3EAE5F",
    }
    for path, digest in expected.items():
        assert _sha(path) == digest


def test_seed_streams_are_pairable_and_disjoint():
    config = json.loads(
        (ROOT / "experiments" / "marl_online_development.json").read_text()
    )
    training = set(config["seeds"])
    evaluation = set(config["evaluation_seeds"])
    validation = {
        base + offset
        for base in config["validation_seed_bases"]
        for offset in range(max(task["controller_decisions"] for task in config["tasks"]))
    }
    assert training.isdisjoint(evaluation)
    assert training.isdisjoint(validation)
    assert evaluation.isdisjoint(validation)
