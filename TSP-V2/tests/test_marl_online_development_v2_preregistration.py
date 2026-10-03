import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str):
    return json.loads((ROOT / "experiments" / name).read_text())


def test_development_v2_grid_and_configuration_are_frozen():
    path = ROOT / "experiments" / "marl_online_development_v2.json"
    config = json.loads(path.read_text())
    assert _sha(path) == "e09e0290bfa6d17386370fd7772c259d5fe0ddabbb8fe5eafaf72c3c3ebd7dfe"
    assert config["implementation_commit"] == (
        "ef08d1f56e62f3094dded54299a5acfa361344db"
    )
    assert (
        len(config["tasks"])
        * len(config["couplings"])
        * len(config["methods"])
        * len(config["seeds"])
        == config["mandatory_development_gates"]["complete_unique_runs"]
        == 40
    )


def test_development_v2_preserves_frozen_scientific_design():
    previous = _load("marl_online_development.json")
    current = _load("marl_online_development_v2.json")
    for key in (
        "methods",
        "couplings",
        "common_controller",
        "tasks",
        "mandatory_development_gates",
        "stop_rule",
    ):
        assert current[key] == previous[key]


def test_development_v2_sources_match_registered_hashes():
    expected = {
        "experiments/harl_online_runtime_bridge_v2.py": "0b04fd1ed6a41855b5f1ed7cb4b1a5014dfbe801b25156136651a7d00287e316",
        "experiments/run_marl_online_drift_queue_v2b.py": "ce1a498bc24dc3c50c1ff4e883df18e7c72a511fd43808d1325218870583fe70",
        "experiments/run_marl_online_drift_queue.py": "5979b34ac234be73d7fe15956bd5f40dce9149e55cbd0777e33462118af07259",
        "experiments/analyze_marl_online_drift_queue.py": "cdbb757d3a03bee6fbc4ccab3f29e5a6096e56d3cdca51b91138f17140ef70a3",
        "experiments/online_drift_queue_controller.py": "f61a5bc1e1e92d366d8396c223003eaa9e3152fa4307badb970381326ce9a2fd",
        "slurm/marl_online_development_v2_a30.sbatch": "457a0dd6e6dd46e7cccf3037c3e08cb0acbd8d7ec67c67d5d712f82d17b16af8",
    }
    for relative, digest in expected.items():
        assert _sha(ROOT / relative) == digest


def test_development_v2_seed_streams_are_new_pairable_and_disjoint():
    current = _load("marl_online_development_v2.json")
    previous = _load("marl_online_development.json")
    qualifications = [
        _load("marl_online_runtime_qualification.json"),
        _load("marl_online_runtime_qualification_v2.json"),
    ]
    training = set(current["seeds"])
    evaluation = set(current["evaluation_seeds"])
    validation = {
        base + offset
        for base in current["validation_seed_bases"]
        for offset in range(max(task["controller_decisions"] for task in current["tasks"]))
    }
    assert training.isdisjoint(evaluation)
    assert training.isdisjoint(validation)
    assert evaluation.isdisjoint(validation)
    old = set(previous["seeds"]) | set(previous["evaluation_seeds"])
    old |= set(previous["validation_seed_bases"])
    for config in qualifications:
        old |= set(config["training_seeds"])
        old |= set(config["evaluation_seeds"])
        old |= set(config["validation_seed_bases"])
    assert training.isdisjoint(old)
    assert evaluation.isdisjoint(old)
    assert set(current["validation_seed_bases"]).isdisjoint(old)
