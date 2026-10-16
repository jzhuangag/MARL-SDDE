import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualification_manifest_and_grid_are_frozen():
    path = ROOT / "experiments" / "marl_online_runtime_qualification.json"
    config = json.loads(path.read_text())
    assert _sha(path) == "a8ff69258eae80d2320d9f7376fa92789edc879b9c302437d25113e783eb244d"
    assert config["implementation_commit"] == (
        "291b402f854568553d747946d584cff315759c34"
    )
    assert (
        len(config["tasks"])
        * len(config["couplings"])
        * len(config["methods"])
        == config["mandatory_pass_conditions"]["complete_unique_runs"]
        == 8
    )


def test_qualification_sources_match_registered_hashes():
    config = json.loads(
        (ROOT / "experiments" / "marl_online_runtime_qualification.json").read_text()
    )
    for relative, expected in config["source_sha256"].items():
        path = ROOT.parent / relative
        assert _sha(path) == expected


def test_qualification_seeds_are_disjoint_from_stopped_development():
    qualification = json.loads(
        (ROOT / "experiments" / "marl_online_runtime_qualification.json").read_text()
    )
    stopped = json.loads(
        (ROOT / "experiments" / "marl_online_development.json").read_text()
    )
    stopped_seeds = set(stopped["seeds"]) | set(stopped["evaluation_seeds"])
    stopped_seeds |= set(stopped["validation_seed_bases"])
    qualification_seeds = set(qualification["training_seeds"])
    qualification_seeds |= set(qualification["evaluation_seeds"])
    qualification_seeds |= set(qualification["validation_seed_bases"])
    assert qualification_seeds.isdisjoint(stopped_seeds)
    assert len(qualification_seeds) == 24
