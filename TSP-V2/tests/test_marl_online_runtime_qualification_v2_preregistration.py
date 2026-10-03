import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_second_qualification_manifest_and_grid_are_frozen():
    path = ROOT / "experiments" / "marl_online_runtime_qualification_v2.json"
    config = json.loads(path.read_text())
    assert _sha(path) == "e4210d88edc3bb87b7caffaa1d2f54bd179b3b3b15ad6f72ec402a4afd857fc2"
    assert config["implementation_commit"] == (
        "ef08d1f56e62f3094dded54299a5acfa361344db"
    )
    assert (
        len(config["tasks"])
        * len(config["couplings"])
        * len(config["methods"])
        == config["mandatory_pass_conditions"]["complete_unique_runs"]
        == 8
    )


def test_second_qualification_sources_match_registered_hashes():
    config = json.loads(
        (ROOT / "experiments" / "marl_online_runtime_qualification_v2.json").read_text()
    )
    for relative, expected in config["source_sha256"].items():
        assert _sha(ROOT.parent / relative) == expected


def test_second_qualification_seeds_do_not_reuse_prior_qualification():
    current = json.loads(
        (ROOT / "experiments" / "marl_online_runtime_qualification_v2.json").read_text()
    )
    previous = json.loads(
        (ROOT / "experiments" / "marl_online_runtime_qualification.json").read_text()
    )
    keys = ("training_seeds", "validation_seed_bases", "evaluation_seeds")
    current_seeds = {value for key in keys for value in current[key]}
    previous_seeds = {value for key in keys for value in previous[key]}
    assert current_seeds.isdisjoint(previous_seeds)
    assert len(current_seeds) == 24
