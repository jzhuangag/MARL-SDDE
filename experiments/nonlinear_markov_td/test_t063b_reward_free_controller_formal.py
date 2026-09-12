"""Static and algebraic tests for the prospective T-063B design."""

import json
from pathlib import Path

from experiments.nonlinear_markov_td.run_t060a_minatar_fixed_q_pilot import canonical_config_hash
from experiments.nonlinear_markov_td.run_t063b_reward_free_controller_formal import (
    aggregate_collision_gate,
    load_config,
)
from experiments.nonlinear_markov_td.analyze_t063b_reward_free_controller_formal import replay_equal


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs" / "t063b_reward_free_controller_formal_preregistration.json"


def test_t063b_configuration_hash_and_base_provenance() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert canonical_config_hash(spec) == spec["configuration_sha256"]
    config = load_config()
    assert len(config["pilot_seeds"]) == 512
    assert config["configuration_sha256"] == spec["configuration_sha256"]


def test_t063b_seed_registry_is_new_and_contiguous() -> None:
    config = load_config()
    seeds = config["pilot_seeds"]
    assert seeds == list(range(202608057201, 202608057713))
    excluded = set()
    for interval in config["formal_spec"]["t063b"]["seed_isolation"]["excluded_intervals"]:
        excluded.update(range(interval[0], interval[1] + 1))
    assert not excluded.intersection(seeds)


def test_t063b_uses_aggregate_exact_binomial_gate() -> None:
    gate = load_config()["collision_gate"]
    assert gate["kind"] == "aggregate_exact_binomial_upper_confidence"
    assert gate["blockwise_maximum"] == "descriptive_only"
    assert gate["probe_blocks_per_seed_task"] == 96
    assert gate["alpha"] == 0.05


def test_t063b_zero_collisions_pass_conservative_gate() -> None:
    config = load_config()
    endpoints = [
        {"master_seed": 202608057201 + index, "game": "asterix", "rho": 0.0, "match_count": 0}
        for index in range(1536)
    ]
    result = aggregate_collision_gate(config, endpoints)
    assert result["total_matches"] == 0
    assert result["pass"] is True


def test_t063b_excessive_aggregate_collisions_fail() -> None:
    config = load_config()
    endpoints = [
        {"master_seed": 202608057201, "game": "asterix", "rho": 0.0, "match_count": 96},
        {"master_seed": 202608057202, "game": "asterix", "rho": 0.0, "match_count": 96},
    ]
    result = aggregate_collision_gate(config, endpoints)
    assert result["one_sided_upper_probability"] == 1.0
    assert result["pass"] is False


def test_t063b_spec_contains_no_outcome_adaptive_fields() -> None:
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "controller_risk" not in text
    assert "endpoint" not in text
    assert "pilot" not in text.split("base_preregistration", 1)[-1]


def test_t063b_replay_gate_allows_only_serialization_scale_error() -> None:
    assert replay_equal({"x": 1.0}, {"x": 1.0 + 1e-14}, atol=1e-12, rtol=1e-12)
    assert not replay_equal({"x": 1.0}, {"x": 1.0 + 1e-6}, atol=1e-12, rtol=1e-12)
