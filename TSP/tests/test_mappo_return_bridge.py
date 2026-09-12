import importlib.util
import json
import random
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import numpy as np
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "run_mappo_return_bridge.py"
SPEC = importlib.util.spec_from_file_location("tsp_mappo_bridge", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

ANALYZER_PATH = ROOT / "experiments" / "analyze_mappo_returns.py"
ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "tsp_mappo_analyzer", ANALYZER_PATH
)
ANALYZER = importlib.util.module_from_spec(ANALYZER_SPEC)
assert ANALYZER_SPEC.loader is not None
ANALYZER_SPEC.loader.exec_module(ANALYZER)


def test_dual_budget_horizon_charges_every_rollout() -> None:
    # q=8 binds the message budget; q=1 binds the environment budget.
    assert MODULE.usable_updates(8, 25, 10_000, 2_000, 100) == 33
    assert MODULE.usable_updates(1, 25, 10_000, 2_000, 100) == 80


def test_invalid_budget_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        MODULE.usable_updates(0, 25, 10_000, 2_000, 100)
    with pytest.raises(ValueError):
        MODULE.usable_updates(4, 25, 10_000, 2_000, -1)


def test_pinned_harl_numpy_aliases_are_installed_without_source_edit() -> None:
    previous_int = np.__dict__.pop("int", None)
    previous_bool = np.__dict__.pop("bool", None)
    try:
        MODULE.install_numpy_legacy_aliases()
        assert np.int is int
        assert np.bool is bool
    finally:
        if previous_int is None:
            np.__dict__.pop("int", None)
        else:
            np.int = previous_int
        if previous_bool is None:
            np.__dict__.pop("bool", None)
        else:
            np.bool = previous_bool


def test_smacv2_capability_streams_are_reproducible_and_distinct() -> None:
    def fake_wrapper():
        nested = SimpleNamespace(rng=np.random.default_rng())
        parent = SimpleNamespace(
            rng=np.random.default_rng(), pos_generator=nested
        )
        other = SimpleNamespace(rng=np.random.default_rng())
        wrapper = SimpleNamespace(
            env_key_to_distribution_map={"team": parent, "position": other}
        )
        wrapper.reset = lambda: random.random()
        return wrapper

    first = fake_wrapper()
    second = fake_wrapper()
    third = fake_wrapper()
    assert MODULE.seed_smacv2_capability_generators(first, 17) == 3
    assert MODULE.seed_smacv2_capability_generators(second, 17) == 3
    assert MODULE.seed_smacv2_capability_generators(third, 18) == 3

    def draws(wrapper):
        parent, other = wrapper.env_key_to_distribution_map.values()
        return (
            parent.rng.normal(size=5),
            parent.pos_generator.rng.normal(size=5),
            other.rng.normal(size=5),
        )

    first_draws = draws(first)
    second_draws = draws(second)
    third_draws = draws(third)
    for left, right in zip(first_draws, second_draws):
        np.testing.assert_array_equal(left, right)
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(first_draws, third_draws)
    )
    assert not np.array_equal(first_draws[0], first_draws[1])

    random.seed(1234)
    caller_state = random.getstate()
    first_reset = first.reset()
    assert random.getstate() == caller_state
    second_reset = first.reset()
    assert random.getstate() == caller_state
    same_seed_reset = second.reset()
    assert first_reset == same_seed_reset
    assert first_reset != second_reset


def test_specification_has_exact_cost_identities() -> None:
    args = Namespace(
        env_name="pettingzoo_mpe",
        q=4,
        rollout_length=25,
        message_budget=50_000,
        environment_budget=10_000,
        server_overhead=100,
        scenario="simple_spread_v2",
        continuous_actions=False,
        coupling="independent",
        critic_lr=5e-4,
        actor_lr=5e-4,
        seed=92001,
        seed_registry_base=92001,
        seed_registry_size=128,
    )
    spec = MODULE.specification(args)
    updates = spec["usable_updates"]
    assert spec["charged_training_actor_transitions"] == updates * 4 * 25
    assert spec["charged_training_environment_ticks"] == updates * 25
    assert spec["charged_training_messages"] == updates * (100 + 4 * 25)
    assert spec["charged_training_messages"] <= args.message_budget
    assert spec["charged_training_environment_ticks"] <= args.environment_budget


def test_smacv2_specification_records_map_not_mpe_scenario() -> None:
    args = Namespace(
        env_name="smacv2",
        map_name="terran_10_vs_10",
        q=8,
        rollout_length=200,
        message_budget=15_000_000,
        environment_budget=2_000_000,
        server_overhead=800,
        scenario="unused",
        continuous_actions=False,
        coupling="shared",
        critic_lr=5e-4,
        actor_lr=5e-4,
        seed=108001,
        seed_registry_base=108001,
        seed_registry_size=128,
    )
    spec = MODULE.specification(args)
    assert spec["environment"] == "smacv2"
    assert spec["task"] == "terran_10_vs_10"
    assert spec["map_name"] == "terran_10_vs_10"
    assert spec["scenario"] is None
    assert spec["message_cost_per_update"] == 2400


def test_cyclic_coupling_preserves_each_workers_seed_multiset() -> None:
    base = 93001
    size = 17
    registry = list(range(base, base + size))
    for coupling in ("independent", "shared"):
        for rank in range(8):
            observed = [
                MODULE.worker_seed(coupling, seed, rank, base, size)
                for seed in registry
            ]
            assert sorted(observed) == registry
    for seed in registry:
        shared = [
            MODULE.worker_seed("shared", seed, rank, base, size)
            for rank in range(8)
        ]
        assert len(set(shared)) == 1
        independent = [
            MODULE.worker_seed("independent", seed, rank, base, size)
            for rank in range(8)
        ]
        assert len(set(independent)) == 8


def test_public_uniform_preserves_each_categorical_inverse_cdf() -> None:
    probabilities = torch.tensor(
        [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=torch.float64
    )
    low = MODULE.common_categorical_sample(
        probabilities, torch.tensor([[0.1]], dtype=torch.float64)
    )
    middle = MODULE.common_categorical_sample(
        probabilities, torch.tensor([[0.4]], dtype=torch.float64)
    )
    high = MODULE.common_categorical_sample(
        probabilities, torch.tensor([[0.8]], dtype=torch.float64)
    )
    assert low.squeeze(-1).tolist() == [0, 0]
    assert middle.squeeze(-1).tolist() == [1, 0]
    assert high.squeeze(-1).tolist() == [2, 2]


def test_development_grid_is_complete_and_contains_no_formal_seeds() -> None:
    config = json.loads(
        (ROOT / "experiments" / "marl_return_development_grid.json").read_text(
            encoding="utf-8"
        )
    )
    actions = config["fixed_action_grid"]
    seeds = config["development_seed_registry"]["stage_d0_seeds"]
    expected = (
        len(config["coupling_regimes"])
        * len(actions["q_rollout_workers"])
        * len(actions["critic_lr"])
        * len(seeds)
    )
    assert expected == config["stage_d0_runs"] == 16
    assert "formal_seed_registry" not in config
    assert config["role"].startswith("development-only")


def test_frozen_analyzer_selects_strong_fixed_and_regime_oracles(tmp_path) -> None:
    config_path = ROOT / "experiments" / "marl_return_development_grid.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"pass": True}), encoding="utf-8")
    rollout_length = 25
    for coupling in config["coupling_regimes"]:
        for q in config["fixed_action_grid"]["q_rollout_workers"]:
            for critic_lr in config["fixed_action_grid"]["critic_lr"]:
                value = 5.0
                if q == 1 and critic_lr == 0.00025:
                    value = 10.0
                if coupling == "independent" and q == 4 and critic_lr == 0.0005:
                    value = 12.0
                if coupling == "shared" and q == 8 and critic_lr == 0.0005:
                    value = 13.0
                run_dir = tmp_path / f"{coupling}-{q}-{critic_lr}"
                run_dir.mkdir()
                message_cost = 100 + q * rollout_length
                usable = min(5_000 // message_cost, 2_500 // rollout_length)
                metadata = {
                    "coupling": coupling,
                    "q_rollout_workers": q,
                    "rollout_length": rollout_length,
                    "critic_lr": critic_lr,
                    "message_budget": 5_000,
                    "environment_budget": 2_500,
                    "message_cost_per_update": message_cost,
                    "environment_cost_per_update": rollout_length,
                    "usable_updates": usable,
                    "charged_training_messages": usable * message_cost,
                    "charged_training_environment_ticks": usable * rollout_length,
                    "seed": 93001,
                    "upstream_modified": False,
                    "harl_commit": config["upstream"]["commit"],
                }
                (run_dir / "tsp_bridge_metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                (run_dir / "progress.txt").write_text(
                    f"0,{value}\n{usable * q * rollout_length},{value}\n",
                    encoding="utf-8",
                )

    result = ANALYZER.evaluate_development_gate(
        tmp_path, config_path, audit_path
    )
    assert result["run_count"] == result["expected_run_count"] == 16
    assert result["strong_global_fixed"]["q"] == 1
    assert result["strong_global_fixed"]["critic_lr"] == pytest.approx(0.00025)
    assert [row["q"] for row in result["per_regime_oracle"]] == [4, 8]
    assert result["oracle_relative_headroom"] == pytest.approx(0.25)
    assert result["all_mandatory_gates_pass"]
    assert result["decision"] == "authorize-controller-design"
