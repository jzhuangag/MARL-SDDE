import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "run_mappo_return_bridge.py"
SPEC = importlib.util.spec_from_file_location("tsp_mappo_bridge", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_dual_budget_horizon_charges_every_rollout() -> None:
    # q=8 binds the message budget; q=1 binds the environment budget.
    assert MODULE.usable_updates(8, 25, 10_000, 2_000, 100) == 33
    assert MODULE.usable_updates(1, 25, 10_000, 2_000, 100) == 80


def test_invalid_budget_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        MODULE.usable_updates(0, 25, 10_000, 2_000, 100)
    with pytest.raises(ValueError):
        MODULE.usable_updates(4, 25, 10_000, 2_000, -1)


def test_specification_has_exact_cost_identities() -> None:
    args = Namespace(
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
