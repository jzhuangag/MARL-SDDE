from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .analyze_two_clocks_standard_pilot import analyze
from .run_two_clocks_standard_pilot import (
    METHODS,
    PROFILES,
    _segmented_discounted_returns,
    barrier_update_count,
    packet_opportunities,
)


CONFIG_PATH = Path(__file__).with_name("two_clocks_standard_pilot_config.json")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_registered_schedules_have_equal_packet_work_and_adaptive_depth() -> None:
    config = _config()
    for task in config["tasks"].values():
        horizon = task["logical_horizon"]
        for profile, services in task["service_profiles"].items():
            packets = packet_opportunities(services, horizon)
            assert packets > 0
            assert barrier_update_count(services, horizon) <= packets
            if profile == "heterogeneous":
                assert packets / barrier_update_count(services, horizon) >= 2.5


def test_fixed_transition_block_resets_return_to_go_at_episode_boundaries() -> None:
    rewards = np.asarray([1.0, 2.0, 10.0, 20.0])
    episode_ends = np.asarray([False, True, False, True])
    actual = _segmented_discounted_returns(rewards, episode_ends, 0.5)
    np.testing.assert_allclose(actual, [2.0, 2.0, 20.0, 20.0])


def test_pilot_seed_and_formal_separation_is_frozen() -> None:
    config = _config()
    seeds = [seed for values in config["pilot_seeds"].values() for seed in values]
    assert len(seeds) == len(set(seeds)) == 8
    assert config["formal_seeds"] == []
    assert config["methods"] == list(METHODS)
    assert config["service_profiles"] == list(PROFILES)


def _synthetic_summary(task: str, config: dict, code_commit: str, config_hash: str) -> dict:
    task_config = config["tasks"][task]
    rows = []
    for profile in PROFILES:
        services = task_config["service_profiles"][profile]
        packets = packet_opportunities(services, task_config["logical_horizon"])
        for seed in config["pilot_seeds"][task]:
            for method in METHODS:
                updates = packets if method != "frozen_barrier" else barrier_update_count(
                    services, task_config["logical_horizon"]
                )
                gain = 0.0
                if method == "two_clocks_async":
                    gain = 0.2 if profile == "heterogeneous" else 0.02
                if method == "delay_scaled_async":
                    gain = 0.1 if profile == "heterogeneous" else 0.01
                baseline_steps = config["baseline_episodes"] * task_config["episode_length"]
                charged = baseline_steps + packets * task_config["episode_length"]
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "service_profile": profile,
                        "seed": seed,
                        "completed_packets": packets,
                        "optimizer_updates": updates,
                        "charged_environment_steps": charged,
                        "charged_actor_transitions": charged * task_config["agents"],
                        "initial_return": 1.0,
                        "terminal_return": 2.0 + gain,
                        "return_change": 1.0 + gain,
                        "logical_time_auc": 2.0 + gain,
                        "maximum_self_fresh_error": 0.0,
                        "mean_event_delay": 1.0,
                        "clipped_packet_fraction": 0.0,
                        "curve": [
                            {"logical_time": fraction}
                            for fraction in config["evaluation_fractions"]
                        ],
                    }
                )
    rows.sort(key=lambda row: (row["seed"], row["service_profile"], row["method"]))
    return {
        "experiment_id": config["experiment_id"],
        "task": task,
        "config_sha256": config_hash,
        "code_commit": code_commit,
        "methods": list(METHODS),
        "profiles": list(PROFILES),
        "seeds": config["pilot_seeds"][task],
        "rows": rows,
        "formal_authorized": False,
    }


def test_analyzer_accepts_only_complete_reproducible_positive_mechanism() -> None:
    config_bytes = CONFIG_PATH.read_bytes()
    import hashlib

    config = json.loads(config_bytes)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    code_commit = "a" * 40
    primary = {
        task: _synthetic_summary(task, config, code_commit, config_hash)
        for task in config["tasks"]
    }
    payloads = {
        task: (json.dumps(summary, sort_keys=True) + "\n").encode()
        for task, summary in primary.items()
    }
    result = analyze(
        primary=primary,
        reproduction=primary,
        primary_bytes=payloads,
        reproduction_bytes=payloads,
        config=config,
        config_sha256=config_hash,
        required_code_commit=code_commit,
        manifests_verified=True,
    )
    assert result["all_mandatory_gates_passed"]
    assert result["formal_authorized"] is False


def test_analyzer_rejects_a_trivial_fallback() -> None:
    config_bytes = CONFIG_PATH.read_bytes()
    import hashlib

    config = json.loads(config_bytes)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    code_commit = "b" * 40
    primary = {
        task: _synthetic_summary(task, config, code_commit, config_hash)
        for task in config["tasks"]
    }
    for summary in primary.values():
        for row in summary["rows"]:
            if row["method"] == "two_clocks_async":
                row["logical_time_auc"] = 2.0
    payloads = {
        task: (json.dumps(summary, sort_keys=True) + "\n").encode()
        for task, summary in primary.items()
    }
    result = analyze(
        primary=primary,
        reproduction=primary,
        primary_bytes=payloads,
        reproduction_bytes=payloads,
        config=config,
        config_sha256=config_hash,
        required_code_commit=code_commit,
        manifests_verified=True,
    )
    assert not result["gates"]["P3_heterogeneous_auc_improvement_minimum"]
    assert not result["all_mandatory_gates_passed"]
