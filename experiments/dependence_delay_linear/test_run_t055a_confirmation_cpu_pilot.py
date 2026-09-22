from math import prod

from experiments.dependence_delay_linear.run_t053a_sampled_cpu_pilot import (
    load_config as load_t053a_config,
    scenario_rows,
)
from experiments.dependence_delay_linear.run_t055a_confirmation_cpu_pilot import (
    load_config,
)


def test_t055a_uses_64_entirely_new_seeds():
    config = load_config()
    old = set(load_t053a_config()["pilot_seeds"])
    seeds = config["pilot_seeds"]
    assert len(seeds) == len(set(seeds)) == 64
    assert not old.intersection(seeds)
    assert seeds == list(range(202608035001, 202608035065))


def test_t055a_preserves_the_scientific_design_and_gates():
    new = load_config()
    old = load_t053a_config()
    assert new["tasks"] == old["tasks"]
    assert new["kernel_sha256"] == old["kernel_sha256"]
    assert new["grid"] == old["grid"]
    assert new["learning"] == old["learning"]
    assert new["probe"] == old["probe"]
    assert new["comparators"] == old["comparators"]
    assert list(new["mandatory_gates"]) == list(old["mandatory_gates"])
    for gate in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P12"):
        assert new["mandatory_gates"][gate] == old["mandatory_gates"][gate]


def test_t055a_workload_is_static_and_cpu_only():
    config = load_config()
    cells = prod(
        [
            len(config["tasks"]),
            len(config["grid"]["delays"]),
            len(config["grid"]["message_overheads"]),
            len(config["grid"]["correlations"]),
        ]
    )
    assert cells == len(scenario_rows(config)) == 84
    assert config["expected_workload"] == {
        "cells": 84,
        "seeds": 64,
        "endpoints": 5376,
        "long_learning_trajectories": 91392,
        "stored_full_trajectories": 0,
        "recommended_hardware": "local CPU",
    }
    assert not config["authorization"]["formal"]
    assert not config["authorization"]["gpu"]
    assert not config["authorization"]["hpc4"]


def test_t055a_prior_outcomes_only_informed_power():
    analysis = load_config()["analysis"]
    assert analysis["uses_T053A_for_power_only"] is True
    assert analysis["uses_prior_outcome_rows"] is False
    assert analysis["T053A_seeds_reused"] is False
