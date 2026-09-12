from __future__ import annotations

import json
from pathlib import Path

from .analyze_harl_layer0_cpu_pilot import analyze


def test_frozen_configuration_has_expected_charge() -> None:
    path = Path(__file__).with_name("harl_layer0_cpu_pilot_config.json")
    configuration = json.loads(path.read_text(encoding="utf-8"))
    assert configuration["expected_actor_transitions_per_run"] == (
        2 * configuration["packets_per_run"] + configuration["baseline_episodes"]
    ) * configuration["episode_length"] * 3
    assert len(configuration["seeds"]) == 8


def test_analyzer_requires_reproduction_even_when_scientific_gates_pass() -> None:
    path = Path(__file__).with_name("harl_layer0_cpu_pilot_config.json")
    configuration = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for profile in configuration["service_profiles"]:
        for seed in configuration["seeds"]:
            for mode in configuration["modes"]:
                rows.append(
                    {
                        "service_profile": profile,
                        "seed": seed,
                        "mode": mode,
                        "initial_return": -100.0,
                        "final_return": -80.0 if mode == "strategic_split" else -90.0,
                        "return_change": 20.0 if mode == "strategic_split" else 10.0,
                        "mean_scale": 0.5 if mode == "strategic_split" else 1.0,
                        "maximum_debt": 1.0,
                        "intermediate_scale_fraction": 0.5
                        if mode == "strategic_split"
                        else 0.0,
                        "charged_actor_transitions": 61200,
                        "completed_actor_transitions": 61200,
                        "maximum_self_fresh_error": 0.0,
                    }
                )
    modes = {
        "strategic_split": {
            "mean_return_change": 20.0,
            "mean_final_return": -80.0,
            "lower_quartile_mean_final_return": -80.0,
        },
        "raw_full_data": {
            "mean_return_change": 10.0,
            "mean_final_return": -90.0,
            "lower_quartile_mean_final_return": -90.0,
        },
        "raw_half_data": {
            "mean_return_change": 10.0,
            "mean_final_return": -90.0,
            "lower_quartile_mean_final_return": -90.0,
        },
    }
    contrasts = {
        baseline: {
            "mean_paired_final_return_difference": 10.0,
            "lower_quartile_mean_return_difference": 10.0,
            "relative_mean_shortfall": 0.0,
            "strategic_strictly_better_fraction": 1.0,
        }
        for baseline in ("raw_full_data", "raw_half_data")
    }
    primary = {
        "rows": rows,
        "cells": {
            profile: {"modes": modes, "strategic_contrasts": contrasts}
            for profile in configuration["service_profiles"]
        },
    }
    result = analyze(configuration, primary, byte_equal=False)
    assert all(
        value for gate, value in result["gate_values"].items() if gate != "L9_reproducibility"
    )
    assert result["gate_values"]["L9_reproducibility"] is False
    assert result["gpu_pilot_authorized"] is False
