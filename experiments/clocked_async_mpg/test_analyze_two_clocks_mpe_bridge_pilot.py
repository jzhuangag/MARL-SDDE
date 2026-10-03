from __future__ import annotations

from .analyze_two_clocks_mpe_bridge_pilot import analyze_rows


def _synthetic() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    config = {
        "pilot_seeds": [11, 12],
        "tasks": {"task_a": {"service_horizon": 4.0}, "task_b": {"service_horizon": 4.0}},
        "checkpoint_fractions": [0.0, 0.5, 1.0],
        "maximum_event_delay": 8,
        "expected_endpoint_rows": 32,
        "expected_curve_rows": 96,
    }
    endpoints = []
    curves = []
    for seed in config["pilot_seeds"]:
        for task in config["tasks"]:
            for profile in ("balanced", "heterogeneous"):
                for method in ("offdiag_async", "raw_async", "delay_scaled_async", "frozen_barrier"):
                    initial = -10.0
                    gain = 2.0
                    if profile == "heterogeneous" and method == "offdiag_async":
                        gain = 3.0
                    elif profile == "heterogeneous" and method == "frozen_barrier":
                        gain = 2.0
                    auc = initial + gain / 2.0
                    endpoints.append(
                        {
                            "seed": seed,
                            "task": task,
                            "profile": profile,
                            "method": method,
                            "initial_policy_digest": f"policy-{seed}-{task}",
                            "frozen_control_variate_digest": f"cv-{seed}-{task}",
                            "initial_return": initial,
                            "final_return": initial + gain,
                            "return_change": gain,
                            "return_auc": auc,
                            "completed_packets": 10,
                            "optimizer_updates": 8,
                            "completed_environment_steps": 100,
                            "completed_actor_transitions": 300,
                            "cancelled_environment_steps": 2,
                            "cancelled_actor_transitions": 6,
                            "baseline_environment_steps": 20,
                            "baseline_actor_transitions": 60,
                            "evaluation_environment_steps": 30,
                            "evaluation_actor_transitions": 90,
                            "cumulative_step_norm": 0.5 if method == "offdiag_async" else 0.8,
                            "cumulative_policy_kl": 0.1,
                            "cumulative_teammate_birth_arrival_kl": 0.1,
                            "maximum_owner_error": 0.0,
                            "maximum_event_delay": 4,
                            "offdiag_lyapunov_scale": 0.6,
                            "lyapunov_condition_max": 1.0,
                            "logical_service_time": 4.0,
                        }
                    )
                    for time, value in ((0.0, initial), (2.0, initial + gain / 2.0), (4.0, initial + gain)):
                        curves.append(
                            {
                                "seed": seed,
                                "task": task,
                                "profile": profile,
                                "method": method,
                                "logical_time": time,
                                "evaluation_return": value,
                            }
                        )
    return endpoints, curves, config


def test_frozen_analysis_accepts_broad_phase_witness() -> None:
    endpoints, curves, config = _synthetic()
    result = analyze_rows(endpoints, curves, config)
    assert result["mandatory_without_reproduction_passed"]
    assert result["metrics"]["heterogeneous_normalized_auc_gain"] == 0.05


def test_frozen_analysis_rejects_nonlearning_method() -> None:
    endpoints, curves, config = _synthetic()
    for row in endpoints:
        if row["method"] == "raw_async":
            row["return_change"] = -1.0
    result = analyze_rows(endpoints, curves, config)
    assert not result["gates"]["P2_positive_learning"]
