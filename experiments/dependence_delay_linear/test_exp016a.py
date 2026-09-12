"""Static preregistration audits for EXP-016A.

These tests do not draw a trajectory or compute a scientific outcome.
"""

import inspect
import json
import math
from pathlib import Path
import unittest

from run_exp016a import (
    DELTA,
    GAMMA,
    PILOT_SEEDS,
    POLICIES,
    PREREG_PARENT,
    Action,
    budget_ray,
    budgets,
    build_manifest,
    canonical_json,
    config_hash,
    controller_decision,
    crn_stream_key,
    information_only_score,
    load_frozen_manifest,
    qualification_margin,
    rounded_budget_scale,
    threshold_region,
    validate_manifest,
    workload_estimate,
)


ROOT = Path(__file__).resolve().parents[2]


class Exp016APreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_frozen_manifest()
        cls.scenarios = cls.manifest["positive_scenarios"]

    def test_starting_parent_and_no_scientific_outcomes(self):
        self.assertEqual(self.manifest["prereg_parent"], PREREG_PARENT)
        self.assertFalse(self.manifest["scientific_outcomes_present"])

    def test_threshold_ordering_and_separated_scope(self):
        for scenario in self.scenarios:
            self.assertLessEqual(scenario["B_N_raw"], scenario["B_S"])
            self.assertLessEqual(scenario["B_N_raw"], scenario["B_oracle"])
            self.assertLessEqual(scenario["B_oracle"], scenario["B_S"])
            self.assertLessEqual(scenario["lambda"], 1.0 - GAMMA)
            self.assertGreaterEqual(scenario["oracle_gap_relative"], 0.03)

    def test_registered_rounding_rules(self):
        for scenario in self.scenarios:
            bn, bs = scenario["B_N_raw"], scenario["B_S"]
            scales = {point["name"]: point["scale"] for point in scenario["budget_points"]}
            self.assertEqual(scales["half_bn"], math.floor(0.5 * bn))
            self.assertEqual(scales["near_bn"], math.floor(0.9 * bn))
            self.assertEqual(scales["above_bs"], math.ceil(1.1 * bs))
            self.assertEqual(scales["double_bs"], math.ceil(2.0 * bs))
            self.assertEqual(scales["gray_mid"], rounded_budget_scale("gray_mid", bn, bs))

    def test_below_bn_fallback_is_deterministic(self):
        for scenario in self.scenarios:
            for point in scenario["budget_points"]:
                if point["region"] == "below_bn":
                    self.assertEqual(
                        controller_decision(point["scale"], scenario["B_N_raw"], scenario["B_S"], 1000.0),
                        "fallback",
                    )

    def test_above_bs_qualification_is_frozen(self):
        for scenario in self.scenarios:
            ray = budget_ray(
                scenario["budget_ray"]["name"],
                scenario["overhead"],
                scenario["maximum_agents"],
            )
            probe_data = scenario["sufficient_probe"]
            from run_exp016a import Probe

            probe = Probe(**probe_data)
            qualified, margins = qualification_margin(
                scenario["B_S"], probe, scenario["theta_high"], scenario["lambda"],
                scenario["overhead"], scenario["delay"], scenario["maximum_agents"],
                ray, scenario["epsilon_safe"],
            )
            self.assertTrue(qualified)
            self.assertLessEqual(margins["safety_relative"], scenario["epsilon_safe"])
            self.assertGreaterEqual(margins["high_gain_relative"], 0.005)

    def test_full_dual_budget_and_delay_accounting(self):
        for scenario in self.scenarios:
            ray = budget_ray(
                scenario["budget_ray"]["name"], scenario["overhead"], scenario["maximum_agents"]
            )
            message, environment = budgets(scenario["B_S"], ray)
            probe = scenario["sufficient_probe"]
            self.assertLessEqual(probe["n_sufficient"] * (scenario["overhead"] + probe["q"]), message)
            self.assertLessEqual(probe["n_sufficient"] * probe["b"] + scenario["delay"], environment)
            if scenario["delay"] > 0:
                necessary = scenario["necessary_probe"]
                no_delay_cost = max(
                    necessary["n_necessary"] * (scenario["overhead"] + necessary["q"]) / ray.beta_message,
                    necessary["n_necessary"] * necessary["b"] / ray.beta_environment,
                )
                self.assertGreaterEqual(scenario["B_N_raw"], no_delay_cost)

    def test_controller_has_no_hidden_state_inputs(self):
        parameters = inspect.signature(controller_decision).parameters
        for forbidden in ("theta_true", "lambda_true", "regime", "latent", "oracle_action"):
            self.assertNotIn(forbidden, parameters)

    def test_information_only_baseline_cannot_access_learning_risk(self):
        parameters = inspect.signature(information_only_score).parameters
        for forbidden in ("downstream_risk", "oracle_gap", "wrong_commit", "delay", "epsilon_safe"):
            self.assertNotIn(forbidden, parameters)

    def test_formal_seed_isolation(self):
        registry = json.loads((ROOT / "docs" / "exp016a_seed_registry.json").read_text(encoding="utf-8"))
        expected_pilot = tuple(
            registry["pilot"]["start"] + k * registry["pilot"]["step"]
            for k in range(registry["pilot"]["count"])
        )
        formal = {
            registry["formal"]["start"] + k * registry["formal"]["step"]
            for k in range(registry["formal"]["count"])
        }
        self.assertEqual(PILOT_SEEDS, expected_pilot)
        self.assertTrue(set(PILOT_SEEDS).isdisjoint(formal))
        self.assertFalse(registry["formal"]["accessible_to_pilot_runner"])
        source = (Path(__file__).with_name("run_exp016a.py")).read_text(encoding="utf-8")
        self.assertNotIn(str(registry["formal"]["start"]), source)

    def test_common_random_numbers_align_without_policy_or_future_reuse(self):
        first = crn_stream_key(PILOT_SEEDS[0], "scenario", "high", 7, 2)
        for _policy in POLICIES:
            self.assertEqual(first, crn_stream_key(PILOT_SEEDS[0], "scenario", "high", 7, 2))
        self.assertNotEqual(first, crn_stream_key(PILOT_SEEDS[0], "scenario", "high", 8, 2))
        self.assertNotEqual(first, crn_stream_key(PILOT_SEEDS[0], "scenario", "high", 7, 3))

    def test_gray_zone_has_no_core_gate_label(self):
        for scenario in self.scenarios:
            gray = next(point for point in scenario["budget_points"] if point["name"] == "gray_mid")
            self.assertEqual(gray["region"], threshold_region(gray["scale"], scenario["B_N_raw"], scenario["B_S"]))
            self.assertEqual(gray["region"], "gray_zone")
        gates = json.loads((ROOT / "docs" / "exp016a_gate_table.json").read_text(encoding="utf-8"))
        self.assertFalse(gates["gray_zone_core_gate_eligible"])

    def test_negative_controls_are_excluded(self):
        for control in self.manifest["negative_controls"]:
            self.assertFalse(control["theorem_scope"])
            self.assertFalse(control["positive_gate_eligible"])

    def test_configuration_hash_and_generation_are_deterministic(self):
        without_hash = dict(self.manifest)
        observed_hash = without_hash.pop("configuration_sha256")
        self.assertEqual(observed_hash, config_hash(without_hash))
        self.assertEqual(canonical_json(build_manifest()), canonical_json(self.manifest))

    def test_manifest_validator_and_static_workload(self):
        self.assertEqual(validate_manifest(self.manifest), [])
        estimate = workload_estimate(self.manifest)
        self.assertEqual(estimate["positive_base_scenarios"], 54)
        self.assertEqual(estimate["policy_count"], 10)
        self.assertEqual(estimate["pilot_seed_count"], 64)
        self.assertLess(estimate["estimated_cpu_wall_hours_single_process"], 6.0)
        self.assertLess(estimate["estimated_peak_memory_gb"], 32.0)
        self.assertLess(estimate["estimated_disk_gb"], 20.0)

    def test_directional_delta_is_frozen(self):
        self.assertEqual(DELTA, 0.025)


if __name__ == "__main__":
    unittest.main()
