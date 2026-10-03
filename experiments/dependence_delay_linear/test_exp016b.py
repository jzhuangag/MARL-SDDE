"""Static tests for the outcome-free EXP-016B preregistration."""

import json
from pathlib import Path
import unittest

from run_exp016b import (
    FINITE_BUDGET_NAMES,
    FORBIDDEN_INFORMATION_ONLY_INPUTS,
    PILOT_SEEDS,
    POLICIES,
    T018_GRID_HASH,
    build_bundle,
    build_scenario_manifest,
    validate_bundle,
    workload_estimate,
)
from t018_static_scan import information_only_taint_audit


ROOT = Path(__file__).resolve().parents[2]


class Exp016BPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = build_bundle()
        cls.manifest = cls.bundle["manifest"]

    def test_frozen_files_equal_rebuilt_configuration(self):
        mapping = {
            "manifest": "exp016b_scenario_manifest.json",
            "gates": "exp016b_gate_table.json",
            "seeds": "exp016b_seed_registry.json",
            "power": "exp016b_power_audit.json",
        }
        for key, filename in mapping.items():
            frozen = json.loads((ROOT / "docs" / filename).read_text(encoding="utf-8"))
            self.assertEqual(frozen, self.bundle[key])
        self.assertEqual(validate_bundle(self.bundle), [])

    def test_hash_sampling_is_deterministic_and_outcome_independent(self):
        rebuilt = build_scenario_manifest()
        frozen_without_bundle_hash = dict(self.manifest)
        frozen_without_bundle_hash.pop("configuration_sha256")
        self.assertEqual(rebuilt, frozen_without_bundle_hash)
        rule = self.manifest["selection_algorithm"]
        self.assertEqual(rule["depends_only_on"], [
            "scenario metadata", "T-018 grid hash", "SHA-256 ordering"
        ])
        self.assertIn("random outcome", rule["forbidden_dependencies"])
        self.assertIn("largest observed effect", rule["forbidden_dependencies"])
        self.assertEqual(self.manifest["source_grid_hash"], T018_GRID_HASH)

    def test_finite_and_censored_budget_populations_are_separate(self):
        finite = [row for row in self.manifest["scenarios"] if row["B_value_status"] == "finite"]
        censored = [row for row in self.manifest["scenarios"] if row["B_value_status"] == "search_censored"]
        self.assertEqual(len(finite), 96)
        self.assertEqual(len(censored), 8)
        for row in finite:
            self.assertNotEqual(row["B_value"], 2_000_001)
            self.assertEqual([point["name"] for point in row["budget_points"]], list(FINITE_BUDGET_NAMES))
            self.assertTrue(row["finite_threshold_gate_eligible"])
        for row in censored:
            self.assertIsNone(row["B_value"])
            self.assertFalse(row["finite_threshold_gate_eligible"])
            self.assertTrue(all("B_value" not in point["name"] for point in row["budget_points"]))

    def test_all_required_strata_and_cell_types_are_retained(self):
        finite = [row for row in self.manifest["scenarios"] if row["B_value_status"] == "finite"]
        expected = {
            "Q": 3, "theta_high": 4, "lambda": 4, "D": 4, "overhead": 3,
            "budget_ray": 3, "epsilon_safe": 2,
        }
        for field, count in expected.items():
            self.assertEqual(len({row[field] for row in finite}), count)
        self.assertTrue({"message", "environment"}.issubset({row["binding_type"] for row in finite}))
        self.assertIn("mixed_practical_and_neutral", {row["effect_profile"] for row in finite})
        self.assertIn(0, {row["D"] for row in finite})
        self.assertTrue(any(row["D"] > 0 for row in finite))

    def test_policy_and_layer_contracts(self):
        self.assertEqual(tuple(self.manifest["policies"]), POLICIES)
        self.assertEqual(len(POLICIES), 8)
        contract = self.manifest["information_only_contract"]
        self.assertEqual(tuple(contract["forbidden_inputs"]), FORBIDDEN_INFORMATION_ONLY_INPUTS)
        self.assertTrue(information_only_taint_audit()["passes"])
        layer_b = self.manifest["layer_specifications"]["B_affine_markov_td_transfer"]
        self.assertTrue(layer_b["actual_td_updates"])
        self.assertTrue(layer_b["actual_delay_queue"])
        self.assertTrue(layer_b["complete_dual_budget_charging"])
        self.assertTrue(layer_b["stability_screened_action_catalogue"])
        self.assertFalse(layer_b["hidden_rho_theta_regime_inputs"])
        self.assertFalse(layer_b["actor_critic"])
        self.assertFalse(layer_b["preconditioner"])

    def test_gates_and_seed_registry_are_frozen(self):
        gates = self.bundle["gates"]
        self.assertEqual([gate["id"] for gate in gates["gates"]], [f"P{i}" for i in range(1, 13)])
        self.assertTrue(all(gate["mandatory"] for gate in gates["gates"]))
        self.assertEqual(gates["cell_practical_effect_threshold_relative"], 0.03)
        self.assertEqual(gates["scenario_level_coverage_gate"], 0.60)
        seeds = self.bundle["seeds"]
        self.assertEqual(tuple(seeds["pilot_seeds"]), PILOT_SEEDS)
        self.assertEqual(len(set(PILOT_SEEDS)), 96)
        self.assertIsNone(seeds["formal_seeds"])

    def test_power_and_workload_are_feasible_without_outcomes(self):
        power = self.bundle["power"]
        self.assertTrue(power["power_target_met"])
        self.assertGreaterEqual(power["power_at_practical_threshold"], 0.80)
        estimate = workload_estimate(self.manifest)
        self.assertTrue(estimate["within_local_limits"])
        self.assertLessEqual(estimate["estimated_single_process_cpu_hours"], 6.0)
        self.assertLessEqual(estimate["estimated_peak_memory_gb"], 32.0)
        self.assertLessEqual(estimate["estimated_disk_gb"], 20.0)
        self.assertEqual(estimate["recommended_execution"], "local CPU")
        self.assertFalse(any(payload["scientific_outcomes_present"] for payload in self.bundle.values()))
        self.assertFalse((ROOT / "experiments/dependence_delay_linear/results/exp016b").exists())


if __name__ == "__main__":
    unittest.main()
