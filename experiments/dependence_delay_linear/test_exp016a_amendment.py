"""Static tests for EXP-016A Amendment 1.

No test here generates a trajectory, pilot output, formal output, or
scientific outcome.
"""

import json
import math
from pathlib import Path
import unittest

from exp016a_amendment import (
    ANOMALY_SINGLE_CELL_ERROR_THRESHOLD,
    ORIGINAL_CONFIGURATION_SHA256,
    amendment_configuration_hash,
    build_feasibility_audit,
    conservative_critical_value,
    emit_payload,
    feasibility_cells,
    g6_zero_error_feasibility,
    one_sided_zero_event_bound,
    paired_difference_moments,
    revised_gate_table,
)
from run_exp016a import PILOT_SEEDS, load_frozen_manifest, workload_estimate


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_PREREG_FILES = (
    "docs/exp016a_preregistration.md",
    "docs/exp016a_gate_table.json",
    "docs/exp016a_scenario_manifest.json",
    "docs/exp016a_seed_registry.json",
    "docs/exp016a_analysis_plan.md",
)
AMENDMENT_FILES = (
    "docs/exp016a_preregistration_amendment_1.md",
    "docs/exp016a_gate_table_v2.json",
    "docs/exp016a_analysis_plan_v2.md",
    "docs/exp016a_feasibility_audit.md",
    "docs/exp016a_feasibility_audit.json",
)


class Exp016AAmendmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_frozen_manifest()
        cls.payload = emit_payload()
        cls.audit = cls.payload["audit"]
        cls.gate_table = cls.payload["gate_table_v2"]

    def test_original_provenance_files_still_exist_and_hash_is_original(self):
        for relative in ORIGINAL_PREREG_FILES:
            self.assertTrue((ROOT / relative).is_file(), relative)
        self.assertEqual(
            self.manifest["configuration_sha256"], ORIGINAL_CONFIGURATION_SHA256
        )
        self.assertFalse(self.manifest["scientific_outcomes_present"])

    def test_amendment_files_exist_and_json_matches_generator(self):
        for relative in AMENDMENT_FILES:
            self.assertTrue((ROOT / relative).is_file(), relative)
        frozen_audit = json.loads(
            (ROOT / "docs/exp016a_feasibility_audit.json").read_text(encoding="utf-8")
        )
        frozen_gates = json.loads(
            (ROOT / "docs/exp016a_gate_table_v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(frozen_audit, self.audit)
        self.assertEqual(frozen_gates, self.gate_table)

    def test_amendment_hash_binds_original_hash_and_payloads(self):
        expected = amendment_configuration_hash(
            build_feasibility_audit(), revised_gate_table(ORIGINAL_CONFIGURATION_SHA256)
        )
        self.assertEqual(self.audit["amendment_configuration_sha256"], expected)
        self.assertEqual(self.gate_table["amendment_configuration_sha256"], expected)
        self.assertEqual(
            self.audit["original_configuration_sha256"], ORIGINAL_CONFIGURATION_SHA256
        )

    def test_g6_zero_error_feasibility_calculation(self):
        result = g6_zero_error_feasibility()
        self.assertEqual(result["status"], "RED_FLAG")
        self.assertEqual(result["cells_per_direction"], 108)
        self.assertEqual(result["pilot_seeds_per_cell"], 64)
        self.assertEqual(result["n_zero_events_with_holm_floor"], 304)
        self.assertEqual(result["n_zero_events_without_multiplicity"], 119)
        self.assertFalse(result["original_per_cell_gate_feasible"])
        self.assertAlmostEqual(
            result["zero_event_bound_at_64_with_holm_floor"],
            one_sided_zero_event_bound(64, 0.05 / 108),
        )

    def test_revised_gate_parser_splits_g6_and_keeps_all_mandatory(self):
        gate_ids = [gate["id"] for gate in self.gate_table["gates"]]
        self.assertIn("G6a", gate_ids)
        self.assertIn("G6b", gate_ids)
        self.assertNotIn("G6", gate_ids)
        self.assertTrue(self.gate_table["all_mandatory"])
        self.assertTrue(all(gate["mandatory"] for gate in self.gate_table["gates"]))
        self.assertFalse(self.gate_table["scientific_outcomes_present"])

    def test_active_subset_construction_and_identical_policy_exclusion(self):
        sizes = self.audit["prospective_family_sizes"]
        identical = self.audit["identical_path_counts"]
        self.assertEqual(sizes["above_bs_high_cells"], 108)
        self.assertEqual(sizes["g4_practical_effect_subset"], 108)
        self.assertEqual(sizes["g8_learning_value_active_subset"], 0)
        self.assertGreater(identical["learning_vs_information_only_high"], 0)
        self.assertEqual(self.audit["stop_gate_decision"], "B")
        self.assertFalse(self.audit["pilot_authorization"])

    def test_exact_paired_moment_records_are_finite_for_gain_cells(self):
        cells = feasibility_cells(self.manifest)
        high_cells = [cell for cell in cells if cell["regime"] == "high"]
        for cell in high_cells:
            self.assertTrue(math.isfinite(cell["g4_expected_gain"]))
            self.assertTrue(math.isfinite(cell["g4_paired_variance"]))
            self.assertGreaterEqual(cell["g4_paired_variance"], 0.0)
            self.assertTrue(math.isfinite(cell["g4_prospective_se_64"]))
        self.assertGreater(conservative_critical_value(len(high_cells)), 0.0)

    def test_direct_paired_difference_formula_shape(self):
        cell = feasibility_cells(self.manifest)[0]
        self.assertIn("g4_expected_gain", cell)
        self.assertIn("g8_expected_gain", cell)
        self.assertGreaterEqual(cell["g4_paired_variance"], 0.0)
        self.assertGreaterEqual(cell["g8_paired_variance"], 0.0)
        self.assertTrue(callable(paired_difference_moments))

    def test_seed_isolation_and_workload_are_unchanged(self):
        registry = json.loads(
            (ROOT / "docs/exp016a_seed_registry.json").read_text(encoding="utf-8")
        )
        formal = {
            registry["formal"]["start"] + k * registry["formal"]["step"]
            for k in range(registry["formal"]["count"])
        }
        self.assertEqual(len(PILOT_SEEDS), 64)
        self.assertTrue(set(PILOT_SEEDS).isdisjoint(formal))
        self.assertEqual(self.audit["final_pilot_seed_count"], 64)
        self.assertEqual(self.audit["seed_decision"], "keep_original_64_pilot_seeds")
        self.assertEqual(workload_estimate(self.manifest), self.audit["workload"])

    def test_no_scientific_output_and_anomaly_threshold(self):
        self.assertFalse(self.audit["scientific_outcomes_present"])
        self.assertEqual(ANOMALY_SINGLE_CELL_ERROR_THRESHOLD, 0.10)
        self.assertFalse((ROOT / "experiments/dependence_delay_linear/results/exp016a_pilot").exists())
        self.assertFalse((ROOT / "experiments/dependence_delay_linear/results/exp016a_formal").exists())


if __name__ == "__main__":
    unittest.main()
