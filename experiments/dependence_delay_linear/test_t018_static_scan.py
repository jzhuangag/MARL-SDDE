"""Static tests for T-018 preregistration and analytic helpers."""

import json
import math
from pathlib import Path
import unittest

from t018_static_scan import (
    AMENDMENT_1_HASH,
    BROAD_PREVALENCE_GATE,
    ORIGINAL_EXP016A_HASH,
    PRACTICAL_EFFECT_THRESHOLD,
    build_manifest,
    build_safety_alignment,
    config_hash,
    information_only_taint_audit,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


class T018StaticScanPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (ROOT / "docs/t018_scan_manifest.json").read_text(encoding="utf-8")
        )

    def test_manifest_hash_and_scope(self):
        self.assertEqual(validate_manifest(self.manifest), [])
        without_hash = dict(self.manifest)
        observed = without_hash.pop("grid_hash")
        self.assertEqual(observed, config_hash(without_hash))
        self.assertEqual(self.manifest["grid_scenario_count"], 3456)
        self.assertEqual(self.manifest["budget_points_per_scenario"], 10)
        self.assertEqual(build_manifest()["grid_hash"], observed)

    def test_history_hashes_are_bound(self):
        self.assertEqual(self.manifest["original_exp016a_hash"], ORIGINAL_EXP016A_HASH)
        self.assertEqual(self.manifest["amendment_1_hash"], AMENDMENT_1_HASH)
        self.assertEqual(
            self.manifest["starting_head"],
            "3bdca34c4c5f0f55e3534f64f47042790d8a3daf",
        )

    def test_safety_metric_alignment(self):
        alignment = build_safety_alignment()
        self.assertEqual(alignment["theorem_metric"], "S_mean")
        self.assertTrue(alignment["s_mean_aligned"])
        self.assertFalse(alignment["s_path_control_claim_allowed"])
        # Strict example: X=-1 with prob .5 and X=1 with prob .5.
        expected_positive_part = max(0.0, 0.5 * -1.0 + 0.5 * 1.0)
        positive_part_expected = 0.5 * max(0.0, -1.0) + 0.5 * max(0.0, 1.0)
        self.assertLess(expected_positive_part, positive_part_expected)

    def test_information_only_taint_audit(self):
        audit = information_only_taint_audit()
        self.assertTrue(audit["passes"])
        self.assertEqual(audit["leaks"], [])
        for forbidden in ("downstream_risk", "wrong_commit", "epsilon_safe", "oracle_action"):
            self.assertNotIn(forbidden, audit["signature_parameters"])

    def test_preregistered_novelty_thresholds_are_not_weakened(self):
        definitions = self.manifest["definitions"]
        self.assertGreaterEqual(
            definitions["broad_prevalence_gate"], BROAD_PREVALENCE_GATE
        )
        self.assertGreaterEqual(
            definitions["practical_effect_threshold"], PRACTICAL_EFFECT_THRESHOLD
        )
        self.assertEqual(definitions["Z"], "{B: B_id <= B < B_value}")

    def test_no_outcome_statement(self):
        self.assertIn("no trajectory", self.manifest["no_outcome_statement"])
        self.assertFalse(
            (ROOT / "experiments/dependence_delay_linear/results/t018_static_scan").exists()
        )

    def test_static_scan_results_pass_novelty_gates(self):
        results_path = ROOT / "docs/t018_static_scan_results.json"
        if not results_path.exists():
            self.skipTest("T-018 scan results belong to the second commit")
        result = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(result["grid_hash"], self.manifest["grid_hash"])
        self.assertEqual(result["scenario_count"], 3456)
        self.assertEqual(result["cell_count"], 69120)
        self.assertGreaterEqual(result["Z_nonempty_scenario_fraction"], 0.25)
        self.assertGreaterEqual(result["effect_Z_cell_fraction"], 0.03)
        self.assertGreater(result["message_binding_scenario_count"], 0)
        self.assertGreater(result["environment_binding_scenario_count"], 0)
        self.assertTrue(all(result["novelty_gates"].values()))
        self.assertEqual(result["final_decision"], "A")
        self.assertFalse(result["scientific_outcomes_present"])


if __name__ == "__main__":
    unittest.main()
