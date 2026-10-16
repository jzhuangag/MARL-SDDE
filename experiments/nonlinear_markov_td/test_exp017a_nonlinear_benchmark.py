"""Deterministic static and smoke tests for EXP-017A preregistration."""

import inspect
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import run_exp017a_nonlinear_pilot as runner
from exp017a_nonlinear_config import (
    BUDGETS,
    CORRELATIONS,
    DELAY_TRACES,
    FORMAL_SEEDS,
    MIXING_PROFILES,
    PILOT_SEEDS,
    POLICIES,
    TASKS,
    build_static_manifest,
    delay_value,
    expected_runs,
    trace_summary,
)


class Exp017ANonlinearBenchmarkTests(unittest.TestCase):
    def test_scope_and_population_are_frozen(self) -> None:
        manifest = build_static_manifest()
        self.assertEqual(set(TASKS), {"cartpole", "acrobot"})
        self.assertEqual(CORRELATIONS, (0.0, 0.5, 0.9))
        self.assertEqual(set(DELAY_TRACES), {"zero", "edge_jitter", "wan_bursty"})
        self.assertEqual(set(BUDGETS), {"message_binding", "environment_binding"})
        self.assertEqual(len(POLICIES), 11)
        self.assertEqual(len(PILOT_SEEDS), 2)
        self.assertIsNone(FORMAL_SEEDS)
        self.assertEqual(manifest["pilot_expected_runs"], expected_runs())
        self.assertFalse(manifest["scientific_outcomes_present"])

    def test_known_mixing_certificates_satisfy_separation(self) -> None:
        for profile in MIXING_PROFILES.values():
            self.assertAlmostEqual(
                profile["lambda_upper"] + profile["gamma_certificate"], 1.0
            )
            self.assertLess(profile["lambda_upper"], 1.0)

    def test_delay_traces_are_heterogeneous_and_bounded(self) -> None:
        self.assertEqual(trace_summary("zero")["maximum"], 0)
        for name in ("edge_jitter", "wan_bursty"):
            values = {
                delay_value(name, tick, agent)
                for tick in range(256)
                for agent in range(32)
            }
            self.assertGreater(len(values), 2)
            self.assertLessEqual(max(values), DELAY_TRACES[name]["maximum"])
        self.assertGreater(
            trace_summary("wan_bursty")["p90"],
            trace_summary("edge_jitter")["p90"],
        )

    def test_common_private_assignment_has_registered_pair_probability(self) -> None:
        for rho in CORRELATIONS:
            pair_shared = []
            marginal_common = []
            for seed in range(4000, 6000):
                assignment = runner.source_assignment(seed, rho)
                marginal_common.append(assignment[0] == 0)
                pair_shared.append(assignment[0] == 0 and assignment[1] == 0)
                self.assertTrue(np.all((assignment == 0) | (assignment == np.arange(1, 33))))
            self.assertLess(abs(np.mean(pair_shared) - rho), 0.04)
            self.assertLess(abs(np.mean(marginal_common) - math.sqrt(rho)), 0.04)

    def test_information_only_api_has_no_hidden_or_learning_value_input(self) -> None:
        audit = runner.information_only_taint_audit()
        self.assertTrue(audit["passes"], audit)
        parameters = inspect.signature(runner.choose_information_only_action).parameters
        for forbidden in (
            "true_rho",
            "heldout_error",
            "mc_return",
            "source_assignment",
            "outcome_data",
            "teacher",
        ):
            self.assertNotIn(forbidden, parameters)

    def test_controller_contains_no_inverse_or_preconditioner(self) -> None:
        source = inspect.getsource(runner.choose_action) + inspect.getsource(
            runner.controller_score
        )
        self.assertNotIn("inverse", source.lower())
        self.assertNotIn("precondition", source.lower())
        self.assertNotIn("hessian", source.lower())

    def test_standard_task_banks_are_finite(self) -> None:
        for task in TASKS:
            bank = runner.generate_transition_bank(
                task, "fast_regeneration", 17, length=12, source_count=3
            )
            self.assertEqual(bank.states.shape[0], 3)
            self.assertEqual(bank.states.shape[1], 12)
            self.assertTrue(np.isfinite(bank.states).all())
            self.assertTrue(np.isfinite(bank.following).all())
            self.assertTrue(np.isfinite(bank.rewards).all())

    def test_small_cpu_configuration_respects_budgets(self) -> None:
        training = runner.generate_transition_bank(
            "cartpole", "fast_regeneration", 31, length=96
        )
        evaluation = runner.generate_transition_bank(
            "cartpole",
            "fast_regeneration",
            32,
            length=runner.EVALUATION_TRANSITIONS,
            source_count=1,
        )
        original = dict(runner.BUDGETS["message_binding"])
        try:
            runner.BUDGETS["message_binding"] = {
                "message_bytes": 4_000_000,
                "environment_steps": 48,
            }
            trace, endpoint = runner.run_configuration(
                seed=PILOT_SEEDS[0],
                task_name="cartpole",
                mixing_name="fast_regeneration",
                rho=0.5,
                delay_trace="edge_jitter",
                budget_name="message_binding",
                policy="learning_aware",
                training_bank=training,
                evaluation_bank=evaluation,
                device=torch.device("cpu"),
            )
        finally:
            runner.BUDGETS["message_binding"] = original
        self.assertTrue(trace)
        self.assertTrue(endpoint["finite"])
        self.assertLessEqual(endpoint["messages"], endpoint["message_budget"])
        self.assertLessEqual(
            endpoint["environment_steps"], endpoint["environment_budget"]
        )
        self.assertGreater(endpoint["unapplied_gradient_groups"], 0)

    def test_no_pilot_output_exists_in_repository(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.assertFalse(
            (root / "experiments/nonlinear_markov_td/results/exp017a_pilot").exists()
        )

    def test_frozen_registry_hashes_runner_and_analyzer(self) -> None:
        root = Path(__file__).resolve().parents[2]
        registry_path = root / "docs/exp017a_pilot_registry.json"
        if not registry_path.exists():
            self.skipTest("registry is generated by the preregistration freeze")
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for relative, expected in registry["file_sha256"].items():
            data = (root / relative).read_bytes().replace(b"\r\n", b"\n")
            observed = hashlib.sha256(data).hexdigest()
            self.assertEqual(observed, expected, relative)
        self.assertIsNone(registry["formal_seeds"])
        self.assertFalse(registry["scientific_outcomes_present"])


if __name__ == "__main__":
    unittest.main()
