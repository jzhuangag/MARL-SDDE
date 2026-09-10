import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
MODULE_PATH = ROOT / "experiments" / "lyapunov_probe_commit.py"
SPEC = importlib.util.spec_from_file_location("tsp_probe_commit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RUNNER_PATH = ROOT / "experiments" / "run_mappo_probe_commit.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("tsp_probe_commit_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)

ANALYZER_PATH = ROOT / "experiments" / "analyze_mappo_probe_commit.py"
ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "tsp_probe_commit_analyzer", ANALYZER_PATH
)
ANALYZER = importlib.util.module_from_spec(ANALYZER_SPEC)
assert ANALYZER_SPEC.loader is not None
sys.modules[ANALYZER_SPEC.name] = ANALYZER
ANALYZER_SPEC.loader.exec_module(ANALYZER)

CONFIRM_ANALYZER_PATH = (
    ROOT / "experiments" / "analyze_mappo_probe_commit_confirmation.py"
)
CONFIRM_ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "tsp_probe_commit_confirmation_analyzer", CONFIRM_ANALYZER_PATH
)
CONFIRM_ANALYZER = importlib.util.module_from_spec(CONFIRM_ANALYZER_SPEC)
assert CONFIRM_ANALYZER_SPEC.loader is not None
sys.modules[CONFIRM_ANALYZER_SPEC.name] = CONFIRM_ANALYZER
CONFIRM_ANALYZER_SPEC.loader.exec_module(CONFIRM_ANALYZER)


def common_factor_fingerprints(rho: float, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(128, 1))
    private = rng.normal(size=(128, 8))
    return np.sqrt(rho) * common + np.sqrt(1.0 - rho) * private


def test_certificate_separates_independent_and_shared_fingerprints() -> None:
    low = MODULE.average_pairwise_correlation_certificate(
        common_factor_fingerprints(0.0), delta=0.05
    )
    high = MODULE.average_pairwise_correlation_certificate(
        common_factor_fingerprints(0.9), delta=0.05
    )
    assert low.upper < 1.0 / 3.0
    assert high.estimate > 0.8
    assert high.upper > 0.8


def test_zero_variance_is_conservative() -> None:
    certificate = MODULE.average_pairwise_correlation_certificate(
        np.ones((128, 8)), delta=0.05
    )
    assert certificate.upper == 1.0
    assert certificate.standard_error == float("inf")


def test_lyapunov_rule_selects_q8_low_and_q1_high() -> None:
    low = MODULE.choose_participation(
        common_factor_fingerprints(0.0), [1, 8], 100, 25, delta=0.05
    )
    high = MODULE.choose_participation(
        common_factor_fingerprints(0.9), [1, 8], 100, 25, delta=0.05
    )
    assert low.selected_q == 8
    assert high.selected_q == 1
    assert low.scores[8] < low.scores[1]
    assert high.scores[1] < high.scores[8]


def test_score_has_the_registered_one_third_boundary() -> None:
    below = 1.0 / 3.0 - 1e-6
    above = 1.0 / 3.0 + 1e-6
    assert MODULE.message_limited_lyapunov_score(8, below, 100, 25) < 5.0
    assert MODULE.message_limited_lyapunov_score(8, above, 100, 25) > 5.0
    assert MODULE.message_limited_lyapunov_score(1, below, 100, 25) == 5.0


def test_probe_and_learning_are_both_fully_charged() -> None:
    accounting = MODULE.controller_accounting(
        selected_q=1,
        probe_q=8,
        probe_blocks=128,
        rollout_length=25,
        message_budget=5_000_000,
        environment_budget=1_000_000,
        server_overhead=100,
    )
    assert accounting.probe_messages == 128 * 300
    assert accounting.probe_environment_ticks == 128 * 25
    assert accounting.training_updates == (5_000_000 - 38_400) // 125
    assert accounting.total_messages <= 5_000_000
    assert accounting.total_environment_ticks <= 1_000_000


def test_probe_exhaustion_is_rejected() -> None:
    with pytest.raises(ValueError, match="probe exhausts"):
        MODULE.controller_accounting(
            selected_q=1,
            probe_q=8,
            probe_blocks=128,
            rollout_length=25,
            message_budget=10_000,
            environment_budget=1_000_000,
            server_overhead=100,
        )


def test_development_two_config_has_disjoint_seeds_and_no_formal_registry() -> None:
    config = json.loads(
        (ROOT / "experiments" / "marl_probe_commit_development.json").read_text(
            encoding="utf-8"
        )
    )
    training = config["seed_registry"]["development_training"]
    probe = config["seed_registry"]["development_probe"]
    assert set(training).isdisjoint(probe)
    assert set(training).isdisjoint({93001})
    assert "formal" not in json.dumps(config["seed_registry"]).lower()
    assert config["controller"]["probe_blocks"] == 128
    assert config["controller"]["candidate_q"] == [1, 8]


def test_charged_progress_includes_probe_cost(tmp_path) -> None:
    progress = tmp_path / "progress.txt"
    progress.write_text("200000,-100.0\n400000,-80.0\n", encoding="utf-8")
    rows = RUNNER.charged_progress(
        progress,
        selected_q=8,
        rollout_length=25,
        server_overhead=100,
        message_budget=5_000_000,
        environment_budget=1_000_000,
        probe_messages=38_400,
        probe_environment_ticks=3_200,
    )
    assert rows[0]["cumulative_messages"] == 38_400 + 1000 * 300
    assert rows[0]["cumulative_environment_ticks"] == 3_200 + 1000 * 25
    assert rows[0]["budget_fraction"] == pytest.approx(338_400 / 5_000_000)
    assert rows[-1]["team_return"] == -80.0


def test_charged_progress_rejects_fractional_update(tmp_path) -> None:
    progress = tmp_path / "progress.txt"
    progress.write_text("201,-100.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integer update"):
        RUNNER.charged_progress(
            progress,
            selected_q=8,
            rollout_length=25,
            server_overhead=100,
            message_budget=5_000_000,
            environment_budget=1_000_000,
            probe_messages=38_400,
            probe_environment_ticks=3_200,
        )


def test_probe_fingerprint_reduction_preserves_only_worker_axis() -> None:
    residuals = np.arange(5 * 8 * 3 * 1, dtype=float).reshape(5, 8, 3, 1)
    fingerprint = RUNNER.residual_fingerprint(residuals)
    assert fingerprint.shape == (8,)
    np.testing.assert_allclose(fingerprint, residuals.mean(axis=(0, 2, 3)))


def test_probe_fingerprint_rejects_missing_worker_axis() -> None:
    with pytest.raises(ValueError, match="time and rollout-worker"):
        RUNNER.residual_fingerprint(np.ones(8))


def _write_fixed_result(
    root: Path, coupling: str, seed: int, q: int, team_return: float, commit: str
) -> None:
    run = root / f"fixed_q{q}_{coupling}_{seed}"
    run.mkdir(parents=True)
    rollout_length = 25
    message_budget = 5_000_000
    environment_budget = 1_000_000
    message_cost = 100 + q * rollout_length
    usable = min(
        message_budget // message_cost,
        environment_budget // rollout_length,
    )
    metadata = {
        "q_rollout_workers": q,
        "rollout_length": rollout_length,
        "message_cost_per_update": message_cost,
        "environment_cost_per_update": rollout_length,
        "message_budget": message_budget,
        "environment_budget": environment_budget,
        "usable_updates": usable,
        "charged_training_messages": usable * message_cost,
        "charged_training_environment_ticks": usable * rollout_length,
        "coupling": coupling,
        "seed": seed,
        "upstream_modified": False,
        "harl_commit": commit,
    }
    (run / "tsp_bridge_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    final_transitions = usable * q * rollout_length
    (run / "progress.txt").write_text(
        f"0,{team_return}\n{final_transitions},{team_return}\n", encoding="utf-8"
    )


def _write_controller_result(
    root: Path,
    coupling: str,
    seed: int,
    selected_q: int,
    team_return: float,
    commit: str,
) -> None:
    run = root / f"controller_{coupling}_{seed}"
    run.mkdir(parents=True)
    accounting = MODULE.controller_accounting(
        selected_q=selected_q,
        probe_q=8,
        probe_blocks=128,
        rollout_length=25,
        message_budget=5_000_000,
        environment_budget=1_000_000,
        server_overhead=100,
    ).to_dict()
    metadata = {
        "coupling": coupling,
        "training_seed": seed,
        "message_budget": 5_000_000,
        "environment_budget": 1_000_000,
        "decision": {
            "selected_q": selected_q,
            "certificate": {
                "estimate": 0.0 if coupling == "independent" else 0.9,
                "upper": 0.1 if coupling == "independent" else 0.95,
            },
        },
        "accounting": accounting,
        "selection_overhead_fraction": 0.01,
        "upstream_modified": False,
        "harl_commit": commit,
    }
    (run / "tsp_probe_commit_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (run / "charged_progress.csv").write_text(
        "budget_fraction,team_return\n"
        f"0.01,{team_return}\n1.0,{team_return}\n",
        encoding="utf-8",
    )


def test_synthetic_registered_result_set_passes_all_gates(tmp_path) -> None:
    config_path = ROOT / "experiments" / "marl_probe_commit_development.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commit = config["upstream"]["commit"]
    result_root = tmp_path / "results"
    for coupling in config["coupling_regimes"]:
        for seed in config["seed_registry"]["development_training"]:
            _write_fixed_result(result_root, coupling, seed, 1, -105.0, commit)
            _write_fixed_result(result_root, coupling, seed, 8, -100.0, commit)
            _write_controller_result(
                result_root,
                coupling,
                seed,
                8 if coupling == "independent" else 1,
                -99.0 if coupling == "independent" else -90.0,
                commit,
            )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"pass": True}), encoding="utf-8")
    result = ANALYZER.evaluate_gates(result_root, config_path, audit)
    assert result["run_count"] == 24
    assert result["all_mandatory_gates_pass"] is True
    assert result["decision"] == "authorize-confirmation-preregistration"
    assert result["independent_relative_auc_vs_fixed_q8"] == pytest.approx(0.01)
    assert result["shared_relative_auc_vs_fixed_q8"] == pytest.approx(0.10)


def test_duplicate_registered_result_stops_gate(tmp_path) -> None:
    config_path = ROOT / "experiments" / "marl_probe_commit_development.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commit = config["upstream"]["commit"]
    result_root = tmp_path / "results"
    for coupling in config["coupling_regimes"]:
        for seed in config["seed_registry"]["development_training"]:
            _write_fixed_result(result_root, coupling, seed, 1, -105.0, commit)
            _write_fixed_result(result_root, coupling, seed, 8, -100.0, commit)
            _write_controller_result(
                result_root, coupling, seed, 8, -99.0, commit
            )
    duplicate = result_root / "duplicate"
    duplicate.mkdir()
    source = result_root / "fixed_q8_independent_94001"
    for name in ("tsp_bridge_metadata.json", "progress.txt"):
        (duplicate / name).write_bytes((source / name).read_bytes())
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"pass": True}), encoding="utf-8")
    result = ANALYZER.evaluate_gates(result_root, config_path, audit)
    assert result["gates"]["finite_complete_exact_accounting_and_clean_upstream"] is False
    assert result["decision"] == "stop"


def test_one_sided_lower_t_uses_registered_formula() -> None:
    values = np.arange(1.0, 9.0)
    critical = 1.894578605061305
    expected = values.mean() - critical * values.std(ddof=1) / np.sqrt(8)
    assert CONFIRM_ANALYZER.one_sided_lower_t(values, critical) == pytest.approx(
        expected
    )


def test_synthetic_confirmation_lattice_passes_all_gates(tmp_path) -> None:
    config_path = ROOT / "experiments" / "marl_probe_commit_confirmation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commit = config["upstream"]["commit"]
    result_root = tmp_path / "confirmation"
    for coupling in config["coupling_regimes"]:
        for seed in config["seed_registry"]["confirmation_training"]:
            _write_fixed_result(result_root, coupling, seed, 1, -100.5, commit)
            _write_fixed_result(result_root, coupling, seed, 8, -100.0, commit)
            _write_controller_result(
                result_root,
                coupling,
                seed,
                8 if coupling == "independent" else 1,
                -99.0 if coupling == "independent" else -90.0,
                commit,
            )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"pass": True}), encoding="utf-8")
    result = CONFIRM_ANALYZER.evaluate_confirmation(
        result_root, config_path, audit
    )
    assert result["run_count"] == 48
    assert result["independent_controller_vs_q8_mean"] == pytest.approx(0.01)
    assert result["shared_controller_vs_q8_mean"] == pytest.approx(0.10)
    assert result["shared_controller_vs_q1_lower"] > -0.02
    assert result["all_mandatory_gates_pass"] is True
    assert result["decision"] == "admit-confirmed-return-evidence"


def test_confirmation_registry_is_disjoint_and_complete() -> None:
    development = json.loads(
        (ROOT / "experiments" / "marl_probe_commit_development.json").read_text(
            encoding="utf-8"
        )
    )
    confirmation = json.loads(
        (ROOT / "experiments" / "marl_probe_commit_confirmation.json").read_text(
            encoding="utf-8"
        )
    )
    dev_seeds = set(development["seed_registry"]["development_training"]) | set(
        development["seed_registry"]["development_probe"]
    )
    confirm_training = set(
        confirmation["seed_registry"]["confirmation_training"]
    )
    confirm_probe = set(confirmation["seed_registry"]["confirmation_probe"])
    assert dev_seeds.isdisjoint(confirm_training | confirm_probe)
    assert confirm_training.isdisjoint(confirm_probe)
    assert confirmation["planned_runs"]["total"] == 48
    assert len(confirm_training) == len(confirm_probe) == 8
