"""Execute and validate the frozen EXP-016B CPU pilot.

This file is deliberately separate from :mod:`run_exp016b`, whose static
preregistration interface remains unchanged.  The pilot reads the frozen
scenario, seed, gate, and power files and refuses to run when their registered
configuration hash differs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from run_adaptation_cost_pilot import (
    Action,
    available_updates,
    exact_markov_terminal_mse,
    kalman_log_likelihood,
    oracle_action,
    scheduled_updates,
)
from run_exp016a import THETA_LOW, budget_ray, budgets
from t018_static_scan import Scenario, identification_scale, scenario_grid


TASK = "EXP-016B"
EXPECTED_CONFIGURATION_SHA256 = (
    "6dfdf87521700c2ddae9b81947e0ecc01ee33ebcf5fcda34b09e9e3c3f7f7ee5"
)
LAYERS = ("A_gaussian_mechanism", "B_affine_markov_td_transfer")
POLICIES = (
    "oracle_evaluation_only",
    "always_all",
    "fixed_small_q",
    "information_only",
    "learning_aware",
    "no_delay_learning_aware",
    "message_only_planning_ablation",
    "environment_only_planning_ablation",
)
REGIMES = ("low", "high")
SMOKE_SEEDS = (90340101, 90340102)
AFFINE_CONTRACTION = 0.5
FLOAT_FORMAT = "%.17g"


@dataclass(frozen=True)
class ProbePlan:
    q: int
    b: int
    n: int

    @property
    def action(self) -> Action:
        return Action(self.q, self.b)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_frozen_bundle() -> dict[str, object]:
    root = repository_root()
    files = {
        "manifest": root / "docs" / "exp016b_scenario_manifest.json",
        "gates": root / "docs" / "exp016b_gate_table.json",
        "seeds": root / "docs" / "exp016b_seed_registry.json",
        "power": root / "docs" / "exp016b_power_audit.json",
    }
    bundle = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in files.items()}
    hashes = {str(value.get("configuration_sha256")) for value in bundle.values()}
    if hashes != {EXPECTED_CONFIGURATION_SHA256}:
        raise RuntimeError(f"frozen configuration mismatch: {sorted(hashes)}")
    manifest = bundle["manifest"]
    if manifest.get("scientific_outcomes_present") is not False:
        raise RuntimeError("preregistration manifest is outcome-tainted")
    if tuple(manifest.get("policies", ())) != POLICIES:
        raise RuntimeError("frozen policy order mismatch")
    return bundle


def scenario_lookup() -> dict[str, Scenario]:
    return {scenario.scenario_id: scenario for scenario in scenario_grid()}


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _ar1_path(rng: np.random.RandomState, length: int, variance: float, coefficient: float) -> np.ndarray:
    values = np.empty(length, dtype=np.float64)
    values[0] = rng.normal(scale=math.sqrt(variance))
    innovation = math.sqrt(variance * max(0.0, 1.0 - coefficient * coefficient))
    for index in range(1, length):
        values[index] = coefficient * values[index - 1] + rng.normal(scale=innovation)
    return values


def generate_potential_observations(
    seed: int,
    scenario: Scenario,
    regime: str,
    layer: str,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return common path and cumulative idiosyncratic sums by agent.

    The potential observations are policy independent.  A q-agent observation
    mean at physical time t is ``common[t] + epsilon_prefix[q-1,t] / q``.
    """

    theta = THETA_LOW if regime == "low" else scenario.theta_high
    rng = np.random.RandomState(stable_seed(seed, scenario.scenario_id, regime, layer))
    common = _ar1_path(rng, length, theta, scenario.lam)
    epsilon = rng.normal(size=(scenario.maximum_agents, length))
    return common, np.cumsum(epsilon, axis=0)


def action_observations(
    common: np.ndarray,
    epsilon_prefix: np.ndarray,
    action: Action,
    start_environment: int,
    count: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    indices = start_environment + action.b * np.arange(1, count + 1, dtype=np.int64)
    if int(indices[-1]) >= common.size:
        raise RuntimeError("potential path shorter than charged environment horizon")
    return common[indices] + epsilon_prefix[action.q - 1, indices] / action.q


def identify_from_path(
    observations: np.ndarray,
    scenario: Scenario,
    probe: ProbePlan,
) -> tuple[str, float]:
    direction = math.sqrt(probe.q) * observations
    coefficient = scenario.lam ** probe.b
    low = kalman_log_likelihood(direction, THETA_LOW, probe.q, coefficient)
    high = kalman_log_likelihood(direction, scenario.theta_high, probe.q, coefficient)
    llr = float(high - low)
    return ("high" if llr > 0.0 else "low"), llr


def public_fixed_small_action(
    scenario: Scenario, message_budget: int, environment_budget: int
) -> Action:
    candidates = [Action(2, b) for b in (1, 2, 4, 8)]
    return min(
        candidates,
        key=lambda action: (
            max(
                exact_markov_terminal_mse(
                    theta,
                    scenario.lam,
                    action,
                    message_budget,
                    environment_budget,
                    scenario.overhead,
                    scenario.delay,
                )
                for theta in (THETA_LOW, scenario.theta_high)
            ),
            action.b,
        ),
    )


@lru_cache(maxsize=None)
def registered_probe(scenario: Scenario) -> ProbePlan:
    _scale, raw = identification_scale(scenario)
    return ProbePlan(int(raw["q"]), int(raw["b"]), int(raw["n"]))


def information_only_probe_decision(
    scale: int, identification_threshold_scale: int, probe: ProbePlan
) -> tuple[bool, ProbePlan, str]:
    """Identification-only rule with no learning-value or safety input."""

    return (
        scale >= identification_threshold_scale,
        probe,
        "identification_threshold",
    )


def policy_probe_decision(
    policy: str,
    scale: int,
    scenario_record: Mapping[str, object],
    scenario: Scenario,
    probe: ProbePlan,
) -> tuple[bool, ProbePlan, str]:
    bid = int(scenario_record["B_id"])
    bvalue = scenario_record["B_value"]
    if policy == "learning_aware":
        qualifies = bvalue is not None and scale >= int(bvalue)
        return bool(qualifies), probe, "learning_value_threshold"
    if policy == "no_delay_learning_aware":
        if bvalue is None:
            return False, probe, "zero_delay_planning"
        ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
        ignored_delay_scale = int(math.ceil(scenario.delay / ray.beta_environment))
        threshold = max(bid, int(bvalue) - ignored_delay_scale)
        return scale >= threshold, probe, "zero_delay_planning"
    ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
    message_budget, environment_budget = budgets(scale, ray)
    if policy == "message_only_planning_ablation":
        qualifies = message_budget >= probe.n * (scenario.overhead + probe.q)
        return bool(qualifies), probe, "message_only_feasibility"
    if policy == "environment_only_planning_ablation":
        qualifies = environment_budget >= probe.n * probe.b
        return bool(qualifies), probe, "environment_only_feasibility"
    return False, probe, "no_probe_policy"


def select_after_identification(
    identified: str,
    scenario: Scenario,
    message_budget: int,
    environment_budget: int,
) -> Action:
    if identified == "low":
        return Action(scenario.maximum_agents, 1)
    return oracle_action(
        scenario.theta_high,
        scenario.lam,
        message_budget,
        environment_budget,
        scenario.overhead,
        scenario.delay,
        scenario.maximum_agents,
    )[0]


def execute_affine_td(observations: np.ndarray, delay: int) -> tuple[float, float, int]:
    """Run scalar affine TD with an explicit FIFO delay queue.

    The mean field is h(w)=-a*w with root zero.  Each observation supplies the
    additive Markov reward noise.  Gradients are computed from the predictable
    stale iterate and applied after exactly ``delay`` update opportunities.
    """

    parameter = 1.0
    pending: list[float] = []
    applied = 0
    averaged_sum = 0.0
    # Strictly inside the scalar delayed-affine stability screen.  The rule is
    # fixed by delay and the public contraction, never by an observed outcome.
    step = 0.1 / (AFFINE_CONTRACTION * (delay + 1.0))
    for observation in observations:
        increment = step * (float(observation) - AFFINE_CONTRACTION * parameter)
        pending.append(increment)
        if len(pending) > delay:
            parameter += pending.pop(0)
            applied += 1
            averaged_sum += parameter
    reported = parameter if applied == 0 else averaged_sum / applied
    error = float(reported * reported)
    return error, float((AFFINE_CONTRACTION * reported) ** 2), applied


def simulate_policy(
    seed: int,
    layer: str,
    scenario_record: Mapping[str, object],
    scenario: Scenario,
    budget_point: Mapping[str, object],
    regime: str,
    policy: str,
    common: np.ndarray,
    epsilon_prefix: np.ndarray,
) -> dict[str, object]:
    scale = int(budget_point["scale"])
    ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
    message_budget, environment_budget = budgets(scale, ray)
    theta_true = THETA_LOW if regime == "low" else scenario.theta_high
    baseline = Action(scenario.maximum_agents, 1)
    probe = registered_probe(scenario)
    probe_used = False
    identified: str | None = None
    llr: float | None = None
    decision_basis = "evaluation_or_fixed"
    remaining_message = message_budget
    remaining_environment = environment_budget
    probe_message = 0
    probe_environment = 0

    if policy == "oracle_evaluation_only":
        selected = oracle_action(
            theta_true,
            scenario.lam,
            message_budget,
            environment_budget,
            scenario.overhead,
            scenario.delay,
            scenario.maximum_agents,
        )[0]
    elif policy == "always_all":
        selected = baseline
    elif policy == "fixed_small_q":
        selected = public_fixed_small_action(scenario, message_budget, environment_budget)
    else:
        if policy == "information_only":
            wants_probe, planned_probe, decision_basis = information_only_probe_decision(
                scale, int(scenario_record["B_id"]), probe
            )
        else:
            wants_probe, planned_probe, decision_basis = policy_probe_decision(
                policy, scale, scenario_record, scenario, probe
            )
        probe = planned_probe
        feasible = bool(
            probe.n > 0
            and probe.n * (scenario.overhead + probe.q) <= message_budget
            and probe.n * probe.b <= environment_budget
        )
        probe_used = bool(wants_probe and feasible)
        if probe_used:
            probe_values = action_observations(
                common, epsilon_prefix, probe.action, 0, probe.n
            )
            identified, llr = identify_from_path(probe_values, scenario, probe)
            probe_message = probe.n * (scenario.overhead + probe.q)
            probe_environment = probe.n * probe.b
            remaining_message -= probe_message
            remaining_environment -= probe_environment
            selected = select_after_identification(
                identified, scenario, remaining_message, remaining_environment
            )
        else:
            selected = baseline

    scheduled = scheduled_updates(
        selected,
        remaining_message,
        remaining_environment,
        scenario.overhead,
    )
    observations = action_observations(
        common, epsilon_prefix, selected, probe_environment, scheduled
    )
    usable = max(0, scheduled - scenario.delay)
    if layer == "A_gaussian_mechanism":
        # The downstream learner is a running mean with one registered
        # pseudo-observation at its nonzero initial iterate.  This makes a
        # zero-update fallback/risk finite without substituting analytic MSE.
        estimate = (1.0 + float(np.sum(observations[:usable]))) / (1.0 + usable)
        terminal = float(estimate * estimate)
        parameter_error = float("nan")
        teacher_error = float("nan")
        applied = usable
    elif layer == "B_affine_markov_td_transfer":
        terminal, teacher_error, applied = execute_affine_td(observations, scenario.delay)
        parameter_error = terminal
    else:
        raise ValueError(layer)

    messages_used = probe_message + scheduled * (scenario.overhead + selected.q)
    environment_used = probe_environment + scheduled * selected.b
    finite = bool(
        math.isfinite(terminal)
        and (layer != "B_affine_markov_td_transfer" or math.isfinite(teacher_error))
    )
    return {
        "task": TASK,
        "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "seed": seed,
        "layer": layer,
        "scenario_id": scenario.scenario_id,
        "budget_point": str(budget_point["name"]),
        "scale": scale,
        "regime": regime,
        "policy": policy,
        "theta_true_evaluation_only": theta_true,
        "lambda_public": scenario.lam,
        "maximum_agents": scenario.maximum_agents,
        "delay": scenario.delay,
        "overhead": scenario.overhead,
        "budget_ray": scenario.ray_name,
        "binding_type": str(scenario_record["binding_type"]),
        "effect_profile": str(scenario_record["effect_profile"]),
        "epsilon_safe": scenario.epsilon_safe,
        "B_id": int(scenario_record["B_id"]),
        "B_value": scenario_record["B_value"],
        "B_value_status": str(scenario_record["B_value_status"]),
        "message_budget": message_budget,
        "environment_budget": environment_budget,
        "probe_used": probe_used,
        "certificate_claimed": bool(probe_used and scale >= int(scenario_record["B_id"])),
        "probe_q": probe.q if probe_used else 0,
        "probe_b": probe.b if probe_used else 0,
        "probe_n": probe.n if probe_used else 0,
        "identified_regime": identified,
        "identification_correct": None if identified is None else identified == regime,
        "log_likelihood_ratio": llr,
        "decision_basis": decision_basis,
        "fallback": selected == baseline and not probe_used,
        "selected_q": selected.q,
        "selected_b": selected.b,
        "scheduled_updates": scheduled,
        "usable_updates_after_delay": applied,
        "messages_used": messages_used,
        "environment_used": environment_used,
        "dual_budget_valid": bool(
            messages_used <= message_budget and environment_used <= environment_budget
        ),
        "terminal_learning_risk": terminal,
        "TD_parameter_error": parameter_error,
        "Bellman_teacher_error": teacher_error,
        "finite": finite,
    }


def maximum_path_length(scenario_record: Mapping[str, object], scenario: Scenario) -> int:
    ray = budget_ray(scenario.ray_name, scenario.overhead, scenario.maximum_agents)
    maximum_scale = max(int(point["scale"]) for point in scenario_record["budget_points"])
    _message, environment = budgets(maximum_scale, ray)
    return int(environment + 8 * max(1, scenario.delay) + 16)


def rows_for_seed(seed: int, manifest: Mapping[str, object]) -> list[dict[str, object]]:
    lookup = scenario_lookup()
    rows: list[dict[str, object]] = []
    for scenario_record in manifest["scenarios"]:
        scenario = lookup[str(scenario_record["scenario_id"])]
        length = maximum_path_length(scenario_record, scenario)
        for layer in scenario_record["layers"]:
            for regime in REGIMES:
                common, epsilon_prefix = generate_potential_observations(
                    seed, scenario, regime, layer, length
                )
                for budget_point in scenario_record["budget_points"]:
                    for policy in POLICIES:
                        rows.append(
                            simulate_policy(
                                seed,
                                layer,
                                scenario_record,
                                scenario,
                                budget_point,
                                regime,
                                policy,
                                common,
                                epsilon_prefix,
                            )
                        )
    return rows


def _seed_part(args: tuple[int, str]) -> tuple[int, int, str]:
    seed, output_dir_raw = args
    bundle = load_frozen_bundle()
    output_dir = Path(output_dir_raw)
    rows = rows_for_seed(seed, bundle["manifest"])
    frame = pd.DataFrame(rows)
    frame.sort_values(
        ["layer", "scenario_id", "budget_point", "regime", "policy"],
        inplace=True,
        kind="mergesort",
    )
    target = output_dir / "parts" / f"seed-{seed}.csv"
    frame.to_csv(target, index=False, float_format=FLOAT_FORMAT, lineterminator="\n")
    return seed, len(frame), str(target)


def combine_parts(output_dir: Path, seeds: Sequence[int]) -> Path:
    target = output_dir / "metrics.csv"
    with target.open("wb") as destination:
        for index, seed in enumerate(seeds):
            source = output_dir / "parts" / f"seed-{seed}.csv"
            with source.open("rb") as handle:
                if index:
                    handle.readline()
                shutil.copyfileobj(handle, destination, length=1024 * 1024)
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    output_dir: Path,
    seeds: Sequence[int],
    workers: int,
    *,
    mode: str | None = None,
) -> dict[str, object]:
    bundle = load_frozen_bundle()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    (output_dir / "parts").mkdir(parents=True)
    started = time.time()
    expected_per_seed = int(bundle["manifest"]["layer_A_finite_count"])  # provenance only
    del expected_per_seed
    completed = []
    tasks = [(int(seed), str(output_dir)) for seed in seeds]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for seed, rows, path in executor.map(_seed_part, tasks):
            completed.append({"seed": seed, "rows": rows, "path": path})
            print(f"completed seed {seed}: {rows} rows", flush=True)
    metrics = combine_parts(output_dir, seeds)
    registered_mode = mode or (
        "pilot" if tuple(seeds) == tuple(bundle["seeds"]["pilot_seeds"]) else "smoke"
    )
    if registered_mode not in {"smoke", "pilot", "formal"}:
        raise ValueError(f"unsupported execution mode: {registered_mode}")
    metadata = {
        "task": TASK,
        "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "mode": registered_mode,
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "workers": workers,
        "rows": int(sum(item["rows"] for item in completed)),
        "elapsed_seconds": time.time() - started,
        "metrics_sha256": sha256_file(metrics),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scientific_outcomes_generated": registered_mode in {"pilot", "formal"},
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "pilot"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_frozen_bundle()
    seeds = SMOKE_SEEDS if args.command == "smoke" else tuple(bundle["seeds"]["pilot_seeds"])
    print(json.dumps(run(args.output_dir, seeds, args.workers), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
