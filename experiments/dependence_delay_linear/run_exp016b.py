"""EXP-016B preregistration, static manifest, and workload auditor.

This module intentionally exposes only validate/estimate/dry-run/freeze.  It
cannot generate a scientific trajectory, pilot result, or formal result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
TASK = "EXP-016B"
TITLE = "Premature Adaptation under Finite Learning Horizons"
ERRATUM_PARENT = "b893bb54fef78168774a6c5607e4d7e43e8db2ad"
T018_GRID_HASH = "c5d2dd5ddac7540888d708ab59d4e3954994da018951797a79d05200ef0ee2db"
FINITE_SAMPLE_COUNT = 96
LAYER_B_SAMPLE_COUNT = 48
PILOT_SEEDS = tuple(range(20340101, 20340197))
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
LAYERS = ("A_gaussian_mechanism", "B_affine_markov_td_transfer")
FINITE_BUDGET_NAMES = (
    "below_B_id",
    "at_B_id",
    "Z_midpoint",
    "near_B_value_below",
    "at_B_value",
    "above_B_value",
)
FORBIDDEN_INFORMATION_ONLY_INPUTS = (
    "downstream_risk",
    "wrong_commit_loss",
    "epsilon_safe",
    "oracle_action",
    "hidden_theta",
    "hidden_regime",
    "outcome_data",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_corrected_scan() -> dict[str, object]:
    return json.loads(
        (repository_root() / "docs/t018_corrected_scan_results.json").read_text(
            encoding="utf-8"
        )
    )


def binding_type(record: Mapping[str, object]) -> str:
    message = int(record["message_binding_Z_cells"]) > 0
    environment = int(record["environment_binding_Z_cells"]) > 0
    if message and environment:
        return "dual"
    if message:
        return "message"
    if environment:
        return "environment"
    return "none"


def effect_profile(record: Mapping[str, object]) -> str:
    effect = int(record["effect_Z_cells"])
    registered = int(record["registered_Z_cells"])
    if effect == 0:
        return "neutral_only"
    if effect == registered:
        return "practical_only"
    return "mixed_practical_and_neutral"


def selection_metadata(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "scenario_id": record["scenario_id"],
        "Q": record["Q"],
        "theta_high": record["theta_high"],
        "lambda": record["lambda"],
        "D": record["delay"],
        "overhead": record["overhead"],
        "budget_ray": record["budget_ray"],
        "epsilon_safe": record["epsilon_safe"],
        "binding_type": binding_type(record),
        "effect_profile": effect_profile(record),
    }


def selection_hash(record: Mapping[str, object]) -> str:
    payload = {
        "algorithm": "exp016b_marginal_stratified_sha256_v1",
        "grid_hash": T018_GRID_HASH,
        "metadata": selection_metadata(record),
    }
    return sha256_json(payload)


def _tokens(record: Mapping[str, object]) -> tuple[str, ...]:
    metadata = selection_metadata(record)
    fields = (
        "Q",
        "theta_high",
        "lambda",
        "D",
        "overhead",
        "budget_ray",
        "epsilon_safe",
        "binding_type",
        "effect_profile",
    )
    return tuple(f"{field}={metadata[field]}" for field in fields)


def stratified_hash_sample(
    records: Sequence[Mapping[str, object]], count: int
) -> list[Mapping[str, object]]:
    """Greedy marginal balancing with SHA-256 as the sole tie-breaker."""
    remaining = sorted(records, key=lambda row: (selection_hash(row), row["scenario_id"]))
    selected: list[Mapping[str, object]] = []
    counts: dict[str, int] = {}
    while remaining and len(selected) < count:
        minimum = min(sum(counts.get(token, 0) for token in _tokens(row)) for row in remaining)
        index = next(
            index
            for index, row in enumerate(remaining)
            if sum(counts.get(token, 0) for token in _tokens(row)) == minimum
        )
        chosen = remaining.pop(index)
        selected.append(chosen)
        for token in _tokens(chosen):
            counts[token] = counts.get(token, 0) + 1
    if len(selected) != count:
        raise ValueError(f"requested {count} records from only {len(records)}")
    return selected


def finite_budget_points(record: Mapping[str, object]) -> list[dict[str, int | str]]:
    b_id = int(record["B_id"])
    b_value = int(record["B_value"])
    if b_value == 2_000_001 or b_value <= b_id:
        raise ValueError("finite budget construction requires B_id < B_value without sentinel")
    gap = b_value - b_id
    values = (
        max(0, b_id - 1),
        b_id,
        b_id + (gap - 1) // 2,
        b_value - 1,
        b_value,
        b_value + max(1, math.ceil(0.10 * gap)),
    )
    return [
        {"name": name, "scale": int(scale)}
        for name, scale in zip(FINITE_BUDGET_NAMES, values)
    ]


def censored_budget_points(record: Mapping[str, object]) -> list[dict[str, int | str]]:
    b_id = int(record["B_id"])
    return [
        {"name": "below_B_id", "scale": max(0, b_id - 1)},
        {"name": "at_B_id", "scale": b_id},
    ]


def scenario_entry(record: Mapping[str, object], layers: Sequence[str]) -> dict[str, object]:
    metadata = selection_metadata(record)
    metadata.update(
        {
            "B_id": record["B_id"],
            "B_value": record["B_value"],
            "B_value_status": record["B_value_status"],
            "selection_sha256": selection_hash(record),
            "layers": list(layers),
            "budget_points": (
                finite_budget_points(record)
                if record["B_value_status"] == "finite"
                else censored_budget_points(record)
            ),
            "finite_threshold_gate_eligible": record["B_value_status"] == "finite",
        }
    )
    return metadata


def build_scenario_manifest(scan: Mapping[str, object] | None = None) -> dict[str, object]:
    scan = scan or load_corrected_scan()
    if scan["grid_hash"] != T018_GRID_HASH:
        raise ValueError("corrected T-018 grid hash mismatch")
    finite = [
        row for row in scan["scenario_records"] if row["B_value_status"] == "finite"
    ]
    censored = [
        row
        for row in scan["scenario_records"]
        if row["B_value_status"] == "search_censored"
    ]
    selected = stratified_hash_sample(finite, FINITE_SAMPLE_COUNT)
    layer_b_ids = {
        row["scenario_id"]
        for row in stratified_hash_sample(selected, LAYER_B_SAMPLE_COUNT)
    }
    entries = [
        scenario_entry(
            row,
            LAYERS if row["scenario_id"] in layer_b_ids else (LAYERS[0],),
        )
        for row in selected
    ]
    entries.extend(scenario_entry(row, LAYERS) for row in sorted(censored, key=selection_hash))
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "title": TITLE,
        "source_grid_hash": T018_GRID_HASH,
        "source_erratum_commit": ERRATUM_PARENT,
        "selection_algorithm": {
            "name": "exp016b_marginal_stratified_sha256_v1",
            "rule": "greedily minimize current marginal-stratum counts; break ties by SHA-256 ordering",
            "depends_only_on": ["scenario metadata", "T-018 grid hash", "SHA-256 ordering"],
            "forbidden_dependencies": ["random outcome", "trajectory", "largest observed effect"],
            "strata": [
                "Q", "theta_high", "lambda", "D", "overhead", "budget_ray",
                "epsilon_safe", "binding_type", "effect_profile",
            ],
        },
        "finite_source_count": len(finite),
        "censored_source_count": len(censored),
        "finite_selected_count": len(selected),
        "layer_A_finite_count": len(selected),
        "layer_B_finite_count": len(layer_b_ids),
        "censored_descriptive_count": len(censored),
        "policies": list(POLICIES),
        "regimes": ["low", "high"],
        "layer_specifications": {
            "A_gaussian_mechanism": {
                "mechanism": "actual Gaussian common-factor trajectories and individual observations",
                "learner": "actual downstream learner",
                "analytic_risk_as_observed_outcome": False,
            },
            "B_affine_markov_td_transfer": {
                "mechanism": "existing finite-gap affine delayed Markov TD environment",
                "actual_td_updates": True,
                "actual_delay_queue": True,
                "complete_dual_budget_charging": True,
                "stability_screened_action_catalogue": True,
                "hidden_rho_theta_regime_inputs": False,
                "actor_critic": False,
                "preconditioner": False,
                "cpu_runnable": True,
            },
        },
        "information_only_contract": {
            "allowed_inputs": [
                "public theta pair", "public mixing certificate", "q", "b",
                "overhead", "budget ray",
            ],
            "forbidden_inputs": list(FORBIDDEN_INFORMATION_ONLY_INPUTS),
        },
        "metrics": [
            "empirical_terminal_learning_risk", "oracle_regret",
            "paired_information_only_minus_learning_aware_risk", "S_mean",
            "S_path_descriptive_only", "identification_error", "probe_commit_fallback",
            "selected_q_b", "message_environment_use", "usable_updates_after_delay",
            "crossover_location", "TD_parameter_error_layer_B",
            "Bellman_teacher_error_layer_B",
        ],
        "exposed_commands": ["validate", "estimate", "dry-run", "freeze"],
        "forbidden_execution": ["scientific trajectory", "pilot", "formal", "HPC4", "GPU"],
        "scenarios": entries,
        "scientific_outcomes_present": False,
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def build_gate_table() -> dict[str, object]:
    statements = {
        "P1": "all outputs finite; zero actual dual-budget violations and hidden-state leaks",
        "P2": "below-B_id policies do not fabricate a reliable certificate",
        "P3": "in every Z-active cell information-only probes while learning-aware falls back",
        "P4": "Layer-A practical-Z aggregate paired mean difference is positive, its simultaneous one-sided lower bound is positive, and its point estimate is at least 3% of always-all risk",
        "P5": "at least 60% of registered scenario-level practical-Z families satisfy the P4 sign and 3% point-estimate criteria; denominator is frozen and neutral cells cannot enter it",
        "P6": "all neutral-Z cells are retained and reported without contributing to P4/P5",
        "P7": "at/above-B-value learning-aware adapts and convergent policies have plan agreement without a forced nonzero gap",
        "P8": "aggregate empirical crossover direction lies within or adjacent to the registered [B_id,B_value] bracket",
        "P9": "theorem-facing S_mean upper bound is at most epsilon_safe; S_path is descriptive only",
        "P10": "delay-active, message-binding, and environment-binding subsets are nonempty and match preregistered directional contrasts",
        "P11": "Layer-B affine-TD aggregate strata have the same risk-difference sign as Layer A; every registered task is retained",
        "P12": "clean same-seed rerun reproduces core CSV/JSON byte for byte",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "alpha_familywise_one_sided": 0.01,
        "cell_practical_effect_threshold_relative": 0.03,
        "scenario_level_coverage_gate": 0.60,
        "coverage_denominator": "all registered practical-effect finite scenario families fixed in the scenario manifest",
        "formal_authorization_rule": "no formal run if any mandatory P1-P12 gate fails",
        "gates": [
            {"id": gate_id, "mandatory": True, "statement": statement}
            for gate_id, statement in statements.items()
        ],
        "scientific_outcomes_present": False,
    }


def build_seed_registry() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "pilot_seeds": list(PILOT_SEEDS),
        "pilot_seed_count": len(PILOT_SEEDS),
        "seed_blocking": "common random numbers across policies within layer/scenario/budget/regime",
        "excluded_prior_seed_ranges": ["EXP-014", "EXP-015", "EXP-016A"],
        "formal_seeds": None,
        "formal_seed_status": "not assigned; any mandatory pilot gate failure forbids formal",
        "scientific_outcomes_present": False,
    }


def build_power_audit() -> dict[str, object]:
    alpha = 0.01
    aggregate_families = 8
    standardized_effect = 0.03 / 0.075
    critical = NormalDist().inv_cdf(1.0 - alpha / aggregate_families)
    noncentrality = math.sqrt(len(PILOT_SEEDS)) * standardized_effect
    power = NormalDist().cdf(noncentrality - critical)
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "method": "outcome-free exact paired-mean variance under frozen paired-moment design assumptions with Bonferroni one-sided Gaussian calibration",
        "paired_moment_assumptions": {
            "minimum_practical_mean_relative": 0.03,
            "paired_standard_deviation_upper_relative": 0.075,
            "variance_of_paired_mean": "0.075^2 / n",
        },
        "aggregate_primary_family_count": aggregate_families,
        "familywise_alpha_one_sided": alpha,
        "pilot_seed_count": len(PILOT_SEEDS),
        "critical_z": critical,
        "noncentrality_at_practical_threshold": noncentrality,
        "power_at_practical_threshold": power,
        "power_target": 0.80,
        "power_target_met": power >= 0.80,
        "rare_identification_errors": "descriptive per cell; deterministic theorem/runtime compliance mandatory; aggregate directional calibration only",
        "cvar90": "secondary descriptive unless effective tail sample size is separately shown adequate",
        "scientific_outcomes_present": False,
    }


def build_bundle() -> dict[str, object]:
    manifest = build_scenario_manifest()
    gates = build_gate_table()
    seeds = build_seed_registry()
    power = build_power_audit()
    configuration_sha256 = sha256_json(
        {"manifest": manifest, "gates": gates, "seeds": seeds, "power": power}
    )
    for payload in (manifest, gates, seeds, power):
        payload["configuration_sha256"] = configuration_sha256
    return {"manifest": manifest, "gates": gates, "seeds": seeds, "power": power}


def workload_estimate(manifest: Mapping[str, object]) -> dict[str, object]:
    layer_a_finite_cells = int(manifest["layer_A_finite_count"]) * 6 * 2
    layer_b_finite_cells = int(manifest["layer_B_finite_count"]) * 6 * 2
    censored_cells_per_layer = int(manifest["censored_descriptive_count"]) * 2 * 2
    layer_a_cells = layer_a_finite_cells + censored_cells_per_layer
    layer_b_cells = layer_b_finite_cells + censored_cells_per_layer
    policy_count = len(manifest["policies"])
    seed_count = len(PILOT_SEEDS)
    layer_a_trajectories = layer_a_cells * policy_count * seed_count
    layer_b_trajectories = layer_b_cells * policy_count * seed_count
    cpu_hours = (layer_a_trajectories * 0.008 + layer_b_trajectories * 0.025) / 3600.0
    rows = layer_a_trajectories + layer_b_trajectories
    return {
        "layer_A_cells": layer_a_cells,
        "layer_B_cells": layer_b_cells,
        "logical_cell_count": layer_a_cells + layer_b_cells,
        "policy_count": policy_count,
        "pilot_seed_count": seed_count,
        "layer_A_trajectories": layer_a_trajectories,
        "layer_B_trajectories": layer_b_trajectories,
        "estimated_trajectories": rows,
        "estimated_single_process_cpu_hours": cpu_hours,
        "estimated_peak_memory_gb": 8.0,
        "estimated_disk_gb": rows * 800.0 / 1e9,
        "recommended_execution": "local CPU",
        "within_local_limits": cpu_hours <= 6.0 and 8.0 <= 32.0 and rows * 800.0 / 1e9 <= 20.0,
        "estimate_basis": "static cell/operation accounting only; no trajectory benchmark or scientific outcome",
    }


def validate_bundle(bundle: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    manifest = bundle["manifest"]
    gates = bundle["gates"]
    seeds = bundle["seeds"]
    power = bundle["power"]
    hashes = {payload["configuration_sha256"] for payload in bundle.values()}
    if len(hashes) != 1:
        errors.append("configuration hashes disagree")
    if manifest["source_grid_hash"] != T018_GRID_HASH:
        errors.append("source grid hash mismatch")
    if len(manifest["policies"]) != 8 or set(manifest["policies"]) != set(POLICIES):
        errors.append("policy registry mismatch")
    if [gate["id"] for gate in gates["gates"]] != [f"P{i}" for i in range(1, 13)]:
        errors.append("mandatory gate registry mismatch")
    if not all(gate["mandatory"] for gate in gates["gates"]):
        errors.append("all P1-P12 gates must be mandatory")
    if len(seeds["pilot_seeds"]) != 96 or len(set(seeds["pilot_seeds"])) != 96:
        errors.append("pilot seed registry mismatch")
    if not power["power_target_met"]:
        errors.append("outcome-free power target not met")
    scenarios = manifest["scenarios"]
    finite = [row for row in scenarios if row["B_value_status"] == "finite"]
    censored = [row for row in scenarios if row["B_value_status"] == "search_censored"]
    if len(finite) != FINITE_SAMPLE_COUNT or len(censored) != 8:
        errors.append("scenario sample count mismatch")
    for row in finite:
        if row["B_value"] == 2_000_001:
            errors.append(f"sentinel used by {row['scenario_id']}")
        if [point["name"] for point in row["budget_points"]] != list(FINITE_BUDGET_NAMES):
            errors.append(f"finite budget registry mismatch for {row['scenario_id']}")
    for row in censored:
        if row["B_value"] is not None or row["finite_threshold_gate_eligible"]:
            errors.append(f"censored scenario entered finite gates: {row['scenario_id']}")
        if any("B_value" in point["name"] for point in row["budget_points"]):
            errors.append(f"censored scenario has B_value-derived point: {row['scenario_id']}")
    finite_values = {field: set() for field in ("Q", "theta_high", "lambda", "D", "overhead", "budget_ray", "epsilon_safe", "binding_type")}
    for row in finite:
        for field in finite_values:
            finite_values[field].add(row[field])
    expected_minima = {"Q": 3, "theta_high": 4, "lambda": 4, "D": 4, "overhead": 3, "budget_ray": 3, "epsilon_safe": 2, "binding_type": 2}
    for field, minimum in expected_minima.items():
        if len(finite_values[field]) < minimum:
            errors.append(f"stratum coverage missing for {field}")
    if not any(row["effect_profile"] == "mixed_practical_and_neutral" for row in finite):
        errors.append("neutral Z population missing")
    if any(payload["scientific_outcomes_present"] for payload in bundle.values()):
        errors.append("scientific outcome flag must be false")
    if not workload_estimate(manifest)["within_local_limits"]:
        errors.append("static workload exceeds local limits")
    return errors


def preregistration_markdown(bundle: Mapping[str, object]) -> str:
    manifest = bundle["manifest"]
    workload = workload_estimate(manifest)
    return f"""# EXP-016B preregistration: {TITLE}

This is a prospective, outcome-free preregistration following the independent
T-018 erratum commit `{ERRATUM_PARENT}`. It authorizes no trajectory in this
commit and does not revive EXP-016A.

## Question and scope

When reliable identification is feasible but identification plus delay cost
is not yet amortized by remaining learning benefit, does learning-aware
fallback outperform information-only probing? The primary contrast is inside
the finite learning-value zone, not after both policies converge above
`B_value`.

## Frozen design

- Configuration SHA-256: `{manifest["configuration_sha256"]}`
- Corrected T-018 grid: `{T018_GRID_HASH}`
- Finite scenarios: {manifest["finite_selected_count"]} for Layer A; {manifest["layer_B_finite_count"]} for Layer B
- Search-censored descriptive scenarios: {manifest["censored_descriptive_count"]}, excluded from finite-threshold gates
- Policies: {len(POLICIES)}
- Fresh paired pilot seeds: {len(PILOT_SEEDS)}
- Layers: Gaussian common-factor learner and affine delayed Markov TD transfer

Layer A must use actual common-factor trajectories, individual observations,
and the actual downstream learner; analytic risk is not an observed outcome.
Layer B must use actual TD updates and delay queues, complete dual-budget
charging, and a stability-screened catalogue, with no actor-critic,
preconditioner, or hidden rho/theta/regime input.

The information-only policy may use only the public model pair, mixing
certificate, selected `(q,b)`, overhead, and budget ray. It may not access:
{", ".join(FORBIDDEN_INFORMATION_ONLY_INPUTS)}.

## Static execution decision

Estimated single-process CPU is `{workload["estimated_single_process_cpu_hours"]:.3f}`
hours, peak memory `{workload["estimated_peak_memory_gb"]:.1f}` GB, and disk
`{workload["estimated_disk_gb"]:.3f}` GB. Local CPU is recommended for a
future pilot after this commit. HPC4, `/project`, and GPU remain unauthorized.
"""


def analysis_plan_markdown(bundle: Mapping[str, object]) -> str:
    gates = bundle["gates"]
    return f"""# EXP-016B analysis plan

The primary estimand is the CRN-paired mean terminal-risk difference
`risk(information_only) - risk(learning_aware)` in the registered Layer-A
practical-effect finite-Z population. Report its relative form against
always-all risk, oracle regret, identification error, probe/commit/fallback,
selected `(q,b)`, message/environment use, usable post-delay updates, and
crossover location. Layer B additionally reports TD parameter error and
Bellman/teacher error.

All continuous primary families use paired seed-block means and simultaneous
one-sided Bonferroni bounds at familywise alpha
`{gates["alpha_familywise_one_sided"]}`. Statistical direction (lower bound
above zero) and practical magnitude (point estimate at least 3%) are separate.
Neutral-Z cells remain in their frozen descriptive table and cannot rescue
P4. P5 uses the frozen scenario-level coverage denominator and threshold
`{gates["scenario_level_coverage_gate"]:.0%}`. `S_mean` alone is compared with
`epsilon_safe`; `S_path` and CVaR90 are descriptive.

Identification certification is a deterministic theorem/runtime compliance
gate. Empirical identification error is calibrated only over registered
aggregate families; per-cell rare errors are descriptive implementation
anomaly checks, never the infeasible EXP-016A 64-seed rare-event gate.

If Layer A passes but Layer B lacks aggregate directional consistency, no
multi-agent Markov-learning transfer claim is allowed. Any mandatory P1--P12
failure forbids a formal run. A clean same-seed rerun must reproduce core
CSV/JSON byte for byte before formal authorization.
"""


def power_markdown(power: Mapping[str, object]) -> str:
    assumptions = power["paired_moment_assumptions"]
    return f"""# EXP-016B outcome-free power audit

No trajectory was generated. For the frozen aggregate paired design, the
minimum practical mean is `{assumptions["minimum_practical_mean_relative"]}`
and the design upper paired SD is
`{assumptions["paired_standard_deviation_upper_relative"]}`. Hence the exact
paired-mean variance under these frozen moments is `0.075^2/n`. With
{power["pilot_seed_count"]} fresh seeds, {power["aggregate_primary_family_count"]}
Bonferroni families, and one-sided familywise alpha
`{power["familywise_alpha_one_sided"]}`, the threshold noncentrality is
`{power["noncentrality_at_practical_threshold"]:.6f}`, critical z is
`{power["critical_z"]:.6f}`, and design power is
`{power["power_at_practical_threshold"]:.6f}`.

The paired-SD bound is a prospective design assumption, not an observed
result. If a pilot shows it is violated, power feasibility fails and formal is
not authorized. CVaR90 remains descriptive unless a later frozen audit proves
adequate effective tail size.
"""


def freeze() -> dict[str, object]:
    bundle = build_bundle()
    errors = validate_bundle(bundle)
    if errors:
        raise ValueError("; ".join(errors))
    root = repository_root()
    targets = {
        "docs/exp016b_scenario_manifest.json": json.dumps(bundle["manifest"], indent=2, sort_keys=True) + "\n",
        "docs/exp016b_gate_table.json": json.dumps(bundle["gates"], indent=2, sort_keys=True) + "\n",
        "docs/exp016b_seed_registry.json": json.dumps(bundle["seeds"], indent=2, sort_keys=True) + "\n",
        "docs/exp016b_power_audit.json": json.dumps(bundle["power"], indent=2, sort_keys=True) + "\n",
        "docs/exp016b_preregistration.md": preregistration_markdown(bundle),
        "docs/exp016b_analysis_plan.md": analysis_plan_markdown(bundle),
        "docs/exp016b_power_audit.md": power_markdown(bundle["power"]),
    }
    for relative, content in targets.items():
        (root / relative).write_text(content, encoding="utf-8")
    return {
        "written": list(targets),
        "configuration_sha256": bundle["manifest"]["configuration_sha256"],
        "workload": workload_estimate(bundle["manifest"]),
        "scientific_outcomes_generated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        default="validate",
        choices=("validate", "estimate", "dry-run", "freeze"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    errors = validate_bundle(bundle)
    if errors:
        raise SystemExit("\n".join(errors))
    if args.command == "validate":
        output = {"status": "valid", "scientific_outcomes_generated": False}
    elif args.command == "estimate":
        output = workload_estimate(bundle["manifest"])
        output["scientific_outcomes_generated"] = False
    elif args.command == "dry-run":
        output = {
            "status": "static enumeration valid",
            "configuration_sha256": bundle["manifest"]["configuration_sha256"],
            "workload": workload_estimate(bundle["manifest"]),
            "scientific_outcomes_generated": False,
        }
    else:
        output = freeze()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
