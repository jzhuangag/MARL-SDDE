"""Frozen P1--P12 analysis for EXP-016B pilot output."""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from run_exp016b_pilot import (
    EXPECTED_CONFIGURATION_SHA256,
    LAYERS,
    POLICIES,
    load_frozen_bundle,
    repository_root,
    scenario_lookup,
    sha256_file,
)
from run_exp016a import Action
from t018_static_scan import (
    DELTA,
    PRACTICAL_EFFECT_THRESHOLD,
    THETA_LOW,
    _risk_for_action,
    asymptotic_action,
)


KEYS = ["seed", "layer", "scenario_id", "budget_point", "scale", "regime"]
Z_CRITICAL = 3.0233414397391534


@lru_cache(maxsize=1)
def corrected_scenario_records() -> dict[str, Mapping[str, object]]:
    source = repository_root() / "docs" / "t018_corrected_scan_results.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {str(item["scenario_id"]): item for item in payload["scenario_records"]}


def analytic_cell_tags(manifest: Mapping[str, object]) -> pd.DataFrame:
    lookup = scenario_lookup()
    corrected = corrected_scenario_records()
    rows = []
    for record in manifest["scenarios"]:
        scenario = lookup[str(record["scenario_id"])]
        bid = int(record["B_id"])
        bvalue = record["B_value"]
        probe = corrected[scenario.scenario_id]["id_probe"]
        fallback = Action(scenario.maximum_agents, 1)
        high_action = asymptotic_action(
            scenario.theta_high,
            scenario.lam,
            scenario.overhead,
            scenario.ray_name,
            scenario.maximum_agents,
        )
        for point in record["budget_points"]:
            scale = int(point["scale"])
            in_z = bool(bvalue is not None and bid <= scale < int(bvalue))
            for regime, theta, correct in (
                ("low", THETA_LOW, fallback),
                ("high", scenario.theta_high, high_action),
            ):
                fallback_risk = _risk_for_action(theta, scenario, scale, fallback)
                correct_risk = _risk_for_action(theta, scenario, scale, correct, probe)
                wrong = high_action if regime == "low" else fallback
                wrong_risk = _risk_for_action(theta, scenario, scale, wrong, probe)
                information_risk = (1.0 - DELTA) * correct_risk + DELTA * wrong_risk
                relative = (information_risk - fallback_risk) / max(fallback_risk, 1e-15)
                rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "budget_point": str(point["name"]),
                        "scale": scale,
                        "regime": regime,
                        "in_Z": in_z,
                        "practical_Z": bool(
                            in_z
                            and (
                                math.isinf(relative)
                                or relative >= PRACTICAL_EFFECT_THRESHOLD
                            )
                        ),
                        "neutral_Z": bool(in_z and relative < PRACTICAL_EFFECT_THRESHOLD),
                        "analytic_relative_difference": relative,
                        "delay_active": scenario.delay > 0,
                        "binding_type": str(record["binding_type"]),
                        "finite_threshold_gate_eligible": bool(
                            record["finite_threshold_gate_eligible"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def paired_summary(values: pd.DataFrame) -> dict[str, float | int | bool]:
    seed = values.groupby("seed", sort=True).agg(
        difference=("difference", "mean"), baseline=("always_all", "mean")
    )
    mean = float(seed["difference"].mean())
    standard_error = float(seed["difference"].std(ddof=1) / math.sqrt(len(seed)))
    lower = mean - Z_CRITICAL * standard_error
    baseline = float(seed["baseline"].mean())
    relative = mean / max(baseline, np.finfo(float).eps)
    return {
        "seed_blocks": int(len(seed)),
        "paired_mean_difference": mean,
        "paired_standard_error": standard_error,
        "simultaneous_one_sided_lower": lower,
        "always_all_mean": baseline,
        "relative_difference": relative,
        "positive": bool(mean > 0.0),
        "lower_positive": bool(lower > 0.0),
        "practical_3_percent": bool(relative >= 0.03),
    }


def validate_safety(manifest: Mapping[str, object]) -> dict[str, object]:
    lookup = scenario_lookup()
    corrected = corrected_scenario_records()
    margins = []
    for record in manifest["scenarios"]:
        if record["B_value_status"] != "finite":
            continue
        scenario = lookup[str(record["scenario_id"])]
        source_record = corrected[scenario.scenario_id]
        if int(source_record["B_value"]) != int(record["B_value"]):
            raise RuntimeError(f"B_value reproduction mismatch: {scenario.scenario_id}")
        value_probe = source_record["value_probe"]
        margins.append(
            {
                "scenario_id": scenario.scenario_id,
                "safety_relative": float(value_probe["safety_relative"]),
                "epsilon_safe": scenario.epsilon_safe,
                "pass": bool(float(value_probe["safety_relative"]) <= scenario.epsilon_safe),
            }
        )
    return {
        "scenario_count": len(margins),
        "all_pass": bool(all(item["pass"] for item in margins)),
        "maximum_safety_relative": float(max(item["safety_relative"] for item in margins)),
        "minimum_slack": float(
            min(item["epsilon_safe"] - item["safety_relative"] for item in margins)
        ),
    }


def analyze(metrics_path: Path, output_dir: Path) -> dict[str, object]:
    bundle = load_frozen_bundle()
    manifest = bundle["manifest"]
    frame = pd.read_csv(metrics_path)
    expected_rows = 1_376_256
    required = {
        "configuration_sha256",
        "finite",
        "dual_budget_valid",
        "certificate_claimed",
        "terminal_learning_risk",
        "probe_used",
        "selected_q",
        "selected_b",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"metrics columns missing: {missing}")
    if set(frame["configuration_sha256"].astype(str)) != {EXPECTED_CONFIGURATION_SHA256}:
        raise RuntimeError("metrics configuration hash mismatch")

    policy_counts = frame.groupby("policy", sort=True).size().to_dict()
    complete_policy_grid = set(policy_counts) == set(POLICIES) and len(set(policy_counts.values())) == 1
    pivot = frame.pivot(index=KEYS, columns="policy", values="terminal_learning_risk").reset_index()
    pivot["difference"] = pivot["information_only"] - pivot["learning_aware"]
    pivot["oracle_regret_information_only"] = (
        pivot["information_only"] - pivot["oracle_evaluation_only"]
    )
    tags = analytic_cell_tags(manifest)
    pivot = pivot.merge(
        tags,
        on=["scenario_id", "budget_point", "scale", "regime"],
        how="left",
        validate="many_to_one",
    )

    primary = pivot[
        (pivot["layer"] == LAYERS[0])
        & pivot["finite_threshold_gate_eligible"]
        & pivot["practical_Z"]
    ].copy()
    p4 = paired_summary(primary)

    scenario_rows = []
    for scenario_id, group in primary.groupby("scenario_id", sort=True):
        result = paired_summary(group)
        scenario_rows.append({"scenario_id": scenario_id, **result})
    scenario_frame = pd.DataFrame(scenario_rows)
    scenario_pass = scenario_frame["positive"] & scenario_frame["practical_3_percent"]
    coverage = float(scenario_pass.mean()) if len(scenario_frame) else 0.0

    below = frame[frame["scale"] < frame["B_id"]]
    p2 = bool(not below["certificate_claimed"].astype(bool).any())
    z_info = frame[
        (frame["layer"] == LAYERS[0])
        & (frame["policy"] == "information_only")
        & frame["B_value"].notna()
        & (frame["scale"] >= frame["B_id"])
        & (frame["scale"] < frame["B_value"])
    ]
    z_aware = frame[
        (frame["layer"] == LAYERS[0])
        & (frame["policy"] == "learning_aware")
        & frame["B_value"].notna()
        & (frame["scale"] >= frame["B_id"])
        & (frame["scale"] < frame["B_value"])
    ]
    p3 = bool(len(z_info) and z_info["probe_used"].astype(bool).all() and not z_aware["probe_used"].astype(bool).any())

    neutral = pivot[(pivot["layer"] == LAYERS[0]) & pivot["neutral_Z"]]
    p6 = bool(len(neutral) > 0 and neutral[list(POLICIES)].notna().all().all())

    high_budget = frame[
        (frame["layer"] == LAYERS[0])
        & frame["B_value"].notna()
        & (frame["scale"] >= frame["B_value"])
        & frame["policy"].isin(("information_only", "learning_aware"))
    ]
    plans = high_budget.pivot(
        index=["seed", "scenario_id", "budget_point", "scale", "regime"],
        columns="policy",
        values=["probe_used", "selected_q", "selected_b"],
    )
    p7 = bool(
        (plans[("probe_used", "information_only")] == plans[("probe_used", "learning_aware")]).all()
        and (plans[("selected_q", "information_only")] == plans[("selected_q", "learning_aware")]).all()
        and (plans[("selected_b", "information_only")] == plans[("selected_b", "learning_aware")]).all()
    )

    z_mean = float(primary["difference"].mean())
    above = pivot[
        (pivot["layer"] == LAYERS[0])
        & pivot["finite_threshold_gate_eligible"]
        & pivot["scale"].ge(
            pivot["scenario_id"].map(
                {str(item["scenario_id"]): item["B_value"] for item in manifest["scenarios"]}
            )
        )
    ]
    p8 = bool(z_mean > 0.0 and np.allclose(above["difference"], 0.0, atol=0.0, rtol=0.0))

    safety = validate_safety(manifest)

    subset_results = {}
    for name, mask in {
        "delay_active": primary["delay_active"],
        "message_binding": primary["binding_type"].eq("message"),
        "environment_binding": primary["binding_type"].eq("environment"),
    }.items():
        subset = primary[mask]
        result = paired_summary(subset) if len(subset) else {"positive": False, "relative_difference": float("nan")}
        subset_results[name] = {"rows": int(len(subset)), **result}
    p10 = bool(all(item["rows"] > 0 and item["positive"] for item in subset_results.values()))

    layer_b = pivot[
        (pivot["layer"] == LAYERS[1])
        & pivot["finite_threshold_gate_eligible"]
        & pivot["practical_Z"]
    ]
    layer_b_result = paired_summary(layer_b)
    expected_layer_b_tasks = sum(LAYERS[1] in item["layers"] for item in manifest["scenarios"])
    observed_layer_b_tasks = int(frame[frame["layer"] == LAYERS[1]]["scenario_id"].nunique())
    p11 = bool(
        p4["positive"] == layer_b_result["positive"]
        and layer_b_result["positive"]
        and observed_layer_b_tasks == expected_layer_b_tasks
    )

    p1 = bool(
        len(frame) == expected_rows
        and frame["finite"].astype(bool).all()
        and frame["dual_budget_valid"].astype(bool).all()
        and complete_policy_grid
        and not frame.duplicated(KEYS + ["policy"]).any()
    )
    gate_results = {
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "P4": bool(p4["positive"] and p4["lower_positive"] and p4["practical_3_percent"]),
        "P5": bool(coverage >= float(bundle["gates"]["scenario_level_coverage_gate"])),
        "P6": p6,
        "P7": p7,
        "P8": p8,
        "P9": bool(safety["all_pass"]),
        "P10": p10,
        "P11": p11,
        "P12": False,
    }

    cell_summary = (
        frame.groupby(
            ["layer", "scenario_id", "budget_point", "scale", "regime", "policy"],
            as_index=False,
            sort=True,
        )
        .agg(
            mean_terminal_risk=("terminal_learning_risk", "mean"),
            identification_error=("identification_correct", lambda x: float(1.0 - pd.Series(x).dropna().astype(bool).mean()) if pd.Series(x).notna().any() else float("nan")),
            probe_rate=("probe_used", "mean"),
            fallback_rate=("fallback", "mean"),
            mean_messages=("messages_used", "mean"),
            mean_environment=("environment_used", "mean"),
            mean_usable_updates=("usable_updates_after_delay", "mean"),
            mean_td_parameter_error=("TD_parameter_error", "mean"),
            mean_bellman_teacher_error=("Bellman_teacher_error", "mean"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_path = output_dir / "cell_summary.csv"
    scenario_path = output_dir / "scenario_primary_summary.csv"
    cell_summary.to_csv(cell_path, index=False, float_format="%.17g", lineterminator="\n")
    scenario_frame.to_csv(scenario_path, index=False, float_format="%.17g", lineterminator="\n")

    summary = {
        "task": "EXP-016B-pilot",
        "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "rows": int(len(frame)),
        "seeds": int(frame["seed"].nunique()),
        "primary_layer_A": p4,
        "scenario_level_coverage": coverage,
        "scenario_level_denominator": int(len(scenario_frame)),
        "neutral_Z_rows": int(len(neutral)),
        "subset_results": subset_results,
        "safety_certificate": safety,
        "layer_B": layer_b_result,
        "expected_layer_B_tasks": expected_layer_b_tasks,
        "observed_layer_B_tasks": observed_layer_b_tasks,
        "gate_results": gate_results,
        "mandatory_gates_pass_before_reproduction": bool(all(gate_results[f"P{i}"] for i in range(1, 12))),
        "formal_authorized": False,
        "formal_reason": "P12 requires a clean byte-identical rerun; any P1-P11 failure permanently stops formal for this pilot",
        "cell_summary_sha256": sha256_file(cell_path),
        "scenario_summary_sha256": sha256_file(scenario_path),
    }
    core_results = {
        "task": summary["task"],
        "configuration_sha256": summary["configuration_sha256"],
        "metrics_sha256": summary["metrics_sha256"],
        "rows": summary["rows"],
        "seeds": summary["seeds"],
        "primary_layer_A": summary["primary_layer_A"],
        "scenario_level_coverage": summary["scenario_level_coverage"],
        "scenario_level_denominator": summary["scenario_level_denominator"],
        "neutral_Z_rows": summary["neutral_Z_rows"],
        "subset_results": summary["subset_results"],
        "safety_certificate": summary["safety_certificate"],
        "layer_B": summary["layer_B"],
        "expected_layer_B_tasks": summary["expected_layer_B_tasks"],
        "observed_layer_B_tasks": summary["observed_layer_B_tasks"],
        "gate_results_P1_P11": {
            key: value for key, value in gate_results.items() if key != "P12"
        },
        "cell_summary_sha256": summary["cell_summary_sha256"],
        "scenario_summary_sha256": summary["scenario_summary_sha256"],
    }
    core_path = output_dir / "core_results.json"
    core_path.write_text(
        json.dumps(core_results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary["core_results_sha256"] = sha256_file(core_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(analyze(args.metrics, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
