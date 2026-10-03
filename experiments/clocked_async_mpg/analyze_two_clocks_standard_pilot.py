"""Analyze the frozen Two Clocks standard-environment pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .run_two_clocks_standard_pilot import (
    METHODS,
    PROFILES,
    barrier_update_count,
    packet_opportunities,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_gain(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / max(abs(baseline), 1.0)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty contrast")
    return sum(values) / len(values)


def _rows_by_key(summary: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in summary["rows"]:
        key = (row["service_profile"], int(row["seed"]), row["method"])
        if key in rows:
            raise RuntimeError(f"duplicate pilot row {key}")
        rows[key] = row
    return rows


def _validate_summary(
    summary: dict[str, Any],
    config: dict[str, Any],
    config_sha256: str,
    required_code_commit: str,
) -> list[str]:
    errors: list[str] = []
    task = summary.get("task")
    if task not in config["tasks"]:
        return ["unregistered task summary"]
    if summary.get("experiment_id") != config["experiment_id"]:
        errors.append("experiment identifier mismatch")
    if summary.get("config_sha256") != config_sha256:
        errors.append("configuration hash mismatch")
    if summary.get("code_commit") != required_code_commit:
        errors.append("code commit mismatch")
    if summary.get("formal_authorized") is not False:
        errors.append("pilot summary makes a formal claim")
    if summary.get("methods") != list(METHODS):
        errors.append("method registry mismatch")
    if summary.get("profiles") != list(PROFILES):
        errors.append("profile registry mismatch")
    expected_seeds = [int(value) for value in config["pilot_seeds"][task]]
    if summary.get("seeds") != expected_seeds:
        errors.append("seed registry mismatch")
    rows = _rows_by_key(summary)
    expected_keys = {
        (profile, seed, method)
        for profile in PROFILES
        for seed in expected_seeds
        for method in METHODS
    }
    if set(rows) != expected_keys:
        errors.append("pilot row population is incomplete")
        return errors
    task_config = config["tasks"][task]
    episode_length = int(task_config["episode_length"])
    baseline_steps = int(config["baseline_episodes"]) * episode_length
    for (profile, _, method), row in rows.items():
        services = [float(value) for value in task_config["service_profiles"][profile]]
        horizon = float(task_config["logical_horizon"])
        packets = packet_opportunities(services, horizon)
        updates = packets if method != "frozen_barrier" else barrier_update_count(services, horizon)
        charged_steps = baseline_steps + packets * episode_length
        scalar_keys = (
            "initial_return",
            "terminal_return",
            "return_change",
            "logical_time_auc",
            "maximum_self_fresh_error",
            "mean_event_delay",
            "clipped_packet_fraction",
        )
        if not all(math.isfinite(float(row[key])) for key in scalar_keys):
            errors.append(f"non-finite outcome in {(profile, method)}")
        if int(row["completed_packets"]) != packets:
            errors.append(f"packet accounting mismatch in {(profile, method)}")
        if int(row["optimizer_updates"]) != updates:
            errors.append(f"update accounting mismatch in {(profile, method)}")
        if int(row["charged_environment_steps"]) != charged_steps:
            errors.append(f"environment charging mismatch in {(profile, method)}")
        if int(row["charged_actor_transitions"]) != charged_steps * int(
            task_config["agents"]
        ):
            errors.append(f"actor-transition charging mismatch in {(profile, method)}")
        if float(row["maximum_self_fresh_error"]) > 1e-10:
            errors.append(f"self-fresh invariant failed in {(profile, method)}")
        curve = row["curve"]
        if len(curve) != len(config["evaluation_fractions"]):
            errors.append(f"evaluation curve incomplete in {(profile, method)}")
    return errors


def analyze(
    *,
    primary: dict[str, dict[str, Any]],
    reproduction: dict[str, dict[str, Any]],
    primary_bytes: dict[str, bytes],
    reproduction_bytes: dict[str, bytes],
    config: dict[str, Any],
    config_sha256: str,
    required_code_commit: str,
    manifests_verified: bool,
) -> dict[str, Any]:
    expected_tasks = set(config["tasks"])
    if set(primary) != expected_tasks or set(reproduction) != expected_tasks:
        raise RuntimeError("analysis requires primary and reproduction for every task")
    validation_errors: list[str] = []
    for task in sorted(expected_tasks):
        validation_errors.extend(
            f"{task}: {error}"
            for error in _validate_summary(
                primary[task], config, config_sha256, required_code_commit
            )
        )
        validation_errors.extend(
            f"{task} reproduction: {error}"
            for error in _validate_summary(
                reproduction[task], config, config_sha256, required_code_commit
            )
        )
    reproducible = all(
        primary_bytes[task] == reproduction_bytes[task] for task in expected_tasks
    )

    contrasts: list[dict[str, Any]] = []
    for task in sorted(expected_tasks):
        rows = _rows_by_key(primary[task])
        for profile in PROFILES:
            for seed in config["pilot_seeds"][task]:
                async_row = rows[(profile, int(seed), "two_clocks_async")]
                delay_row = rows[(profile, int(seed), "delay_scaled_async")]
                barrier_row = rows[(profile, int(seed), "frozen_barrier")]
                contrasts.append(
                    {
                        "task": task,
                        "profile": profile,
                        "seed": int(seed),
                        "async_vs_barrier_auc_relative_gain": _relative_gain(
                            float(async_row["logical_time_auc"]),
                            float(barrier_row["logical_time_auc"]),
                        ),
                        "async_vs_delay_auc_relative_gain": _relative_gain(
                            float(async_row["logical_time_auc"]),
                            float(delay_row["logical_time_auc"]),
                        ),
                        "async_vs_barrier_terminal_relative_gain": _relative_gain(
                            float(async_row["terminal_return"]),
                            float(barrier_row["terminal_return"]),
                        ),
                        "async_return_change": float(async_row["return_change"]),
                        "async_updates": int(async_row["optimizer_updates"]),
                        "barrier_updates": int(barrier_row["optimizer_updates"]),
                    }
                )

    heterogeneous = [row for row in contrasts if row["profile"] == "heterogeneous"]
    balanced = [row for row in contrasts if row["profile"] == "balanced"]
    heterogeneous_auc = [row["async_vs_barrier_auc_relative_gain"] for row in heterogeneous]
    delay_auc = [row["async_vs_delay_auc_relative_gain"] for row in heterogeneous]
    threshold = config["mandatory_gates"]
    task_metrics: dict[str, Any] = {}
    for task in sorted(expected_tasks):
        task_heterogeneous = [row for row in heterogeneous if row["task"] == task]
        task_balanced = [row for row in balanced if row["task"] == task]
        task_metrics[task] = {
            "heterogeneous_auc_relative_gain": _mean(
                [row["async_vs_barrier_auc_relative_gain"] for row in task_heterogeneous]
            ),
            "balanced_auc_relative_gain": _mean(
                [row["async_vs_barrier_auc_relative_gain"] for row in task_balanced]
            ),
            "heterogeneous_terminal_relative_gain": _mean(
                [row["async_vs_barrier_terminal_relative_gain"] for row in task_heterogeneous]
            ),
            "heterogeneous_async_return_change": _mean(
                [row["async_return_change"] for row in task_heterogeneous]
            ),
            "heterogeneous_update_ratio": _mean(
                [row["async_updates"] / row["barrier_updates"] for row in task_heterogeneous]
            ),
        }

    all_primary_rows = [row for summary in primary.values() for row in summary["rows"]]
    gates = {
        "P1_validity_and_equal_work": len(validation_errors) == 0,
        "P2_adaptive_depth_ratio_minimum": all(
            values["heterogeneous_update_ratio"]
            >= float(threshold["P2_adaptive_depth_ratio_minimum"])
            for values in task_metrics.values()
        ),
        "P3_heterogeneous_auc_improvement_minimum": (
            _mean(heterogeneous_auc)
            >= float(threshold["P3_heterogeneous_auc_improvement_minimum"])
            and all(
                values["heterogeneous_auc_relative_gain"] > 0.0
                for values in task_metrics.values()
            )
        ),
        "P4_positive_heterogeneous_cells_minimum": sum(
            value > 0.0 for value in heterogeneous_auc
        )
        >= int(threshold["P4_positive_heterogeneous_cells_minimum"]),
        "P5_delay_scaled_comparison": (
            _mean(delay_auc) >= float(threshold["P5_delay_scaled_auc_tolerance"])
            and sum(value > 0.0 for value in delay_auc)
            >= int(threshold["P5_delay_scaled_strict_cells_minimum"])
        ),
        "P6_heterogeneous_gain_exceeds_balanced": all(
            values["heterogeneous_auc_relative_gain"]
            > values["balanced_auc_relative_gain"]
            for values in task_metrics.values()
        ),
        "P7_terminal_relative_shortfall_floor": all(
            values["heterogeneous_terminal_relative_gain"]
            >= float(threshold["P7_terminal_relative_shortfall_floor"])
            for values in task_metrics.values()
        ),
        "P8_positive_learning_change_each_task": all(
            values["heterogeneous_async_return_change"] > 0.0
            for values in task_metrics.values()
        ),
        "P9_clipped_packet_fraction_maximum": _mean(
            [float(row["clipped_packet_fraction"]) for row in all_primary_rows]
        )
        <= float(threshold["P9_clipped_packet_fraction_maximum"]),
        "P10_byte_reproducibility": reproducible,
        "P11_provenance": manifests_verified,
        "P12_no_formal_seeds_or_claim": (
            config["formal_seeds"] == []
            and all(summary["formal_authorized"] is False for summary in primary.values())
            and all(summary["formal_authorized"] is False for summary in reproduction.values())
        ),
    }
    all_passed = all(gates.values())
    return {
        "experiment_id": config["experiment_id"],
        "scope": "pilot validation only; not formal evidence",
        "required_code_commit": required_code_commit,
        "config_sha256": config_sha256,
        "primary_sha256": {
            task: hashlib.sha256(primary_bytes[task]).hexdigest()
            for task in sorted(expected_tasks)
        },
        "reproduction_sha256": {
            task: hashlib.sha256(reproduction_bytes[task]).hexdigest()
            for task in sorted(expected_tasks)
        },
        "validation_errors": validation_errors,
        "contrasts": contrasts,
        "task_metrics": task_metrics,
        "aggregate": {
            "heterogeneous_auc_relative_gain": _mean(heterogeneous_auc),
            "heterogeneous_positive_cells": sum(value > 0.0 for value in heterogeneous_auc),
            "async_vs_delay_auc_relative_gain": _mean(delay_auc),
            "async_vs_delay_positive_cells": sum(value > 0.0 for value in delay_auc),
            "mean_clipped_packet_fraction": _mean(
                [float(row["clipped_packet_fraction"]) for row in all_primary_rows]
            ),
        },
        "gates": gates,
        "all_mandatory_gates_passed": all_passed,
        "formal_authorized": False,
        "decision": (
            "pilot gates passed; a separate formal preregistration may be designed"
            if all_passed
            else "pilot failed; stop without formal escalation or gate changes"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--primary-mamujoco", type=Path, required=True)
    parser.add_argument("--primary-smacv2", type=Path, required=True)
    parser.add_argument("--reproduction-mamujoco", type=Path, required=True)
    parser.add_argument("--reproduction-smacv2", type=Path, required=True)
    parser.add_argument("--required-code-commit", required=True)
    parser.add_argument("--manifests-verified", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = {
        "primary": {
            "mamujoco_ant_4x2": args.primary_mamujoco,
            "smacv2_terran_5v5": args.primary_smacv2,
        },
        "reproduction": {
            "mamujoco_ant_4x2": args.reproduction_mamujoco,
            "smacv2_terran_5v5": args.reproduction_smacv2,
        },
    }
    bytes_by_run = {
        run: {task: path.read_bytes() for task, path in task_paths.items()}
        for run, task_paths in paths.items()
    }
    summaries = {
        run: {
            task: json.loads(bytes_by_run[run][task].decode("utf-8"))
            for task in task_paths
        }
        for run, task_paths in paths.items()
    }
    config_bytes = args.config.read_bytes()
    result = analyze(
        primary=summaries["primary"],
        reproduction=summaries["reproduction"],
        primary_bytes=bytes_by_run["primary"],
        reproduction_bytes=bytes_by_run["reproduction"],
        config=json.loads(config_bytes.decode("utf-8")),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        required_code_commit=args.required_code_commit,
        manifests_verified=args.manifests_verified,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"output_sha256={_sha256(args.output)}")
    print(f"all_mandatory_gates_passed={result['all_mandatory_gates_passed']}")


if __name__ == "__main__":
    main()
