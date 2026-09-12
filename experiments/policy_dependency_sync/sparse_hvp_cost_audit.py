"""Frozen CPU cost audit for a local-factor cross-policy statistic."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from experiments.policy_dependency_sync.hvp_cost_audit import (
    Actor,
    _percentile,
    sha256_file,
)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("experiment_id") != "PDSG-COMP-002":
        raise ValueError("unexpected experiment id")
    runtime = manifest["runtime"]
    if runtime["device"] != "cpu" or runtime["dtype"] != "float32":
        raise ValueError("this audit is frozen to CPU float32")
    if int(runtime["warmup_repetitions"]) < 1:
        raise ValueError("at least one warmup is required")
    if int(runtime["measured_repetitions"]) < 2:
        raise ValueError("at least two measurements are required")
    model = manifest["model_grid"]
    if min(model["agents"]) <= max(model["local_degrees"]):
        raise ValueError("every global system must contain a non-neighbor")
    if int(model["owner"]) != 0:
        raise ValueError("the frozen owner is zero")
    if len(manifest["seeds"]) != len(set(manifest["seeds"])):
        raise ValueError("seeds must be unique")


def configurations(manifest: dict) -> list[dict]:
    model = manifest["model_grid"]
    return [
        {
            "agents": int(agents),
            "hidden": int(hidden),
            "batch": int(batch),
            "degree": int(degree),
            "seed": int(seed),
        }
        for agents, hidden, batch, degree, seed in itertools.product(
            model["agents"],
            model["hidden_widths"],
            model["batch_sizes"],
            model["local_degrees"],
            manifest["seeds"],
        )
    ]


def _build_state(config: dict, manifest: dict) -> dict:
    model = manifest["model_grid"]
    torch.manual_seed(config["seed"])
    actors = nn.ModuleList(
        Actor(
            int(model["observation_dimension"]),
            config["hidden"],
            int(model["action_dimension"]),
        )
        for _ in range(config["agents"])
    )
    local_indices = tuple(range(0, config["degree"] + 1))
    generator = torch.Generator().manual_seed(config["seed"] + 4_000_003)
    observations = {
        index: torch.randn(
            config["batch"],
            int(model["observation_dimension"]),
            generator=generator,
        )
        for index in local_indices
    }
    owner_advantage = torch.randn(
        int(model["action_dimension"]), generator=generator
    )
    interactions = {
        donor: torch.randn(
            int(model["action_dimension"]),
            int(model["action_dimension"]),
            generator=generator,
        )
        / math.sqrt(float(model["action_dimension"]))
        for donor in local_indices[1:]
    }
    return {
        "actors": actors,
        "local_indices": local_indices,
        "observations": observations,
        "owner_advantage": owner_advantage,
        "interactions": interactions,
        "outside_index": config["degree"] + 1,
    }


def _local_surrogate(state: dict) -> torch.Tensor:
    probabilities = {
        index: state["actors"][index](state["observations"][index])
        for index in state["local_indices"]
    }
    owner_probability = probabilities[0]
    loss = 0.1 * torch.mean(owner_probability @ state["owner_advantage"])
    donors = state["local_indices"][1:]
    for donor in donors:
        transformed = owner_probability @ state["interactions"][donor]
        loss = loss + torch.mean(
            torch.sum(transformed * probabilities[donor], dim=-1)
        ) / len(donors)
    return loss


def measured_call(state: dict, alignment_hvp: bool) -> tuple[float, float]:
    started = time.perf_counter_ns()
    loss = _local_surrogate(state)
    owner_parameters = tuple(state["actors"][0].parameters())
    owner_gradient = torch.autograd.grad(
        loss, owner_parameters, create_graph=alignment_hvp
    )
    if alignment_hvp:
        alignment = sum(
            torch.sum(value * value.detach()) for value in owner_gradient
        )
        donor_parameters = tuple(
            parameter
            for donor in state["local_indices"][1:]
            for parameter in state["actors"][donor].parameters()
        )
        cross_gradient = torch.autograd.grad(alignment, donor_parameters)
        derivative_norm = math.sqrt(
            sum(float(torch.sum(value.detach() ** 2)) for value in cross_gradient)
        )
    else:
        derivative_norm = math.sqrt(
            sum(float(torch.sum(value.detach() ** 2)) for value in owner_gradient)
        )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return elapsed_ms, derivative_norm


def structural_support_check(state: dict) -> tuple[bool, bool]:
    loss = _local_surrogate(state)
    owner_gradient = torch.autograd.grad(
        loss, tuple(state["actors"][0].parameters()), create_graph=True
    )
    alignment = sum(torch.sum(value * value.detach()) for value in owner_gradient)
    donor_parameters = tuple(
        parameter
        for donor in state["local_indices"][1:]
        for parameter in state["actors"][donor].parameters()
    )
    outside_parameters = tuple(state["actors"][state["outside_index"]].parameters())
    gradients = torch.autograd.grad(
        alignment,
        donor_parameters + outside_parameters,
        allow_unused=True,
    )
    local = gradients[: len(donor_parameters)]
    outside = gradients[len(donor_parameters) :]
    local_nonzero = all(
        value is not None and float(torch.linalg.vector_norm(value.detach())) > 0.0
        for value in local
    )
    outside_unused = all(value is None for value in outside)
    return local_nonzero, outside_unused


def run_configuration(config: dict, manifest: dict) -> dict:
    state = _build_state(config, manifest)
    local_nonzero, outside_unused = structural_support_check(state)
    runtime = manifest["runtime"]
    for repetition in range(int(runtime["warmup_repetitions"])):
        order = (False, True) if repetition % 2 == 0 else (True, False)
        for alignment_hvp in order:
            measured_call(state, alignment_hvp)

    baseline_times: list[float] = []
    hvp_times: list[float] = []
    baseline_norms: list[float] = []
    hvp_norms: list[float] = []
    for repetition in range(int(runtime["measured_repetitions"])):
        order = (False, True) if repetition % 2 == 0 else (True, False)
        for alignment_hvp in order:
            elapsed, norm = measured_call(state, alignment_hvp)
            destination_times = hvp_times if alignment_hvp else baseline_times
            destination_norms = hvp_norms if alignment_hvp else baseline_norms
            destination_times.append(elapsed)
            destination_norms.append(norm)
    baseline_median = statistics.median(baseline_times)
    hvp_median = statistics.median(hvp_times)
    return {
        **config,
        "global_actor_parameters": sum(
            parameter.numel() for parameter in state["actors"].parameters()
        ),
        "touched_actor_blocks": len(state["local_indices"]),
        "local_derivatives_nonzero": local_nonzero,
        "outside_derivatives_unused": outside_unused,
        "baseline_median_ms": baseline_median,
        "baseline_p90_ms": _percentile(baseline_times, 0.9),
        "hvp_median_ms": hvp_median,
        "hvp_p90_ms": _percentile(hvp_times, 0.9),
        "median_ratio": hvp_median / baseline_median,
        "baseline_min_norm": min(baseline_norms),
        "hvp_min_norm": min(hvp_norms),
        "finite": all(
            math.isfinite(value)
            for value in baseline_times + hvp_times + baseline_norms + hvp_norms
        ),
    }


def _group_medians(rows: list[dict]) -> dict[tuple[int, int, int, int], float]:
    grouped: dict[tuple[int, int, int, int], list[float]] = {}
    for row in rows:
        key = (row["agents"], row["hidden"], row["batch"], row["degree"])
        grouped.setdefault(key, []).append(row["hvp_median_ms"])
    return {key: statistics.median(values) for key, values in grouped.items()}


def evaluate_gates(rows: list[dict]) -> dict:
    ratios = [row["median_ratio"] for row in rows]
    medians = _group_medians(rows)
    global_scaling = []
    for hidden, batch, degree in sorted(
        {(key[1], key[2], key[3]) for key in medians}
    ):
        values = [
            value
            for (agents_key, hidden_key, batch_key, degree_key), value in medians.items()
            if (hidden_key, batch_key, degree_key) == (hidden, batch, degree)
        ]
        global_scaling.append(max(values) / min(values))
    degree_scaling = []
    for agents, hidden, batch in sorted(
        {(key[0], key[1], key[2]) for key in medians}
    ):
        degrees = sorted(
            (key[3], value)
            for key, value in medians.items()
            if key[:3] == (agents, hidden, batch)
        )
        degree_scaling.extend(
            high_value / low_value
            for (_, low_value), (_, high_value) in zip(
                degrees[:-1], degrees[1:], strict=True
            )
        )
    gates = {
        "L1_finite_positive": all(
            row["finite"]
            and row["baseline_median_ms"] > 0.0
            and row["hvp_median_ms"] > 0.0
            and row["baseline_min_norm"] > 0.0
            and row["hvp_min_norm"] > 0.0
            for row in rows
        ),
        "L2_local_nonzero_outside_unused": all(
            row["local_derivatives_nonzero"] and row["outside_derivatives_unused"]
            for row in rows
        ),
        "L3_median_ratio_le_2_5": statistics.median(ratios) <= 2.5,
        "L4_maximum_ratio_le_4": max(ratios) <= 4.0,
        "L5_global_scaling_le_1_35": max(global_scaling) <= 1.35,
        "L6_adjacent_degree_scaling_le_2_2": max(degree_scaling) <= 2.2,
    }
    return {
        "gates": gates,
        "primary_pass": all(gates.values()),
        "median_ratio": statistics.median(ratios),
        "maximum_ratio": max(ratios),
        "maximum_global_scaling": max(global_scaling),
        "maximum_adjacent_degree_scaling": max(degree_scaling),
        "global_scaling_ratios": global_scaling,
        "degree_scaling_ratios": degree_scaling,
    }


def execute(manifest_path: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    torch.set_num_threads(int(manifest["runtime"]["torch_threads"]))
    rows = [run_configuration(config, manifest) for config in configurations(manifest)]
    result = {
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "configuration_rows": rows,
        "summary": evaluate_gates(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def compare(primary_path: Path, reproduction_path: Path) -> dict:
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
    primary_summary = primary["summary"]
    reproduction_summary = reproduction["summary"]
    if not primary_summary["primary_pass"]:
        raise ValueError("the frozen stopping rule forbids reproduction")
    ratios = {}
    for key in ("median_ratio", "maximum_ratio"):
        first = float(primary_summary[key])
        second = float(reproduction_summary[key])
        ratios[key] = abs(second - first) / first
    return {
        "same_manifest": primary["manifest_sha256"] == reproduction["manifest_sha256"],
        "same_gate_decisions": primary_summary["gates"] == reproduction_summary["gates"],
        "relative_differences": ratios,
        "L7_pass": (
            primary["manifest_sha256"] == reproduction["manifest_sha256"]
            and primary_summary["gates"] == reproduction_summary["gates"]
            and max(ratios.values()) <= 0.15
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compare-primary", type=Path)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if arguments.validate:
        print(json.dumps({"valid": True, "configurations": len(configurations(manifest))}))
        return
    if arguments.output is None:
        parser.error("--output is required unless --validate is used")
    result = execute(arguments.manifest, arguments.output)
    if arguments.compare_primary is None:
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                compare(arguments.compare_primary, arguments.output),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
