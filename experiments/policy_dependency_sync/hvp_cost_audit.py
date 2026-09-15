"""Frozen local-CPU audit of batched cross-policy alignment derivatives."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, observation_dim: int, hidden: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(observation), dim=-1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_manifest(manifest: dict) -> None:
    if manifest.get("experiment_id") != "PDSG-COMP-001":
        raise ValueError("unexpected experiment id")
    runtime = manifest["runtime"]
    if runtime["device"] != "cpu" or runtime["dtype"] != "float32":
        raise ValueError("this audit is frozen to CPU float32")
    if int(runtime["warmup_repetitions"]) < 1:
        raise ValueError("at least one warmup is required")
    if int(runtime["measured_repetitions"]) < 2:
        raise ValueError("at least two measurements are required")
    model = manifest["model_grid"]
    if min(model["agents"]) < 3:
        raise ValueError("at least three agents are required")
    if min(model["requested_degrees"]) < 1:
        raise ValueError("degrees must be positive")
    if int(model["owner"]) != 0:
        raise ValueError("the frozen owner is zero")
    if len(manifest["seeds"]) != len(set(manifest["seeds"])):
        raise ValueError("seeds must be unique")


def configurations(manifest: dict) -> list[dict]:
    model = manifest["model_grid"]
    result = []
    for agents, hidden, batch, requested_degree, seed in itertools.product(
        model["agents"],
        model["hidden_widths"],
        model["batch_sizes"],
        model["requested_degrees"],
        manifest["seeds"],
    ):
        result.append(
            {
                "agents": int(agents),
                "hidden": int(hidden),
                "batch": int(batch),
                "requested_degree": int(requested_degree),
                "degree": min(int(requested_degree), int(agents) - 1),
                "seed": int(seed),
            }
        )
    return result


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
    observations = tuple(
        torch.randn(config["batch"], int(model["observation_dimension"]))
        for _ in range(config["agents"])
    )
    owner_advantage = torch.randn(int(model["action_dimension"]))
    donors = tuple(range(1, 1 + config["degree"]))
    interactions = {
        donor: torch.randn(
            int(model["action_dimension"]), int(model["action_dimension"])
        )
        / math.sqrt(float(model["action_dimension"]))
        for donor in donors
    }
    return {
        "actors": actors,
        "observations": observations,
        "owner_advantage": owner_advantage,
        "donors": donors,
        "interactions": interactions,
    }


def _surrogate(state: dict) -> torch.Tensor:
    probabilities = tuple(
        actor(observation)
        for actor, observation in zip(
            state["actors"], state["observations"], strict=True
        )
    )
    owner_probability = probabilities[0]
    loss = 0.1 * torch.mean(owner_probability @ state["owner_advantage"])
    for donor in state["donors"]:
        transformed = owner_probability @ state["interactions"][donor]
        loss = loss + torch.mean(
            torch.sum(transformed * probabilities[donor], dim=-1)
        ) / len(state["donors"])
    return loss


def measured_call(state: dict, alignment_hvp: bool) -> tuple[float, float]:
    started = time.perf_counter_ns()
    loss = _surrogate(state)
    owner_parameters = tuple(state["actors"][0].parameters())
    owner_gradient = torch.autograd.grad(
        loss,
        owner_parameters,
        create_graph=alignment_hvp,
    )
    if alignment_hvp:
        alignment = sum(
            torch.sum(value * value.detach()) for value in owner_gradient
        )
        donor_parameters = tuple(
            parameter
            for donor in state["donors"]
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


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return float(ordered[lower])
    weight = location - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def run_configuration(config: dict, manifest: dict) -> dict:
    state = _build_state(config, manifest)
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
            if alignment_hvp:
                hvp_times.append(elapsed)
                hvp_norms.append(norm)
            else:
                baseline_times.append(elapsed)
                baseline_norms.append(norm)

    baseline_median = statistics.median(baseline_times)
    hvp_median = statistics.median(hvp_times)
    return {
        **config,
        "actor_parameters": sum(
            parameter.numel() for parameter in state["actors"].parameters()
        ),
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


def evaluate_gates(rows: list[dict]) -> dict:
    ratios = [row["median_ratio"] for row in rows]
    grouped: dict[tuple[int, int, int, int], list[float]] = {}
    for row in rows:
        key = (row["agents"], row["hidden"], row["batch"], row["degree"])
        grouped.setdefault(key, []).append(row["hvp_median_ms"])
    group_medians = {key: statistics.median(value) for key, value in grouped.items()}
    degree_scaling = []
    base_keys = {(key[0], key[1], key[2]) for key in group_medians}
    for base in sorted(base_keys):
        available = sorted(
            key[3] for key in group_medians if key[:3] == base
        )
        if len(available) >= 2 and available[0] != available[-1]:
            degree_scaling.append(
                group_medians[(*base, available[-1])]
                / group_medians[(*base, available[0])]
            )
    gates = {
        "C1_finite_positive": all(
            row["finite"]
            and row["baseline_median_ms"] > 0.0
            and row["hvp_median_ms"] > 0.0
            and row["baseline_min_norm"] > 0.0
            and row["hvp_min_norm"] > 0.0
            for row in rows
        ),
        "C2_nonzero_cross_derivatives": all(
            row["hvp_min_norm"] > 0.0 for row in rows
        ),
        "C3_median_ratio_le_4": statistics.median(ratios) <= 4.0,
        "C4_maximum_ratio_le_6": max(ratios) <= 6.0,
        "C5_degree_scaling_le_1_35": bool(degree_scaling)
        and max(degree_scaling) <= 1.35,
    }
    return {
        "gates": gates,
        "primary_pass": all(gates.values()),
        "median_ratio": statistics.median(ratios),
        "maximum_ratio": max(ratios),
        "maximum_degree_scaling": max(degree_scaling),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if arguments.validate:
        print(json.dumps({"valid": True, "configurations": len(configurations(manifest))}))
        return
    if arguments.output is None:
        parser.error("--output is required unless --validate is used")
    result = execute(arguments.manifest, arguments.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
