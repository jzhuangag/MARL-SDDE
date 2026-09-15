"""Run the frozen PDSG-EST-001 local CPU estimator audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.policy_dependency_sync.alignment_estimator import (
    GradientErrorBudget,
    gaussian_max_norm_rms_bound,
    geometric_mean_variance_factor,
    markov_mean_rms_bound,
    optimized_quadratic_score,
    optimized_score_error_bound,
    sparse_candidate_gradient,
    uniform_optimized_score_error_bound,
)


@dataclass(frozen=True)
class Scenario:
    current_gradient: np.ndarray
    base_gradient: np.ndarray
    displacements: tuple[np.ndarray, ...]
    jacobians: tuple[np.ndarray, ...]
    true_candidates: tuple[np.ndarray, ...]


def _unit(rng: np.random.Generator, dimension: int) -> np.ndarray:
    value = rng.normal(size=dimension)
    norm = np.linalg.norm(value)
    if norm == 0.0:
        value[0] = 1.0
        norm = 1.0
    return value / norm


def _orthogonal_unit(
    rng: np.random.Generator, reference: np.ndarray
) -> np.ndarray:
    for _ in range(32):
        value = rng.normal(size=reference.shape)
        value -= float(value @ reference) * reference
        norm = np.linalg.norm(value)
        if norm > 1e-12:
            return value / norm
    basis = np.zeros_like(reference)
    basis[int(np.argmin(np.abs(reference)))] = 1.0
    basis -= float(basis @ reference) * reference
    return basis / np.linalg.norm(basis)


def build_scenario(
    manifest: dict[str, Any],
    degree: int,
    displacement_norm: float,
    jacobian_lipschitz: float,
    sparse_tail: float,
) -> Scenario:
    """Construct the outcome-free geometry fixed in Amendment 1."""
    grid = manifest["grid"]
    seed = np.random.SeedSequence(
        [
            manifest["scenario_generation_seed"],
            degree,
            int(round(displacement_norm * 10000)),
            int(round(jacobian_lipschitz * 10000)),
            int(round(sparse_tail * 10000)),
        ]
    )
    rng = np.random.default_rng(seed)
    dimension = int(grid["dimension"])
    current = _unit(rng, dimension)
    base_orthogonal = _orthogonal_unit(rng, current)
    base_cosine = float(grid["base_gradient_cosine"])
    base = (
        base_cosine * current
        + np.sqrt(1.0 - base_cosine**2) * base_orthogonal
    )
    cosines = np.asarray(grid["candidate_cosines"][str(degree)], dtype=float)
    if cosines.size != degree:
        raise ValueError("candidate cosine list does not match degree")
    cosines = cosines[rng.permutation(degree)]
    displacements: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    for cosine in cosines:
        tangent = _orthogonal_unit(rng, current)
        nominal = cosine * current + np.sqrt(1.0 - cosine**2) * tangent
        displacement = displacement_norm * _unit(rng, dimension)
        jacobian = np.outer(nominal - base, displacement) / (
            displacement_norm**2
        )
        remainder = (
            0.5
            * jacobian_lipschitz
            * displacement_norm**2
            * _unit(rng, dimension)
        )
        tail = sparse_tail * _unit(rng, dimension)
        displacements.append(displacement)
        jacobians.append(jacobian)
        candidates.append(nominal + remainder + tail)
    return Scenario(
        current_gradient=current,
        base_gradient=base,
        displacements=tuple(displacements),
        jacobians=tuple(jacobians),
        true_candidates=tuple(candidates),
    )


def stationary_ar_noise(
    rng: np.random.Generator,
    sample_count: int,
    correlation: float,
    shape: tuple[int, ...],
    stationary_variance_sum: float,
) -> np.ndarray:
    """Generate a stationary coordinate-independent Gaussian AR(1) block."""
    coordinate_count = int(np.prod(shape))
    coordinate_scale = np.sqrt(stationary_variance_sum / coordinate_count)
    values = np.empty((sample_count, *shape), dtype=float)
    values[0] = rng.normal(scale=coordinate_scale, size=shape)
    innovation_scale = coordinate_scale * np.sqrt(1.0 - correlation**2)
    for index in range(1, sample_count):
        values[index] = correlation * values[index - 1]
        values[index] += rng.normal(scale=innovation_scale, size=shape)
    return values


def _candidate_step(
    current: np.ndarray,
    candidate: np.ndarray,
    curvature: float,
    maximum_step: float,
) -> float:
    norm_squared = float(candidate @ candidate)
    if norm_squared == 0.0:
        return 0.0
    return float(
        min(
            max(float(current @ candidate) / (curvature * norm_squared), 0.0),
            maximum_step,
        )
    )


def _true_and_estimated_scores(
    manifest: dict[str, Any],
    scenario: Scenario,
    estimated_current: np.ndarray,
    estimated_base: np.ndarray,
    estimated_candidates: tuple[np.ndarray, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    grid = manifest["grid"]
    curvature = float(grid["curvature"])
    maximum_step = float(grid["maximum_step"])
    queue = float(grid["queue"])
    edge_cost = float(grid["edge_cost"])
    reset = float(grid["edge_reset_benefit"])
    true_scores = {
        "null": optimized_quadratic_score(
            scenario.current_gradient,
            scenario.base_gradient,
            curvature,
            maximum_step,
        )
    }
    estimated_scores = {
        "null": optimized_quadratic_score(
            estimated_current,
            estimated_base,
            curvature,
            maximum_step,
        )
    }
    for index, (true_candidate, estimated_candidate) in enumerate(
        zip(scenario.true_candidates, estimated_candidates, strict=True)
    ):
        name = f"edge-{index}"
        exact_terms = queue * edge_cost - reset
        true_scores[name] = optimized_quadratic_score(
            scenario.current_gradient,
            true_candidate,
            curvature,
            maximum_step,
        ) + exact_terms
        estimated_scores[name] = optimized_quadratic_score(
            estimated_current,
            estimated_candidate,
            curvature,
            maximum_step,
        ) + exact_terms
    return true_scores, estimated_scores


def _parameter_rows(grid: dict[str, Any]) -> Iterable[tuple[Any, ...]]:
    return itertools.product(
        grid["sample_count_per_half"],
        grid["markov_correlation"],
        grid["dependency_degree"],
        grid["displacement_norm"],
        grid["jacobian_lipschitz"],
        grid["sparse_tail"],
    )


def _is_population(
    row: dict[str, Any], population: dict[str, Any], favorable: bool
) -> bool:
    if favorable:
        return bool(
            row["sample_count_per_half"]
            >= population["sample_count_per_half_min"]
            and row["markov_correlation"]
            <= population["markov_correlation_max"]
            and row["jacobian_lipschitz"] == population["jacobian_lipschitz"]
            and row["sparse_tail"] == population["sparse_tail"]
            and row["true_margin"] >= population["true_best_second_margin_min"]
        )
    return bool(
        row["sample_count_per_half"] == population["sample_count_per_half"]
        and row["markov_correlation"] == population["markov_correlation"]
        and row["jacobian_lipschitz"] == population["jacobian_lipschitz"]
        and row["sparse_tail"] == population["sparse_tail"]
        and row["true_margin"] >= population["true_best_second_margin_min"]
    )


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _median_or_nan(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def run_audit(
    manifest: dict[str, Any],
    seeds: list[int] | None = None,
    parameter_rows: Iterable[tuple[Any, ...]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if manifest.get("experiment_id") != "PDSG-EST-001" or manifest.get(
        "amendment"
    ) != 1:
        raise ValueError("runner requires frozen PDSG-EST-001 Amendment 1")
    grid = manifest["grid"]
    seeds = list(manifest["pilot_seeds"] if seeds is None else seeds)
    parameters = list(
        _parameter_rows(grid) if parameter_rows is None else parameter_rows
    )
    dimension = int(grid["dimension"])
    gradient_trace = float(grid["gradient_noise_trace"])
    jvp_trace = float(grid["jvp_noise_trace"])
    curvature = float(grid["curvature"])
    maximum_step = float(grid["maximum_step"])
    action_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    scenario_rows: dict[tuple[Any, ...], dict[str, Any]] = {}

    for parameter_index, parameters_tuple in enumerate(parameters):
        sample_count, rho, degree, displacement_norm, jacobian_lipschitz, tail = (
            parameters_tuple
        )
        sample_count = int(sample_count)
        rho = float(rho)
        degree = int(degree)
        displacement_norm = float(displacement_norm)
        jacobian_lipschitz = float(jacobian_lipschitz)
        tail = float(tail)
        scenario = build_scenario(
            manifest,
            degree,
            displacement_norm,
            jacobian_lipschitz,
            tail,
        )
        dummy_estimates = tuple(scenario.true_candidates)
        true_scores, _ = _true_and_estimated_scores(
            manifest,
            scenario,
            scenario.current_gradient,
            scenario.base_gradient,
            dummy_estimates,
        )
        sorted_true = sorted(true_scores.items(), key=lambda item: (item[1], item[0]))
        oracle_action = sorted_true[0][0]
        true_margin = float(sorted_true[1][1] - sorted_true[0][1])
        variance_factor = geometric_mean_variance_factor(sample_count, rho)
        current_rms = markov_mean_rms_bound(
            sample_count, rho, gradient_trace
        )
        base_rms = current_rms
        jvp_rms = markov_mean_rms_bound(sample_count, rho, jvp_trace)
        deterministic_bias = 0.5 * jacobian_lipschitz * displacement_norm**2 + tail
        base_coordinate_variance = gradient_trace * variance_factor / (
            sample_count * dimension
        )
        jvp_coordinate_variance = (
            jvp_trace
            * variance_factor
            * displacement_norm**2
            / (sample_count * dimension**2)
        )
        maximum_coordinate_std = float(
            np.sqrt(base_coordinate_variance + jvp_coordinate_variance)
        )
        maximum_candidate_rms = gaussian_max_norm_rms_bound(
            action_count=degree + 1,
            dimension=dimension,
            maximum_coordinate_standard_deviation=maximum_coordinate_std,
            maximum_bias_norm=deterministic_bias,
        )
        uniform_error = uniform_optimized_score_error_bound(
            maximum_step=maximum_step,
            curvature=curvature,
            current_gradient_norm_bound=float(
                grid["current_gradient_norm_bound"]
            ),
            candidate_gradient_norm_bound=float(
                grid["candidate_gradient_norm_bound"]
            ),
            current_gradient_rms_error=current_rms,
            maximum_candidate_error_rms=maximum_candidate_rms,
        )
        relative_error = float(
            np.inf if true_margin <= 0.0 else 2.0 * uniform_error / true_margin
        )
        scenario_key = tuple(parameters_tuple)
        scenario_rows[scenario_key] = {
            "sample_count_per_half": sample_count,
            "markov_correlation": rho,
            "dependency_degree": degree,
            "displacement_norm": displacement_norm,
            "jacobian_lipschitz": jacobian_lipschitz,
            "sparse_tail": tail,
            "true_margin": true_margin,
            "oracle_action": oracle_action,
            "current_gradient_rms": current_rms,
            "maximum_candidate_rms": maximum_candidate_rms,
            "uniform_score_error": uniform_error,
            "relative_uniform_error": relative_error,
        }

        for seed in seeds:
            seed_sequence = np.random.SeedSequence(
                [
                    int(seed),
                    parameter_index,
                    sample_count,
                    int(round(rho * 1000)),
                    degree,
                ]
            )
            control_seed, update_seed = seed_sequence.spawn(2)
            control_rng = np.random.default_rng(control_seed)
            update_rng = np.random.default_rng(update_seed)
            current_samples = scenario.current_gradient + stationary_ar_noise(
                control_rng,
                sample_count,
                rho,
                (dimension,),
                gradient_trace,
            )
            base_samples = scenario.base_gradient + stationary_ar_noise(
                control_rng,
                sample_count,
                rho,
                (dimension,),
                gradient_trace,
            )
            estimated_current = current_samples.mean(axis=0)
            estimated_base = base_samples.mean(axis=0)
            estimated_candidates: list[np.ndarray] = []
            for jacobian, displacement in zip(
                scenario.jacobians, scenario.displacements, strict=True
            ):
                jvp_samples = jacobian + stationary_ar_noise(
                    control_rng,
                    sample_count,
                    rho,
                    (dimension, dimension),
                    jvp_trace,
                )
                estimated_candidates.append(
                    sparse_candidate_gradient(
                        base_samples, jvp_samples, displacement
                    )
                )
            true_scores, estimated_scores = _true_and_estimated_scores(
                manifest,
                scenario,
                estimated_current,
                estimated_base,
                tuple(estimated_candidates),
            )
            selected_action = min(
                estimated_scores, key=lambda action: (estimated_scores[action], action)
            )
            selected_index = (
                -1 if selected_action == "null" else int(selected_action.split("-")[1])
            )
            selected_true_gradient = (
                scenario.base_gradient
                if selected_index < 0
                else scenario.true_candidates[selected_index]
            )
            selected_estimated_gradient = (
                estimated_base
                if selected_index < 0
                else estimated_candidates[selected_index]
            )
            selected_step = _candidate_step(
                estimated_current,
                selected_estimated_gradient,
                curvature,
                maximum_step,
            )
            update_samples = selected_true_gradient + stationary_ar_noise(
                update_rng,
                sample_count,
                rho,
                (dimension,),
                gradient_trace,
            )
            update_gradient = update_samples.mean(axis=0)
            realized_score = float(
                -selected_step * (scenario.current_gradient @ update_gradient)
                + 0.5
                * curvature
                * selected_step**2
                * (update_gradient @ update_gradient)
            )
            if selected_index >= 0:
                realized_score += float(
                    grid["queue"] * grid["edge_cost"]
                    - grid["edge_reset_benefit"]
                )

            for action_index, action in enumerate(true_scores):
                if action == "null":
                    candidate_rms = base_rms
                    true_gradient = scenario.base_gradient
                else:
                    edge_index = int(action.split("-")[1])
                    candidate_rms = GradientErrorBudget(
                        base_rms=base_rms,
                        jvp_rms=jvp_rms,
                        displacement_norm=displacement_norm,
                        taylor_remainder=0.5
                        * jacobian_lipschitz
                        * displacement_norm**2,
                        sparse_tail=tail,
                    ).candidate_rms
                    true_gradient = scenario.true_candidates[edge_index]
                action_bound = optimized_score_error_bound(
                    maximum_step=maximum_step,
                    curvature=curvature,
                    current_gradient_norm_bound=float(
                        grid["current_gradient_norm_bound"]
                    ),
                    candidate_gradient_norm_bound=max(
                        float(grid["candidate_gradient_norm_bound"]),
                        float(np.linalg.norm(true_gradient)),
                    ),
                    current_gradient_rms_error=current_rms,
                    candidate_gradient_rms_error=candidate_rms,
                )
                absolute_error = abs(
                    estimated_scores[action] - true_scores[action]
                )
                action_rows.append(
                    {
                        "parameter_index": parameter_index,
                        "seed": seed,
                        "action_index": action_index,
                        "action": action,
                        "favorable": _is_population(
                            scenario_rows[scenario_key],
                            manifest["favorable_population"],
                            favorable=True,
                        ),
                        "true_score": true_scores[action],
                        "estimated_score": estimated_scores[action],
                        "absolute_score_error": absolute_error,
                        "action_error_bound": action_bound,
                        "error_to_bound": absolute_error / action_bound,
                    }
                )
            seed_row = dict(scenario_rows[scenario_key])
            seed_row.update(
                {
                    "parameter_index": parameter_index,
                    "seed": seed,
                    "selected_action": selected_action,
                    "selected_correct": selected_action == oracle_action,
                    "true_score_regret": true_scores[selected_action]
                    - true_scores[oracle_action],
                    "selected_step": selected_step,
                    "realized_update_score": realized_score,
                    "control_samples": sample_count,
                    "update_samples": sample_count,
                    "charged_samples": 2 * sample_count,
                    "control_rollout_id": f"{parameter_index}:{seed}:control",
                    "update_rollout_id": f"{parameter_index}:{seed}:update",
                }
            )
            seed_rows.append(seed_row)

    favorable_seed_rows = [
        row
        for row in seed_rows
        if _is_population(row, manifest["favorable_population"], favorable=True)
    ]
    adverse_seed_rows = [
        row
        for row in seed_rows
        if _is_population(row, manifest["adverse_population"], favorable=False)
    ]
    favorable_action_rows = [row for row in action_rows if row["favorable"]]
    favorable_relative = [row["relative_uniform_error"] for row in favorable_seed_rows]
    adverse_relative = [row["relative_uniform_error"] for row in adverse_seed_rows]

    expected_seed_rows = len(parameters) * len(seeds)
    expected_action_rows = sum((int(row[2]) + 1) * len(seeds) for row in parameters)
    finite_values = all(
        np.isfinite(value)
        for row in seed_rows
        for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and key != "relative_uniform_error"
    ) and all(
        np.isfinite(value)
        for row in action_rows
        for value in (
            row["true_score"],
            row["estimated_score"],
            row["absolute_score_error"],
            row["action_error_bound"],
            row["error_to_bound"],
        )
    )
    charge_ok = all(
        row["charged_samples"]
        == row["control_samples"] + row["update_samples"]
        == 2 * row["sample_count_per_half"]
        and row["control_rollout_id"] != row["update_rollout_id"]
        for row in seed_rows
    )
    e1 = bool(
        finite_values
        and charge_ok
        and len(seed_rows) == expected_seed_rows
        and len(action_rows) == expected_action_rows
    )
    e2_value = _mean_or_nan(
        [float(row["error_to_bound"]) for row in favorable_action_rows]
    )
    e3_value = _mean_or_nan(
        [float(row["selected_correct"]) for row in favorable_seed_rows]
    )
    e4_value = _median_or_nan(favorable_relative)
    e5_adverse = _median_or_nan(adverse_relative)
    e5_ratio = float(
        e5_adverse / e4_value
        if np.isfinite(e5_adverse) and np.isfinite(e4_value) and e4_value > 0.0
        else np.nan
    )

    monotonic_m = True
    monotonic_rho = True
    for row in scenario_rows.values():
        matched_m = [
            other
            for other in scenario_rows.values()
            if all(
                other[key] == row[key]
                for key in (
                    "markov_correlation",
                    "dependency_degree",
                    "displacement_norm",
                    "jacobian_lipschitz",
                    "sparse_tail",
                )
            )
        ]
        ordered_m = sorted(matched_m, key=lambda item: item["sample_count_per_half"])
        monotonic_m &= all(
            left["current_gradient_rms"] >= right["current_gradient_rms"] - 1e-15
            for left, right in zip(ordered_m, ordered_m[1:])
        )
        matched_rho = [
            other
            for other in scenario_rows.values()
            if all(
                other[key] == row[key]
                for key in (
                    "sample_count_per_half",
                    "dependency_degree",
                    "displacement_norm",
                    "jacobian_lipschitz",
                    "sparse_tail",
                )
            )
        ]
        ordered_rho = sorted(matched_rho, key=lambda item: item["markov_correlation"])
        monotonic_rho &= all(
            left["current_gradient_rms"] <= right["current_gradient_rms"] + 1e-15
            for left, right in zip(ordered_rho, ordered_rho[1:])
        )
    degree_accuracy = {
        str(degree): _mean_or_nan(
            [
                float(row["selected_correct"])
                for row in favorable_seed_rows
                if row["dependency_degree"] == degree
            ]
        )
        for degree in grid["dependency_degree"]
    }
    gates = {
        "E1": e1,
        "E2": e2_value <= 1.0,
        "E3": e3_value >= 0.85,
        "E4": e4_value <= 0.75,
        "E5": e5_ratio >= 1.5,
        "E6": bool(monotonic_m and monotonic_rho),
        "E7": all(
            np.isfinite(value) and value >= 0.75
            for value in degree_accuracy.values()
        ),
        "E8": charge_ok,
    }
    summary = {
        "experiment_id": manifest["experiment_id"],
        "amendment": manifest["amendment"],
        "scenario_count": len(parameters),
        "seed_row_count": len(seed_rows),
        "action_row_count": len(action_rows),
        "favorable_seed_row_count": len(favorable_seed_rows),
        "adverse_seed_row_count": len(adverse_seed_rows),
        "favorable_mean_error_to_bound": e2_value,
        "favorable_ranking_accuracy": e3_value,
        "favorable_median_relative_uniform_error": e4_value,
        "adverse_median_relative_uniform_error": e5_adverse,
        "adverse_to_favorable_relative_error_ratio": e5_ratio,
        "favorable_ranking_accuracy_by_degree": degree_accuracy,
        "gates": gates,
        "primary_pass": all(gates.values()),
        "reproduction_authorized": all(gates.values()),
    }
    return action_rows, seed_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    action_rows, seed_rows, summary = run_audit(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    action_path = args.output / "action_rows.csv"
    seed_path = args.output / "seed_rows.csv"
    _write_csv(action_path, action_rows)
    _write_csv(seed_path, seed_rows)
    summary["manifest_sha256"] = _sha256(args.manifest)
    summary["runner_sha256"] = _sha256(Path(__file__))
    summary["action_rows_sha256"] = _sha256(action_path)
    summary["seed_rows_sha256"] = _sha256(seed_path)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
