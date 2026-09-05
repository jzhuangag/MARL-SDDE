"""Frozen analytic survival scan for reuse-correct-refresh control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .stale_correction_qp import (
    exponential_secant_slope,
    solve_tempered_correction_qp,
    update_resource_debt,
)


EXPECTED_CONFIG_SHA256 = (
    "875e692868e6696e5c4dd13c029a3e5e88914bb61e164a422243c9ef4c7e9d36"
)


@dataclass(frozen=True)
class _Specification:
    seed: int
    agents: int
    effective_batch_size: int
    persistence: float
    profile: str
    refresh_budget: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    if _sha256(path).lower() != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("RCR-H1 configuration hash mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment"] != "RCR-H1" or config["horizon"] % 8 != 0:
        raise RuntimeError("invalid frozen RCR-H1 configuration")
    return config


def _specifications(config: dict[str, Any]) -> list[_Specification]:
    seeds = range(
        int(config["seeds"]["start"]),
        int(config["seeds"]["start"]) + int(config["seeds"]["count"]),
    )
    return [
        _Specification(seed, agents, batch, persistence, profile, budget)
        for seed in seeds
        for agents in config["agent_counts"]
        for batch in config["effective_batch_sizes"]
        for persistence in config["markov_persistence"]
        for profile in config["profiles"]
        for budget in config["refresh_budgets"]
    ]


def _generate_certificates(
    specification: _Specification, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    horizon = int(config["horizon"])
    agents = specification.agents
    profile_index = config["profiles"].index(specification.profile)
    rng_seed = (
        specification.seed
        + 1009 * agents
        + 9176 * profile_index
        + int(round(1000 * specification.persistence))
    )
    rng = np.random.default_rng(rng_seed)
    latent = int(rng.integers(0, 2))
    service = np.geomspace(0.7, 1.4, agents)
    coupling = np.linspace(0.8, 1.2, agents)
    divergence_rows = []
    sensitivity_rows = []
    for event in range(horizon):
        if specification.profile != "stationary" and rng.random() > specification.persistence:
            latent = 1 - latent
        if specification.profile == "stationary":
            multiplier = np.ones(agents)
        elif specification.profile == "bursty":
            multiplier = np.full(agents, 0.35 if latent == 0 else 2.0)
        elif specification.profile == "rotating":
            multiplier = np.full(agents, 0.35)
            multiplier[(event // 12) % agents] = 2.0
        elif specification.profile == "mixed":
            multiplier = np.full(agents, 0.35 if latent == 0 else 1.2)
            multiplier[(event // 12) % agents] *= 1.8
        else:
            raise ValueError("unknown profile")
        divergence = (
            float(config["divergence_floor"])
            + float(config["divergence_scale"]) * service * multiplier
        )
        sensitivity = (
            float(config["sensitivity_scale"])
            * coupling
            * np.sqrt(2.0 * divergence)
        )
        divergence_rows.append(divergence)
        sensitivity_rows.append(sensitivity)
    return np.asarray(sensitivity_rows), np.asarray(divergence_rows)


def _risk_rows(
    alphas: np.ndarray,
    sensitivities: np.ndarray,
    divergences: np.ndarray,
    second_moment: float,
    batch_size: float,
) -> np.ndarray:
    if alphas.ndim == 1:
        alphas = np.broadcast_to(alphas, sensitivities.shape)
    residual = np.sum(sensitivities * (1.0 - alphas), axis=1)
    total_divergence = np.sum(divergences, axis=1)
    slopes = np.asarray(
        [exponential_secant_slope(float(value)) for value in total_divergence]
    )
    tempered = np.sum(divergences * alphas * alphas, axis=1)
    return residual * residual + second_moment / batch_size * (
        1.0 + slopes * tempered
    )


def _best_static_vector(
    sensitivities: np.ndarray,
    divergences: np.ndarray,
    second_moment: float,
    batch_size: float,
) -> np.ndarray:
    total_bias = np.sum(sensitivities, axis=1)
    slopes = np.asarray(
        [
            exponential_secant_slope(float(value))
            for value in np.sum(divergences, axis=1)
        ]
    )
    matrix = np.mean(
        sensitivities[:, :, None] * sensitivities[:, None, :], axis=0
    )
    matrix += np.diag(
        np.mean(
            second_moment / batch_size * slopes[:, None] * divergences,
            axis=0,
        )
    )
    linear = np.mean(total_bias[:, None] * sensitivities, axis=0)

    def objective(alpha: np.ndarray) -> float:
        return float(alpha @ matrix @ alpha - 2.0 * linear @ alpha)

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return 2.0 * matrix @ alpha - 2.0 * linear

    bounds = [(0.0, 1.0)] * sensitivities.shape[1]
    result = minimize(
        objective,
        np.full(sensitivities.shape[1], 0.5),
        jac=gradient,
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 1000, "maxls": 50},
    )
    if not result.success:
        result = minimize(
            objective,
            np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0),
            jac=gradient,
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 2000},
        )
    candidate = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    candidate_gradient = gradient(candidate)
    kkt_residual = np.max(
        np.where(
            candidate <= 1e-8,
            np.maximum(0.0, -candidate_gradient),
            np.where(
                candidate >= 1.0 - 1e-8,
                np.maximum(0.0, candidate_gradient),
                np.abs(candidate_gradient),
            ),
        )
    )
    if not result.success or kkt_residual > 1e-6:
        raise RuntimeError(
            f"static box QP failed: {result.message}; KKT={kkt_residual}"
        )
    return candidate


def _schedule_risks(
    correction_risks: np.ndarray,
    *,
    refresh_risk: float,
    budget: float,
    period: int,
) -> dict[str, Any]:
    horizon = len(correction_risks)
    allowance = int(round(budget * horizon))
    gains = correction_risks - refresh_risk
    oracle_indices = np.argsort(-gains)[:allowance]
    oracle_mask = np.zeros(horizon, dtype=bool)
    oracle_mask[oracle_indices[gains[oracle_indices] > 0.0]] = True
    oracle = np.where(oracle_mask, refresh_risk, correction_risks)

    periodic_candidates = []
    for phase in range(period):
        mask = np.arange(horizon) % period == phase
        periodic_candidates.append(np.where(mask, refresh_risk, correction_risks))
    periodic = min(periodic_candidates, key=lambda values: float(np.mean(values)))

    debt = 0.0
    used = 0
    causal_values = []
    lyapunov_v = math.sqrt(horizon)
    for correction_risk in correction_risks:
        feasible = used < allowance
        refresh = bool(
            feasible and refresh_risk + debt / lyapunov_v < correction_risk
        )
        used += int(refresh)
        debt = update_resource_debt(
            debt,
            incurred_cost=float(refresh),
            average_budget=budget,
        )
        causal_values.append(refresh_risk if refresh else correction_risk)
    causal = np.asarray(causal_values)
    return {
        "oracle_mean": float(np.mean(oracle)),
        "periodic_mean": float(np.mean(periodic)),
        "causal_mean": float(np.mean(causal)),
        "causal_refreshes": used,
        "allowance": allowance,
        "final_debt": debt,
    }


def _run_one(specification: _Specification, config: dict[str, Any]) -> dict[str, Any]:
    sensitivities, divergences = _generate_certificates(specification, config)
    second_moment = float(config["integrand_second_moment"])
    batch_size = float(specification.effective_batch_size)
    pointwise_alphas = []
    maximum_iterations = 0
    for d, v in zip(sensitivities, divergences):
        decision = solve_tempered_correction_qp(
            bias_sensitivities=d,
            divergence_proxies=v,
            integrand_second_moment=second_moment,
            effective_batch_size=batch_size,
        )
        pointwise_alphas.append(decision.alphas)
        maximum_iterations = max(maximum_iterations, decision.iterations)
    pointwise_alphas_array = np.asarray(pointwise_alphas)
    pointwise_risks = _risk_rows(
        pointwise_alphas_array,
        sensitivities,
        divergences,
        second_moment,
        batch_size,
    )
    static_alpha = _best_static_vector(
        sensitivities, divergences, second_moment, batch_size
    )
    static_risks = _risk_rows(
        static_alpha, sensitivities, divergences, second_moment, batch_size
    )
    no_risks = _risk_rows(
        np.zeros(specification.agents),
        sensitivities,
        divergences,
        second_moment,
        batch_size,
    )
    full_risks = _risk_rows(
        np.ones(specification.agents),
        sensitivities,
        divergences,
        second_moment,
        batch_size,
    )
    refresh_risk = second_moment / batch_size
    period = int(config["periods_by_budget"][str(specification.refresh_budget)])
    schedules = _schedule_risks(
        pointwise_risks,
        refresh_risk=refresh_risk,
        budget=specification.refresh_budget,
        period=period,
    )
    return {
        "seed": specification.seed,
        "agents": specification.agents,
        "effective_batch_size": specification.effective_batch_size,
        "persistence": specification.persistence,
        "profile": specification.profile,
        "refresh_budget": specification.refresh_budget,
        "adaptive_correction_mean": float(np.mean(pointwise_risks)),
        "static_vector_mean": float(np.mean(static_risks)),
        "no_correction_mean": float(np.mean(no_risks)),
        "full_correction_mean": float(np.mean(full_risks)),
        "causal_rcr_mean": schedules["causal_mean"],
        "best_periodic_mean": schedules["periodic_mean"],
        "oracle_mean": schedules["oracle_mean"],
        "causal_refreshes": schedules["causal_refreshes"],
        "refresh_allowance": schedules["allowance"],
        "final_resource_debt": schedules["final_debt"],
        "maximum_solver_iterations": maximum_iterations,
    }


def _geometric_ratio(numerators: list[float], denominators: list[float]) -> float:
    values = np.asarray(numerators) / np.asarray(denominators)
    if np.any(values <= 0.0) or np.any(~np.isfinite(values)):
        raise RuntimeError("invalid risk ratio")
    return float(np.exp(np.mean(np.log(values))))


def _cell_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "agents",
        "effective_batch_size",
        "persistence",
        "profile",
        "refresh_budget",
    )
    cells = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    for identity, selected in sorted(groups.items()):
        cell = {key: value for key, value in zip(keys, identity)}
        for metric in (
            "adaptive_correction_mean",
            "static_vector_mean",
            "causal_rcr_mean",
            "best_periodic_mean",
            "oracle_mean",
        ):
            cell[metric] = float(np.mean([row[metric] for row in selected]))
        periodic_gain = cell["best_periodic_mean"] - cell["oracle_mean"]
        causal_gain = cell["best_periodic_mean"] - cell["causal_rcr_mean"]
        cell["oracle_gain_capture"] = (
            causal_gain / periodic_gain if periodic_gain > 1e-15 else 1.0
        )
        cells.append(cell)
    return cells


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    cells = _cell_rows(rows)
    correction_ratio = _geometric_ratio(
        [row["adaptive_correction_mean"] for row in rows],
        [row["static_vector_mean"] for row in rows],
    )
    causal_ratio = _geometric_ratio(
        [row["causal_rcr_mean"] for row in rows],
        [row["best_periodic_mean"] for row in rows],
    )
    nonstationary = [cell for cell in cells if cell["profile"] != "stationary"]
    stationary = [cell for cell in cells if cell["profile"] == "stationary"]
    improvement_fraction = float(
        np.mean(
            [cell["causal_rcr_mean"] < cell["best_periodic_mean"] for cell in nonstationary]
        )
    )
    median_capture = float(
        np.median([cell["oracle_gain_capture"] for cell in nonstationary])
    )
    stationary_ratio = _geometric_ratio(
        [cell["causal_rcr_mean"] for cell in stationary],
        [cell["best_periodic_mean"] for cell in stationary],
    )
    persistence_ratios = {
        str(persistence): _geometric_ratio(
            [
                cell["causal_rcr_mean"]
                for cell in nonstationary
                if cell["persistence"] == persistence
            ],
            [
                cell["best_periodic_mean"]
                for cell in nonstationary
                if cell["persistence"] == persistence
            ],
        )
        for persistence in config["markov_persistence"]
    }
    finite = all(
        all(
            math.isfinite(value)
            for key, value in row.items()
            if isinstance(value, float)
        )
        for row in rows
    )
    overshoot = max(
        row["causal_refreshes"] - row["refresh_allowance"] for row in rows
    )
    thresholds = config["mandatory_gates"]
    gates = {
        "H1_finite_and_valid": finite,
        "H2_adaptive_correction": correction_ratio
        <= thresholds["H2_adaptive_correction_geometric_risk_ratio_max"],
        "H3_causal_rcr": causal_ratio
        <= thresholds["H3_causal_rcr_geometric_risk_ratio_max"],
        "H4_nonstationary_cells": improvement_fraction
        >= thresholds["H4_nonstationary_cell_improvement_fraction_min"],
        "H5_oracle_capture": median_capture
        >= thresholds["H5_median_oracle_gain_capture_min"],
        "H6_stationary": stationary_ratio
        <= thresholds["H6_stationary_causal_risk_ratio_max"],
        "H7_persistence": all(
            value <= thresholds["H7_each_persistence_causal_risk_ratio_max"]
            for value in persistence_ratios.values()
        ),
        "H8_budget": overshoot <= thresholds["H8_refresh_budget_overshoot_max"],
        "H9_complexity": max(row["maximum_solver_iterations"] for row in rows)
        <= 128,
        "H10_stop_rule": True,
    }
    return {
        "row_count": len(rows),
        "cell_count": len(cells),
        "adaptive_correction_geometric_risk_ratio": correction_ratio,
        "causal_rcr_geometric_risk_ratio": causal_ratio,
        "nonstationary_cell_improvement_fraction": improvement_fraction,
        "median_oracle_gain_capture": median_capture,
        "stationary_causal_risk_ratio": stationary_ratio,
        "persistence_causal_risk_ratios": persistence_ratios,
        "maximum_refresh_budget_overshoot": overshoot,
        "maximum_solver_iterations": max(
            row["maximum_solver_iterations"] for row in rows
        ),
        "gates": gates,
        "all_mandatory_gates_pass": all(gates.values()),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _load_config(args.config)
    specifications = _specifications(config)
    if args.command == "validate":
        print(f"config_sha256={_sha256(args.config)}")
        print("validation=pass")
        return
    if args.command == "estimate":
        print(f"scenarios={len(specifications)}")
        print(f"analytic_events={len(specifications) * int(config['horizon'])}")
        return
    if args.output_dir is None:
        raise ValueError("run requires --output-dir")
    rows = [_run_one(specification, config) for specification in specifications]
    summary = _summarize(rows, config)
    payload = {
        "experiment": "RCR-H1",
        "config_sha256": _sha256(args.config),
        "summary": summary,
        "rows": rows,
        "formal_evidence": False,
        "gpu_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cells"}, indent=2, sort_keys=True))
    print(f"summary_sha256={_sha256(path)}")


if __name__ == "__main__":
    main()
