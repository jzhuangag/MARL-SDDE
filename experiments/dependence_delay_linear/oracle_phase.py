"""Finite-budget oracle participation surface for EXP-006A."""

from dataclasses import replace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from budget_participation import AGENT_COUNTS
from linear_model import make_agent_delays
from online_participation import (
    FiniteBudgetProxyCache,
    OnlineConfig,
    true_aggregate_lrv,
)


DEPENDENCE_PATHS: Dict[str, Tuple[float, float]] = {
    "global": (1.0, 0.0),
    "clustered": (0.0, 1.0),
    "balanced": (0.5, 0.5),
}
STRENGTHS: Tuple[float, ...] = tuple(np.linspace(0.0, 0.8, 9))
BUDGETS: Tuple[int, ...] = (250, 500, 1000, 2000, 4000, 8000)
ERROR_AMPLITUDES: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)
MAX_DELAYS: Tuple[int, ...] = (0, 4, 16, 32)
OVERHEADS: Tuple[int, ...] = (0, 4, 16)


def oracle_action(
    rho_global: float,
    rho_cluster: float,
    budget: int,
    error_amplitude: float,
    max_delay: int,
    config: OnlineConfig,
    cache: FiniteBudgetProxyCache,
) -> Dict[str, float]:
    """Return the best action and the best risk at a different q."""

    delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=max_delay,
        exponent=config.delay_exponent,
    )
    maximum_delay = int(np.max(delays))
    history = np.full(maximum_delay + 1, error_amplitude, dtype=float)
    per_q: List[Dict[str, float]] = []
    for q in AGENT_COUNTS:
        horizon = budget // (config.update_overhead + q)
        if horizon < 1:
            continue
        selected_delays = delays[:q]
        local_history = history[: int(np.max(selected_delays)) + 1]
        lrv = true_aggregate_lrv(
            np.arange(q), rho_global, rho_cluster, config
        )
        best_eta_row = None
        best_eta_key = None
        for eta in config.eta_grid:
            proxy = cache.get(selected_delays, float(eta), horizon)
            if not bool(proxy["stable"]):
                continue
            final_mean = np.asarray(proxy["transition_power"]).dot(
                local_history
            )
            risk = float(
                final_mean[0] ** 2
                + max(lrv, 1e-10) * float(proxy["noise_gain"])
            )
            key = (risk, float(eta))
            if best_eta_key is None or key < best_eta_key:
                best_eta_key = key
                best_eta_row = {
                    "q": float(q),
                    "eta": float(eta),
                    "risk": risk,
                    "lrv": float(lrv),
                    "horizon": float(horizon),
                }
        if best_eta_row is not None:
            per_q.append(best_eta_row)
    if len(per_q) < 2:
        raise RuntimeError("fewer than two stable participation actions")
    ordered = sorted(
        per_q,
        key=lambda row: (row["risk"], -row["q"], row["eta"]),
    )
    best, runner_up = ordered[:2]
    margin = (runner_up["risk"] - best["risk"]) / max(
        best["risk"], 1e-15
    )
    return {
        "selected_q": best["q"],
        "selected_eta": best["eta"],
        "best_risk": best["risk"],
        "runner_up_q": runner_up["q"],
        "runner_up_risk": runner_up["risk"],
        "relative_margin": float(max(margin, 0.0)),
        "lrv_at_best_q": best["lrv"],
        "horizon_at_best_q": best["horizon"],
    }


def build_surface(
    paths: Iterable[str] = DEPENDENCE_PATHS,
    strengths: Sequence[float] = STRENGTHS,
    budgets: Sequence[int] = BUDGETS,
    errors: Sequence[float] = ERROR_AMPLITUDES,
    max_delays: Sequence[int] = MAX_DELAYS,
    overheads: Sequence[int] = OVERHEADS,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base_config = OnlineConfig()
    for overhead in overheads:
        config = replace(base_config, update_overhead=int(overhead))
        cache = FiniteBudgetProxyCache(config)
        for max_delay in max_delays:
            for budget in budgets:
                for error in errors:
                    for path in paths:
                        global_weight, cluster_weight = DEPENDENCE_PATHS[path]
                        for strength in strengths:
                            rho_global = float(strength * global_weight)
                            rho_cluster = float(
                                strength * cluster_weight
                            )
                            action = oracle_action(
                                rho_global=rho_global,
                                rho_cluster=rho_cluster,
                                budget=int(budget),
                                error_amplitude=float(error),
                                max_delay=int(max_delay),
                                config=config,
                                cache=cache,
                            )
                            rows.append(
                                {
                                    "path": path,
                                    "strength": float(strength),
                                    "rho_global": rho_global,
                                    "rho_cluster": rho_cluster,
                                    "budget": int(budget),
                                    "error_amplitude": float(error),
                                    "max_delay": int(max_delay),
                                    "overhead": int(overhead),
                                    **action,
                                }
                            )
    return pd.DataFrame(rows)


def summarize_tracks(surface: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "path",
        "budget",
        "error_amplitude",
        "max_delay",
        "overhead",
    ]
    rows: List[Dict[str, object]] = []
    for values, group in surface.groupby(keys, sort=True):
        ordered = group.sort_values("strength")
        q_values = ordered["selected_q"].to_numpy(dtype=float)
        nonincreasing = bool(np.all(np.diff(q_values) <= 0))
        responsive = bool(
            nonincreasing
            and q_values[0] >= 16
            and q_values[-1] <= q_values[0] / 4
        )
        rows.append(
            {
                **dict(zip(keys, values)),
                "q_at_zero": float(q_values[0]),
                "q_at_high": float(q_values[-1]),
                "nonincreasing": nonincreasing,
                "responsive": responsive,
                "high_strength_margin": float(
                    ordered.iloc[-1]["relative_margin"]
                ),
                "well_separated": bool(
                    responsive
                    and ordered.iloc[-1]["relative_margin"] >= 0.05
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_delay(surface: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "path",
        "strength",
        "budget",
        "error_amplitude",
        "overhead",
    ]
    rows: List[Dict[str, object]] = []
    for values, group in surface.groupby(keys, sort=True):
        rows.append(
            {
                **dict(zip(keys, values)),
                "distinct_q_across_delays": int(
                    group["selected_q"].nunique()
                ),
                "delay_sensitive": bool(
                    group["selected_q"].nunique() >= 2
                ),
            }
        )
    return pd.DataFrame(rows)


def find_actionable_rectangles(surface: pd.DataFrame) -> pd.DataFrame:
    lookup = surface.set_index(
        [
            "path",
            "strength",
            "budget",
            "error_amplitude",
            "max_delay",
            "overhead",
        ]
    )
    rows: List[Dict[str, object]] = []
    for path in ("global", "clustered"):
        for budget_low, budget_high in zip(BUDGETS[:-1], BUDGETS[1:]):
            for error_low, error_high in zip(
                ERROR_AMPLITUDES[:-1], ERROR_AMPLITUDES[1:]
            ):
                corners = []
                for budget in (budget_low, budget_high):
                    for error in (error_low, error_high):
                        for max_delay in (4, 16):
                            low = lookup.loc[
                                (
                                    path,
                                    0.0,
                                    budget,
                                    error,
                                    max_delay,
                                    4,
                                )
                            ]
                            high = lookup.loc[
                                (
                                    path,
                                    0.8,
                                    budget,
                                    error,
                                    max_delay,
                                    4,
                                )
                            ]
                            corners.append(
                                bool(
                                    low["selected_q"] >= 16
                                    and high["selected_q"] <= 8
                                    and high["relative_margin"] >= 0.02
                                )
                            )
                rows.append(
                    {
                        "path": path,
                        "budget_low": budget_low,
                        "budget_high": budget_high,
                        "error_low": error_low,
                        "error_high": error_high,
                        "pass": bool(all(corners)),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_gates(
    surface: pd.DataFrame,
    tracks: pd.DataFrame,
    delay_summary: pd.DataFrame,
    rectangles: pd.DataFrame,
    expected_rows: int,
) -> Dict[str, object]:
    q_frequency = (
        surface["selected_q"].value_counts(normalize=True).sort_index()
    )
    occupied = int((q_frequency >= 0.01).sum())
    largest_share = float(q_frequency.max())
    responsive_fraction = float(tracks["responsive"].mean())
    path_fractions = (
        tracks.groupby("path")["responsive"].mean().to_dict()
    )
    responsive_paths = sum(
        float(value) >= 0.10 for value in path_fractions.values()
    )
    responsive = tracks[tracks["responsive"]]
    separated_fraction = (
        float(responsive["well_separated"].mean())
        if len(responsive)
        else 0.0
    )
    delay_fraction = float(delay_summary["delay_sensitive"].mean())
    rectangle_count = int(rectangles["pass"].sum())
    finite_columns = [
        "selected_q",
        "selected_eta",
        "best_risk",
        "runner_up_risk",
        "relative_margin",
    ]
    numerical = bool(
        len(surface) == expected_rows
        and np.isfinite(surface[finite_columns].to_numpy()).all()
        and (surface["relative_margin"] >= 0).all()
    )
    gates: Dict[str, object] = {
        "non_degenerate_surface": {
            "pass": bool(occupied >= 3 and largest_share <= 0.85),
            "occupied_q_values": occupied,
            "largest_q_share": largest_share,
        },
        "correlation_responsiveness": {
            "pass": bool(
                responsive_fraction >= 0.15 and responsive_paths >= 2
            ),
            "responsive_fraction": responsive_fraction,
            "path_fractions": {
                key: float(value)
                for key, value in path_fractions.items()
            },
            "paths_at_least_ten_percent": responsive_paths,
        },
        "decision_margin": {
            "pass": bool(separated_fraction >= 0.50),
            "well_separated_fraction": separated_fraction,
            "responsive_track_count": int(len(responsive)),
        },
        "delay_relevance": {
            "pass": bool(delay_fraction >= 0.10),
            "delay_sensitive_fraction": delay_fraction,
        },
        "actionable_region": {
            "pass": bool(rectangle_count >= 1),
            "rectangle_count": rectangle_count,
        },
        "numerical_validity": {
            "pass": numerical,
            "observed_rows": int(len(surface)),
            "expected_rows": int(expected_rows),
        },
    }
    gates["overall"] = {
        "pass": bool(all(value["pass"] for value in gates.values())),
        "criterion": "all six oracle-viability gates pass",
    }
    return gates
