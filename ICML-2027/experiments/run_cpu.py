"""Preregistered CPU learning-value qualification and deterministic analysis."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from learning_value import (action_table, calibration_cost, choose, covers,
                            fit_calibration, forecast, observe_calibration,
                            risk_coefficients, selection_bound, simulate_errors,
                            true_risk)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE/"cpu_preregistration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def contexts(config: dict) -> list[dict]:
    fields = ("target", "rho", "persistence", "delay", "overhead", "actor_budgets")
    rows = []
    for index, values in enumerate(itertools.product(*(config[f] for f in fields))):
        row = dict(zip(fields, values))
        row["cell"] = index
        row["actor_budget"] = row.pop("actor_budgets")
        rows.append(row)
    return rows


def tables(config: dict, cells: list[dict]) -> dict:
    cached = {}
    for cell in cells:
        key = tuple(cell[k] for k in ("persistence", "delay", "overhead", "actor_budget"))
        if key in cached:
            continue
        pm, pa, pt = calibration_cost(config["calibration_workers"], config["calibration_innovations"], cell["overhead"])
        public = dict(qs=config["q_catalogue"], spacings=config["spacing_catalogue"],
                      steps=config["step_catalogue"], persistence=cell["persistence"],
                      delay=cell["delay"], message_budget=config["message_budget"],
                      actor_budget=cell["actor_budget"], overhead=cell["overhead"])
        cached[key] = (action_table(**public, probe_messages=pm, probe_actors=pa, probe_ticks=pt),
                       action_table(**public), (pm, pa, pt))
    return cached


def run(phase: str, out: Path, lock_path: Path | None) -> None:
    started = time.perf_counter()
    config = json.loads(CONFIG.read_text())
    if phase == "development":
        seeds = config["development_seeds"]
    else:
        if lock_path is None:
            raise ValueError("confirmation requires an independent committed lock")
        lock = json.loads(lock_path.read_text())
        if lock["config_sha256"] != sha256(CONFIG) or not lock["development_all_pass"]:
            raise ValueError("confirmation lock does not authorize this configuration")
        if lock["implementation_sha256"] != sha256(HERE/"learning_value.py"):
            raise ValueError("confirmation implementation changed")
        seeds = lock["seeds"]
        if set(seeds).intersection(config["development_seeds"]):
            raise ValueError("development/confirmation seed overlap")
    if out.exists():
        raise FileExistsError("refusing to overwrite an experimental directory")
    out.mkdir(parents=True)
    cells = contexts(config)
    cached = tables(config, cells)
    print(f"{phase}: {len(cells)} contexts x {len(seeds)} seeds", flush=True)
    endpoints, certificates, curves, monte_carlo = [], [], [], []
    for cell in cells:
        key = tuple(cell[k] for k in ("persistence", "delay", "overhead", "actor_budget"))
        paid, unpaid, charges = cached[key]
        truth = dict(target=cell["target"], rho=cell["rho"])
        paid_risks = [true_risk(a, **truth) for a in paid]
        paid_best = paid[int(np.argmin(paid_risks))]
        per_q = {q: min((a for a in paid if a.q == q), key=lambda a: true_risk(a, **truth)) for q in config["q_catalogue"]}
        competing = min(true_risk(a, **truth) for q, a in per_q.items() if q != paid_best.q)
        interior = paid_best.q in (2, 4) and competing/true_risk(paid_best, **truth) >= 1.03
        fixed = {f"fixed_q{q}": min((a for a in unpaid if a.q == q), key=lambda a: true_risk(a, **truth)) for q in config["q_catalogue"]}
        curve_cell = (cell["target"] == 1 and cell["persistence"] == .9 and cell["delay"] == 2
                      and cell["overhead"] == 4 and cell["actor_budget"] == 32768
                      and cell["rho"] in (0, .5, 1))
        for seed_index, seed in enumerate(seeds):
            rng = np.random.default_rng(np.random.SeedSequence([seed, cell["cell"], 0]))
            observations = observe_calibration(rng, **truth, persistence=cell["persistence"],
                                               workers=config["calibration_workers"],
                                               observations=config["calibration_innovations"])
            certificate = fit_calibration(observations, persistence=cell["persistence"], delta=config["calibration_delta"])
            covered = covers(certificate, **truth)
            decisions = {name: choose(paid, certificate, name) for name in ("finite_horizon", "correlation_only", "myopic")}
            selected = decisions["finite_horizon"]
            interval_valid = True
            for action, actual in zip(paid, paid_risks):
                _, low, high = forecast(action, certificate)
                interval_valid &= low-1e-10 <= actual <= high+1e-10
            regret = true_risk(selected, **truth)-true_risk(paid_best, **truth)
            bound = selection_bound(paid, certificate, selected)
            certificates.append({**cell, "seed": seed, "joint_covered": covered,
                                 "all_action_intervals_valid": interval_valid,
                                 "regret": regret, "regret_upper": bound,
                                 "regret_bound_valid": regret <= bound+1e-10,
                                 "interior_qualified": interior, "oracle_q": paid_best.q,
                                 "selected_q": selected.q, "target_estimate": certificate.target,
                                 "initial_risk_interval_width": certificate.initial_risk()[2]-certificate.initial_risk()[1]})
            all_methods = {**decisions, **fixed, "paid_oracle": paid_best}
            for method, action in all_methods.items():
                uses_probe = method not in fixed
                predicted, low, high = forecast(action, certificate)
                endpoints.append({**cell, "seed": seed, "method": method,
                                  "q": action.q, "spacing": action.spacing, "step": action.step,
                                  "updates": action.updates, "risk": true_risk(action, **truth),
                                  "fitted_risk": predicted, "risk_lower": low, "risk_upper": high,
                                  "messages": action.messages, "actor_transitions": action.actor_transitions,
                                  "parallel_ticks": action.parallel_ticks,
                                  "probe_messages": charges[0] if uses_probe else 0,
                                  "probe_actor_transitions": charges[1] if uses_probe else 0,
                                  "complete_q_catalogue": sorted({a.q for a in paid}) == config["q_catalogue"]})
                if curve_cell:
                    for fraction in np.linspace(0, 1, 21):
                        n = int(round(fraction*action.updates))
                        bias, noise, _ = risk_coefficients(action.step, cell["persistence"]**action.spacing, cell["delay"], n)
                        prefix_m = (charges[0] if uses_probe else 0)+n*(cell["overhead"]+action.q)
                        prefix_e = (charges[1] if uses_probe else 0)+n*action.q*action.spacing
                        curves.append({**cell, "seed": seed, "method": method, "updates": n,
                                       "message_fraction": prefix_m/config["message_budget"],
                                       "actor_fraction": prefix_e/cell["actor_budget"],
                                       "budget_fraction": max(prefix_m/config["message_budget"], prefix_e/cell["actor_budget"]),
                                       "risk": bias*cell["target"]**2+noise*(cell["rho"]+(1-cell["rho"])/action.q)})
                    if seed_index == 0 and method in ("finite_horizon", "fixed_q1", "fixed_q8"):
                        checkpoints = np.unique(np.rint(np.linspace(0, action.updates, 21)).astype(int))
                        samples = simulate_errors(np.random.default_rng(np.random.SeedSequence([seed, cell["cell"], 100+action.q])),
                                                  action, **truth, persistence=cell["persistence"], delay=cell["delay"],
                                                  replicates=4096, checkpoints=checkpoints)
                        for n, errors in zip(checkpoints, samples):
                            bias, noise, _ = risk_coefficients(action.step, cell["persistence"]**action.spacing, cell["delay"], int(n))
                            expected = bias*cell["target"]**2+noise*(cell["rho"]+(1-cell["rho"])/action.q)
                            empirical = float(np.square(errors).mean())
                            se = float(np.square(errors).std(ddof=1)/np.sqrt(len(errors)))
                            monte_carlo.append({"cell": cell["cell"], "method": method, "updates": int(n),
                                                "replicates": len(errors), "exact_risk": expected, "empirical_risk": empirical,
                                                "standard_error": se, "within_4_5_standard_errors": abs(empirical-expected) <= 4.5*se+1e-12})
        if cell["cell"] % 60 == 0:
            print(f"completed context {cell['cell']+1}/{len(cells)}", flush=True)
    data = pd.DataFrame(endpoints)
    cert = pd.DataFrame(certificates)
    data.to_csv(out/"endpoints.csv", index=False, float_format="%.17g", lineterminator="\n")
    cert.to_csv(out/"certificates.csv", index=False, float_format="%.17g", lineterminator="\n")
    pd.DataFrame(curves).to_csv(out/"curves.csv", index=False, float_format="%.17g", lineterminator="\n")
    mc = pd.DataFrame(monte_carlo)
    mc.to_csv(out/"monte_carlo.csv", index=False, float_format="%.17g", lineterminator="\n")
    paired = data.pivot(index=["seed", "cell"], columns="method", values="risk")
    aggregate = {method: float(np.exp(np.log(data.loc[data.method.eq(method), "risk"]).mean())) for method in config["methods"]}
    best_fixed = min(fixed, key=lambda name: aggregate[name])
    comparisons = []
    for comparator in (best_fixed, "paid_oracle", "correlation_only", "myopic"):
        logs = np.log(paired["finite_horizon"]/paired[comparator]).groupby("seed").mean()
        se = float(logs.std(ddof=1)/np.sqrt(len(logs)))
        radius = float(t.ppf(.975, len(logs)-1)*se)
        comparisons.append({"comparator": comparator, "ratio": float(np.exp(logs.mean())),
                            "ci95_low": float(np.exp(logs.mean()-radius)), "ci95_high": float(np.exp(logs.mean()+radius)),
                            "seed_pairs": len(logs)})
    comparison = {row["comparator"]: row for row in comparisons}
    qualified = cert[cert.interior_qualified]
    covered_rows = cert[cert.joint_covered]
    cfg = config["mandatory_gates"]
    gates = {
        "all_payloads_finite_and_exact_dual_accounting": bool(np.isfinite(data.select_dtypes(include=[np.number])).all().all()
             and data.messages.le(config["message_budget"]).all() and data.actor_transitions.le(data.actor_budget).all()
             and len(data) == len(cells)*len(seeds)*len(config["methods"])),
        "complete_q_catalogue_in_every_cell": bool(data.complete_q_catalogue.all()),
        "joint_parameter_coverage_fraction_min": float(cert.joint_covered.mean()) >= cfg["joint_parameter_coverage_fraction_min"],
        "risk_intervals_and_selection_bound_valid_on_coverage_event": bool(covered_rows.all_action_intervals_valid.all() and covered_rows.regret_bound_valid.all()),
        "geometric_risk_ratio_to_paid_oracle_max": comparison["paid_oracle"]["ratio"] <= cfg["geometric_risk_ratio_to_paid_oracle_max"],
        "ratio_to_strongest_unpaid_single_fixed_q_max": comparison[best_fixed]["ratio"] <= cfg["ratio_to_strongest_unpaid_single_fixed_q_max"],
        "interior_oracle_qualified_cells_min": int(qualified.cell.nunique()) >= cfg["interior_oracle_qualified_cells_min"],
        "interior_q_selection_fraction_min": bool(len(qualified) and qualified.selected_q.eq(qualified.oracle_q).mean() >= cfg["interior_q_selection_fraction_min"]),
    }
    summary = {"experiment_id": config["experiment_id"], "phase": phase, "cells": len(cells), "seeds": len(seeds),
               "rows": len(data), "scope": config["role"], "config_sha256": sha256(CONFIG),
               "implementation_sha256": sha256(HERE/"learning_value.py"), "runner_sha256": sha256(Path(__file__)),
               "joint_coverage_fraction": float(cert.joint_covered.mean()),
               "risk_interval_coverage_fraction": float(cert.all_action_intervals_valid.mean()),
               "geometric_risks": aggregate, "strongest_unpaid_single_fixed_q": best_fixed,
               "comparisons": comparisons, "interior_qualified_cells": int(qualified.cell.nunique()),
               "interior_selection_fraction": float(qualified.selected_q.eq(qualified.oracle_q).mean()) if len(qualified) else 0.,
               "median_selection_bound": float(cert.regret_upper.median()),
               "median_actual_selection_regret": float(cert.regret.median()),
               "monte_carlo_checkpoints": len(mc), "monte_carlo_all_checks_pass": bool(mc.within_4_5_standard_errors.all()),
               "gates_without_replay": gates, "all_gates_without_replay_pass": all(gates.values()),
               "confirmation_authorized": False}
    dump(out/"summary.json", summary)
    dump(out/"provenance.json", {"elapsed_seconds": time.perf_counter()-started, "python": platform.python_version(),
                                  "numpy": np.__version__, "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                  "command_phase": phase, "lock_sha256": sha256(lock_path) if lock_path else None})
    outputs = sorted(p for p in out.iterdir() if p.is_file() and p.name != "provenance.json")
    (out/"SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in outputs), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "confirmation"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confirmation-lock", type=Path)
    args = parser.parse_args()
    run(args.phase, args.out, args.confirmation_lock)
