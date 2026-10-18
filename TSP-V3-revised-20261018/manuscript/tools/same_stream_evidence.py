"""Outcome-known exact Gaussian risk transfer on immutable EXP-016B evidence.

No random draws or experiment entry points. The first formal seed supplies ONLY
public design metadata; all saved empirical means retain all registered seeds.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OLD = ROOT / "experiments/dependence_delay_linear"
sys.path.insert(0, str(OLD))
import run_exp016b_pilot as protocol
from run_adaptation_cost_pilot import Action, oracle_action, scheduled_updates
from analyze_exp016b_pilot import analytic_cell_tags
from t018_static_scan import DELTA


def forbidden(*args, **kwargs):
    raise RuntimeError("New trajectories and old experiment entry points forbidden")


for name in ("generate_potential_observations", "simulate_policy", "_ar1_path"):
    setattr(protocol, name, forbidden)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


@lru_cache(None)
def program(layer, scheduled, delay):
    n = max(0, scheduled - delay)
    if not n:
        return 1.0, np.zeros(0)
    if layer.startswith("A_"):
        return 1.0 / (n + 1), np.full(n, 1.0 / (n + 1))
    mu = protocol.AFFINE_CONTRACTION
    eta = 0.1 / (mu * (delay + 1))
    impulse = np.empty(n)
    impulse[0] = 1.0
    for t in range(1, n):
        impulse[t] = impulse[t - 1] - (eta * mu * impulse[t - delay - 1] if t > delay else 0.0)
    weights = eta / n * np.cumsum(impulse)[::-1]
    return float(1.0 - mu * weights.sum()), weights


@lru_cache(None)
def risk(layer, scheduled, delay, q, b, lam, theta):
    d, weights = program(layer, scheduled, delay)
    r = lam ** b
    cross = 0.0
    common = 0.0
    carry = 0.0
    power = r
    for weight in weights:
        common += weight * weight + 2 * weight * cross
        cross = r * (cross + weight)
        carry += weight * power
        power *= r
    return (float(d * d + theta * common + np.dot(weights, weights) / q),
            float(d), float(carry))


def posterior_variance(theta, lam, q, b, n):
    variance = theta
    for t in range(n):
        variance /= 1.0 + q * variance
        if t + 1 < n:
            variance = lam ** (2 * b) * variance + (1 - lam ** (2 * b)) * theta
    return variance


def operator_checks():
    count = 0
    for delay in (0, 4, 12, 24):
        for scheduled in (0, delay, delay + 1, delay + 7, delay + 59):
            observations = np.linspace(-0.7, 1.1, scheduled)
            d, w = program(protocol.LAYERS[1], scheduled, delay)
            squared, _, applied = protocol.execute_affine_td(observations, delay)
            predicted = (d + float(w @ observations[:len(w)])) ** 2
            assert applied == len(w) and math.isclose(squared, predicted, abs_tol=2e-14)
            for layer in protocol.LAYERS:
                value, d, _ = risk(layer, scheduled, delay, 4, 2, 0.7, 0.5)
                _, w = program(layer, scheduled, delay)
                toeplitz = 0.7 ** (2 * np.abs(np.arange(len(w))[:, None] - np.arange(len(w))))
                dense = d * d + 0.5 * float(w @ toeplitz @ w) + float(w @ w) / 4
                assert math.isclose(value, dense, abs_tol=2e-14)
                count += 1
    return count


def run():
    started = time.process_time()
    out = HERE / "internal/same_stream_evidence_20261018"
    if out.exists():
        raise FileExistsError(out)
    registry = json.loads((ROOT / "docs/exp016b_formal_registry.json").read_text())
    audit = json.loads((ROOT / "docs/exp016b_formal_reproduction_audit.json").read_text())
    manifest = protocol.load_frozen_bundle()["manifest"]
    source_hashes = {}
    for group in ("frozen_input_sha256", "implementation_sha256"):
        for rel, expected in registry[group].items():
            actual = sha(ROOT / rel)
            assert actual == expected, rel
            source_hashes[rel] = actual
    root = OLD / "results/exp016b_formal_20260801"
    for item in audit["artifacts"]:
        actual = sha(root / item["file"])
        assert actual == item["original_sha256"], item["file"]
        source_hashes[str((root / item["file"]).relative_to(ROOT))] = actual
    core = json.loads((root / "analysis/core_results.json").read_text())
    rows_per_seed = int(core["rows"]) // registry["formal_seed_count"]
    columns = ["seed", "layer", "scenario_id", "budget_point", "scale", "regime", "policy",
               "theta_true_evaluation_only", "lambda_public", "maximum_agents", "delay", "overhead",
               "B_id", "B_value", "B_value_status", "message_budget", "environment_budget",
               "probe_used", "probe_q", "probe_b", "probe_n", "identified_regime", "selected_q",
               "selected_b", "scheduled_updates", "usable_updates_after_delay", "messages_used", "environment_used"]
    meta = pd.read_csv(root / "metrics.csv", nrows=rows_per_seed, usecols=columns)
    assert meta.seed.nunique() == 1 and len(meta) == rows_per_seed
    empirical = pd.read_csv(root / "analysis/cell_summary.csv")
    keys = ["layer", "scenario_id", "budget_point", "scale", "regime", "policy"]
    assert len(empirical) == len(meta) and not meta.duplicated(keys).any()
    assert set(meta.scenario_id) == {item["scenario_id"] for item in manifest["scenarios"]}
    lookup = protocol.scenario_lookup()
    checks = operator_checks()
    rows = []
    for record in meta.to_dict("records"):
        scenario = lookup[record["scenario_id"]]
        layer, delay = record["layer"], int(record["delay"])
        theta, lam = record["theta_true_evaluation_only"], record["lambda_public"]
        bm, be, overhead = int(record["message_budget"]), int(record["environment_budget"]), int(record["overhead"])
        pq, pb, pn = (int(record[name]) for name in ("probe_q", "probe_b", "probe_n"))
        base = {name: record[name] for name in keys + ["B_id", "B_value_status"]}
        base["B_value"] = None if pd.isna(record["B_value"]) else int(record["B_value"])
        base["probe_used"] = bool(record["probe_used"])
        probe_m, probe_e = pn * (overhead + pq), pn * pb
        assert probe_m <= bm and probe_e <= be
        if record["probe_used"]:
            high = oracle_action(scenario.theta_high, lam, bm - probe_m, be - probe_e, overhead, delay, scenario.maximum_agents)[0]
            low = Action(scenario.maximum_agents, 1)
            programs = {}
            for regime, action in (("low", low), ("high", high)):
                scheduled = scheduled_updates(action, bm - probe_m, be - probe_e, overhead)
                programs[regime] = risk(layer, scheduled, delay, action.q, action.b, lam, theta)
            selected = low if record["identified_regime"] == "low" else high
            correct = programs[record["regime"]]
            wrong = programs["high" if record["regime"] == "low" else "low"]
            s2 = max(0.0, theta - posterior_variance(theta, lam, pq, pb, pn))
            delta_j = wrong[0] - correct[0]
            gamma = math.sqrt(4 * (wrong[1] * wrong[2] - correct[1] * correct[2]) ** 2 * s2
                              + 2 * (wrong[2] ** 2 - correct[2] ** 2) ** 2 * s2 ** 2)
            correction = math.sqrt(DELTA) * gamma
            lower = max(0.0, correct[0] - DELTA * max(0.0, -delta_j) - correction)
            upper = correct[0] + DELTA * max(0.0, delta_j) + correction
            base.update(correct_risk=correct[0], wrong_risk=wrong[0], carryover_correction=correction,
                        posterior_variance=theta - s2, risk_lower=lower, risk_upper=upper,
                        program_signature=f"probe:{pq}:{pb}:{pn}:{low.q}:{low.b}:{high.q}:{high.b}")
        else:
            selected = Action(int(record["selected_q"]), int(record["selected_b"]))
            value, _, _ = risk(layer, int(record["scheduled_updates"]), delay, selected.q, selected.b, lam, theta)
            base.update(correct_risk=value, wrong_risk=value, carryover_correction=0.0,
                        posterior_variance=theta, risk_lower=value, risk_upper=value,
                        program_signature=f"fixed:{selected.q}:{selected.b}")
        assert selected.q == record["selected_q"] and selected.b == record["selected_b"]
        scheduled = scheduled_updates(selected, bm - probe_m, be - probe_e, overhead)
        assert scheduled == record["scheduled_updates"]
        assert max(0, scheduled - delay) == record["usable_updates_after_delay"]
        assert probe_m + scheduled * (overhead + selected.q) == record["messages_used"]
        assert probe_e + scheduled * selected.b == record["environment_used"]
        rows.append(base)
    bounds = pd.DataFrame(rows).merge(empirical[keys + ["mean_terminal_risk", "mean_messages", "mean_environment"]], on=keys, validate="one_to_one")
    tags = analytic_cell_tags(manifest)
    contrasts = []
    for cell, group in bounds.groupby(keys[:-1], sort=True):
        policies = group.set_index("policy")
        info, aware = policies.loc["information_only"], policies.loc["learning_aware"]
        same = info.program_signature == aware.program_signature
        low = 0.0 if same else info.risk_lower - aware.risk_upper
        high = 0.0 if same else info.risk_upper - aware.risk_lower
        contrasts.append(dict(zip(keys[:-1], cell), difference_lower=low, difference_upper=high,
                              observed_mean_difference=info.mean_terminal_risk - aware.mean_terminal_risk,
                              identical_program=bool(same), B_id=int(info.B_id),
                              B_value=None if pd.isna(info.B_value) else int(info.B_value),
                              B_value_status=info.B_value_status))
    contrasts = pd.DataFrame(contrasts).merge(tags, on=["scenario_id", "budget_point", "scale", "regime"], validate="many_to_one", suffixes=("", "_tag"))
    assert len(bounds) == rows_per_seed and len(contrasts) * len(protocol.POLICIES) == len(bounds)
    summaries = []
    for layer, group in contrasts.groupby("layer"):
        for name, mask in {
            "all": np.ones(len(group), dtype=bool),
            "below_identification": group.scale < group.B_id,
            "intermediate": group.in_Z,
            "registered_practical_intermediate": group.practical_Z & group.finite_threshold_gate_eligible,
            "above_value": group.B_value.notna() & (group.scale >= group.B_value),
            "search_censored": group.B_value_status.ne("finite"),
        }.items():
            part = group[mask]
            summaries.append(dict(layer=layer, stratum=name, cells=len(part), scenarios=part.scenario_id.nunique(),
                                  positive_intervals=int((part.difference_lower > 0).sum()),
                                  adverse_intervals=int((part.difference_upper < 0).sum()),
                                  identical_programs=int(part.identical_program.sum()),
                                  mean_lower=float(part.difference_lower.mean()) if len(part) else None,
                                  mean_upper=float(part.difference_upper.mean()) if len(part) else None,
                                  observed_mean_difference=float(part.observed_mean_difference.mean()) if len(part) else None))
    assert np.isfinite(bounds.select_dtypes(include="number").drop(columns=["B_value"]).to_numpy()).all()
    out.mkdir()
    bounds.to_csv(out / "risk_bounds.csv", index=False, float_format="%.17g")
    contrasts.to_csv(out / "contrasts.csv", index=False, float_format="%.17g")
    dump(out / "summary.json", {"outcome_known_reanalysis": True, "new_trajectories": 0,
                                "directional_error_bound": DELTA, "strata": summaries,
                                "max_carryover_correction": float(bounds.carryover_correction.max()),
                                "all_scenario_count": bounds.scenario_id.nunique(),
                                "rows": len(bounds), "empirical_seeds_per_cell": registry["formal_seed_count"]})
    dump(out / "integrity.json", {"pass": True, "operator_checks": checks,
                                  "public_protocol_and_ledgers_checked_rows": len(bounds),
                                  "all_eight_policies": sorted(bounds.policy.unique()),
                                  "source_and_evidence_sha256": source_hashes,
                                  "analysis_script_sha256": sha(Path(__file__)),
                                  "analysis_cpu_seconds": time.process_time() - started,
                                  "statistical_scope": "Population intervals, not simultaneous empirical sample-mean coverage; no new gate or confirmation"})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    run()
