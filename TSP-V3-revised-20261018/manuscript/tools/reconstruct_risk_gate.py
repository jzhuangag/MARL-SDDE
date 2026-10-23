"""Deterministic, outcome-known reconstruction; never generates trajectories."""
import hashlib
import json
import os
from pathlib import Path
import time
import argparse

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OUT = HERE / "internal/content_strengthening_20261019/replay"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(registered=False):
    global OUT
    if registered:
        OUT = OUT.parent / "replay_registered_safety"
    start = time.process_time()
    if OUT.exists():
        raise FileExistsError(OUT)
    source = HERE / "internal/same_stream_directional_evidence_20261018/risk_bounds.csv"
    design = OUT.parent / "REPLAY_DESIGN.md"
    implementation = ROOT / "experiments/dependence_delay_linear/run_exp016b_pilot.py"
    registry = json.loads((ROOT / "docs/exp016b_formal_registry.json").read_text())
    manifest_path = ROOT / "docs/exp016b_scenario_manifest.json"
    assert sha(manifest_path) == registry["frozen_input_sha256"]["docs/exp016b_scenario_manifest.json"]
    manifest = json.loads(manifest_path.read_text())
    safety = {s["scenario_id"]: s["epsilon_safe"] for s in manifest["scenarios"]}
    assert sha(implementation) == registry["implementation_sha256"][str(implementation.relative_to(ROOT)).replace("\\", "/")]
    rows = pd.read_csv(source)
    keys = ["layer", "scenario_id", "budget_point", "scale"]
    assert len(rows) == 14336 and rows.policy.nunique() == 8
    assert not rows.duplicated(keys + ["regime", "policy"]).any()
    tables, gates = [], []
    for cell, group in rows.groupby(keys, sort=True):
        info = group[group.policy == "information_only"].set_index("regime")
        base = group[group.policy == "always_all"].set_index("regime")
        assert set(info.index) == set(base.index) == {"high", "low"}
        assert info.probe_used.nunique() == 1
        assert info.program_signature.nunique() == 1
        feasible = bool(info.probe_used.iloc[0])
        epsilon = float(safety[cell[1]]) if registered else 0.0
        for variant, carry in (("risk_corrected_gate", True), ("no_carryover_gate", False)):
            upper = info.risk_upper - (0 if carry else info.carryover_correction)
            admit = feasible and upper["high"] < base.correct_risk["high"] and upper["low"] <= base.correct_risk["low"] + epsilon
            gates.append(dict(zip(keys, cell), policy=variant, candidate_available=feasible,
                              admit=bool(admit), epsilon_safe=epsilon, high_margin=float(base.correct_risk["high"]-upper["high"]),
                              low_slack=float(upper["low"]-base.correct_risk["low"])))
            for regime in ("low", "high"):
                chosen = info.loc[regime] if admit else base.loc[regime]
                record = chosen.to_dict()
                record.update(regime=regime, policy=variant, source_policy="information_only" if admit else "always_all",
                              risk_upper=float(upper[regime]) if admit else float(base.correct_risk[regime]),
                              risk_lower=float(chosen.risk_lower) if carry or not admit else float("nan"))
                tables.append(record)
    gates = pd.DataFrame(gates)
    replay = pd.DataFrame(tables)
    all_rows = pd.concat([rows, replay], ignore_index=True)
    baseline = rows[rows.policy == "always_all"][keys+["regime", "mean_terminal_risk", "correct_risk"]]
    baseline = baseline.rename(columns={"mean_terminal_risk": "fallback_empirical", "correct_risk": "fallback_population"})
    all_rows = all_rows.merge(baseline, on=keys+["regime"], validate="many_to_one")
    all_rows["upper_is_valid_population_bound"] = all_rows.policy != "no_carryover_gate"
    all_rows["empirical_improvement"] = all_rows.fallback_empirical-all_rows.mean_terminal_risk
    all_rows["model_improvement_lower"] = all_rows.fallback_population-all_rows.risk_upper
    all_rows.loc[~all_rows.upper_is_valid_population_bound, "model_improvement_lower"] = np.nan
    summary = []
    for (layer, regime, policy), group in all_rows.groupby(["layer", "regime", "policy"], sort=True):
        summary.append(dict(layer=layer, regime=regime, policy=policy, cells=len(group),
                            mean_risk=float(group.mean_terminal_risk.mean()),
                            mean_fallback_risk=float(group.fallback_empirical.mean()),
                            mean_improvement=float(group.empirical_improvement.mean()),
                            adverse_cells=int((group.empirical_improvement < -1e-14).sum()),
                            mean_model_improvement_lower=(float(group.model_improvement_lower.mean()) if group.upper_is_valid_population_bound.all() else None),
                            mean_messages=float(group.mean_messages.mean()),
                            mean_environment=float(group.mean_environment.mean())))
    assert len(replay) == 3584 and len(gates) == 1792
    corrected = gates[gates.policy == "risk_corrected_gate"]
    ablated = gates[gates.policy == "no_carryover_gate"]
    assert not (corrected.admit.to_numpy() & ~ablated.admit.to_numpy()).any()
    # The corrected policy's population upper risk satisfies the frozen safety rule.
    correct_rows = all_rows[all_rows.policy == "risk_corrected_gate"]
    allowances = correct_rows.apply(lambda r: safety[r.scenario_id] if registered and r.regime == "low" else 0.0, axis=1)
    assert (correct_rows.risk_upper <= correct_rows.fallback_population + allowances + 1e-14).all()
    assert np.isfinite(all_rows[["mean_terminal_risk", "mean_messages", "mean_environment", "risk_upper"]]).all().all()
    OUT.mkdir()
    all_rows.to_csv(OUT / "all_policy_cells.csv", index=False)
    gates.to_csv(OUT / "public_gate_decisions.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    report = dict(status="EXISTING_RESULT_RECONSTRUCTION_NOT_INDEPENDENT_CONFIRMATION",
                  observed_outcomes_known=True, epsilon="original_registered_scenario_values" if registered else 0,
                  candidate="frozen_information_only",
                  saved_seeds_per_cell=192, original_policy_cells=len(rows), replay_policy_cells=len(replay),
                  public_design_cells=len(corrected), admitted=int(corrected.admit.sum()),
                  no_carryover_admitted=int(ablated.admit.sum()),
                  all_old_policies_retained=8, new_trajectories=0, new_random_draws=0,
                  source_sha256=sha(source), design_sha256=sha(design), old_implementation_sha256=sha(implementation),
                  safety_manifest_sha256=sha(manifest_path), analysis_source_sha256=sha(Path(__file__)),
                  cpu_seconds=time.process_time()-start,
                  limits=["outcome-known old grid", "one fixed candidate, not exhaustive probe search",
                          "empirical means reconstructed, no fresh simulator execution or confidence interval",
                          "no certificate for unknown Gaussian parameters or neural learners"])
    (OUT / "integrity.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-safety", action="store_true")
    run(parser.parse_args().registered_safety)
