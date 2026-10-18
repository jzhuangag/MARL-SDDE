"""Independent deterministic checks of the new transfer interface, no sampling."""
import json
import math
from pathlib import Path
import sys

import same_stream_evidence as analysis
import numpy as np
import pandas as pd
from run_adaptation_cost_pilot import bhattacharyya_distance
from t018_static_scan import THETA_LOW


def main():
    out = analysis.HERE / "internal/same_stream_proof_audit_20261018.json"
    if out.exists():
        raise FileExistsError(out)
    evidence = analysis.HERE / "internal/same_stream_evidence_20261018"
    bounds = pd.read_csv(evidence / "risk_bounds.csv")
    old_integrity = json.loads((evidence / "integrity.json").read_text())
    assert old_integrity["analysis_script_sha256"] == analysis.sha(Path(analysis.__file__))
    rows = len(bounds)
    metadata = pd.read_csv(analysis.OLD / "results/exp016b_formal_20260801/metrics.csv", nrows=rows,
                          usecols=["scenario_id", "probe_used", "probe_q", "probe_b", "probe_n"])
    probes = metadata[metadata.probe_used].drop_duplicates()
    lookup = analysis.protocol.scenario_lookup()
    slacks = []
    for row in probes.itertuples():
        scenario = lookup[row.scenario_id]
        exponent = bhattacharyya_distance(THETA_LOW, scenario.theta_high, int(row.probe_q), int(row.probe_n), scenario.lam ** int(row.probe_b))
        slacks.append(exponent - math.log(1 / analysis.DELTA))
    assert min(slacks) >= -1e-12
    # Dense Gaussian conditioning verifies the scalar carryover identity for
    # deterministic, nonzero probe observations. These are fixtures, not data.
    conditional_checks = 0
    for theta in (0.05, 0.5, 4.0):
        for lam in (0.2, 0.7, 0.94):
            qd, bd, nd, q, b, n = 4, 2, 5, 8, 3, 7
            tp = bd * np.arange(1, nd + 1)
            tf = tp[-1] + b * np.arange(1, n + 1)
            pp = theta * lam ** np.abs(tp[:, None] - tp) + np.eye(nd) / qd
            fp = theta * lam ** np.abs(tf[:, None] - tp)
            ff = theta * lam ** np.abs(tf[:, None] - tf) + np.eye(n) / q
            observations = np.linspace(-0.9, 1.3, nd)
            terminal_cross = theta * lam ** (tp[-1] - tp)
            posterior_mean = float(terminal_cross @ np.linalg.solve(pp, observations))
            posterior_var = theta - float(terminal_cross @ np.linalg.solve(pp, terminal_cross))
            assert math.isclose(posterior_var, analysis.posterior_variance(theta, lam, qd, bd, nd), abs_tol=2e-14)
            mean = fp @ np.linalg.solve(pp, observations)
            covariance = ff - fp @ np.linalg.solve(pp, fp.T)
            for layer in analysis.protocol.LAYERS:
                stationary, d, k = analysis.risk(layer, n, 0, q, b, lam, theta)
                _, w = analysis.program(layer, n, 0)
                dense = (d + float(w @ mean)) ** 2 + float(w @ covariance @ w)
                formula = stationary + 2 * d * k * posterior_mean + k * k * (posterior_mean ** 2 - (theta - posterior_var))
                assert math.isclose(dense, formula, abs_tol=4e-14)
                conditional_checks += 1
    stabilities = []
    for delay in sorted({value.delay for key, value in lookup.items() if key in set(metadata.scenario_id)}):
        t = 0.1 / (delay + 1)
        coefficients = np.zeros(delay + 2)
        coefficients[0], coefficients[1] = 1.0, -1.0
        coefficients[-1] += t
        radius = float(np.max(np.abs(np.roots(coefficients))))
        threshold = 2 * math.sin(math.pi / (4 * delay + 2))
        assert t < threshold and radius < 1
        stabilities.append(dict(delay=delay, t=t, crossing_threshold=threshold, spectral_radius=radius))
    contrast = pd.read_csv(evidence / "contrasts.csv")
    archived = pd.read_csv(analysis.OLD / "results/exp016b_formal_20260801/analysis/scenario_primary_summary.csv")
    practical = contrast[contrast.layer.eq(analysis.protocol.LAYERS[0]) & contrast.practical_Z & contrast.finite_threshold_gate_eligible]
    scenario_means = practical.groupby("scenario_id").observed_mean_difference.mean()
    expected = archived.set_index("scenario_id").paired_mean_difference
    assert set(scenario_means.index) == set(expected.index)
    assert np.allclose(scenario_means.sort_index(), expected.sort_index(), atol=2e-15, rtol=2e-13)
    analysis.dump(out, dict(pass_all=True, new_trajectories=0,
                            exact_probe_designs_checked=len(probes), minimum_bhattacharyya_slack=min(slacks),
                            dense_conditional_identity_checks=conditional_checks,
                            original_primary_scenario_means_recovered=len(scenario_means),
                            no_replacement_of_registered_gate=True,
                            declared_fifo_stability=stabilities,
                            evidence_summary_sha256=analysis.sha(evidence / "summary.json"),
                            evidence_integrity_sha256=analysis.sha(evidence / "integrity.json"),
                            audit_script_sha256=analysis.sha(Path(__file__))))
    print(out)


if __name__ == "__main__":
    main()
