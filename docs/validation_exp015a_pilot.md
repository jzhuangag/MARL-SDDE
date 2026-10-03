# EXP-015A pilot validation: paid adaptive participation

## Decision

**Honest implementation-only pilot failure: 7/8 frozen gates passed.**

The only failed gate was short-horizon fallback:

```text
observed fallback rate = 0.777778
required fallback rate >= 0.80
```

The threshold is not rounded and was not changed. No formal seeds, GPU jobs,
HPC4 jobs, or MARL benchmark were started.

The autonomous research decision is **B**: retain the exact fixed-design
lower-bound and information-cost mainline, but stop larger experiments until
the minimum algorithm/theory gaps are closed. The current evidence does not
support option A because EXP-015A did not pass and AC-7--AC-9 remain open. It
does not force option C because the exact KL and empirical transition contain
non-generic dependence on correlation, mixing, participation, stride, delay,
and dual cost.

## Evidence freeze and execution audit

- Required starting commit:
  `e85d1667c85c5104cb9bd1c2b093ed2b46cb628c`.
- Preregistration commit, made before any output:
  `f410d9c2ccb9517359c2edf487a9b69659d7bd37`.
- Pilot seeds: `20271101`--`20271132`, permanently excluded from formal use.
- Rows: 55,296.
- Registered scenarios: 108.
- Hardware: local CPU only.
- Python: 3.11.11.
- NumPy/pandas: 2.2.2/2.2.3.
- Pilot wall time after implementation optimization: 10.70 s.

Exact command:

```powershell
C:\Users\jzhuangag\AppData\Local\anaconda3\envs\ust2\python.exe `
  experiments\dependence_delay_linear\run_adaptation_cost_pilot.py `
  --output-dir `
  experiments\dependence_delay_linear\results\exp015a_pilot_20260731
```

Two earlier invocations were stopped at the tool timeout before the runner
created its output directory. Before the successful evidence run, two
mathematically equivalent performance corrections were made:

1. exact Gaussian KL/Bhattacharyya evaluation changed from repeated Toeplitz
   eigendecomposition to Kalman-innovation recursions;
2. the AR(1) sample-mean variance changed from an explicit lag vector to the
   closed-form geometric sum.

The dense KL identity and all 17 EXP-015A tests were rerun after
these changes. No scientific constant, grid, seed, endpoint, or gate changed.

## Identification phase transition

Paid ETC results:

| Budget | Regime | Correct identification | Fallback | Mean probe messages | Mean probe environment |
|---|---|---:|---:|---:|---:|
| short | low | 0.2196 | 0.7778 | 11.44 | 9.11 |
| short | high | 0.2083 | 0.7778 | 11.44 | 9.11 |
| near | low | 0.9835 | 0.0000 | 160.44 | 48.56 |
| near | high | 0.9696 | 0.0000 | 160.44 | 48.56 |
| long | low | 0.9774 | 0.0000 | 160.44 | 48.56 |
| long | high | 0.9679 | 0.0000 | 160.44 | 48.56 |

The theoretical transition multiplier is 1.0. Empirically it lies between
the registered 0.5 and 1.1 budget levels: identification jumps from about
0.21 to at least 0.97. The bracket contains the prediction; its midpoint is
0.8, an absolute midpoint deviation of 0.2.

The failed fallback gate is localized: among 36 short-horizon scenario
families, 28 always fell back and eight admitted a paid probe. That produces
28/36 = 0.777778. This is a horizon-policy boundary issue, not a numerical
failure or an identification failure above threshold.

## Optimization and safety metrics

High-regime aggregate metrics:

| Budget | Policy | Mean oracle regret | Expected MSE | CVaR90 |
|---|---|---:|---:|---:|
| short | EXP-014B strict fallback | 1.37795 | 1.66655 | 11.7288 |
| short | paid ETC | 0.89501 | 1.18361 | 6.8209 |
| near | EXP-014B strict fallback | 0.89680 | 1.02086 | 6.3701 |
| near | paid ETC | 0.01125 | 0.13531 | 0.7704 |
| long | EXP-014B strict fallback | 0.44275 | 0.48706 | 3.1424 |
| long | paid ETC | 0.00170 | 0.04601 | 0.3107 |
| long | oracle | 0 | 0.04431 | 0.2791 |

Paid ETC improved mean oracle regret over EXP-014B strict fallback in all
36 prespecified high-regime long-horizon scenarios. The median cellwise
paid/strict oracle-regret ratio was `0.0079216`.

Mean safety deficit relative to all-agent stayed small but was not asserted
to be identically zero: in long budgets it was `0.000004` for high regimes
and `0.000018` for low regimes. This is the intended paid-exploration Pareto
metric, not a free no-harm claim.

The no-mixing-correction ablation had high/long mean oracle regret `0.21955`,
versus `0.00170` for paid ETC, confirming that Markov correction is
mechanistically material.

## Frozen gates

| Gate | Result |
|---|---|
| All rows finite | PASS |
| Both budgets valid | PASS |
| No hidden-state leakage | PASS |
| Long identification at least nominal 0.90 in both regimes | PASS |
| Long-short identification increase at least 0.25 | PASS |
| Short paid-ETC fallback at least 0.80 in both regimes | **FAIL: 0.777778** |
| Exploration amortized in at least 75% high/long scenarios and median regret ratio below 0.80 | PASS: 100%, ratio 0.00792 |
| Oracle regret below EXP-014B strict fallback on prespecified high regimes | PASS |

## Reproducibility and hashes

A clean same-seed local rerun reproduced the two scientific artifacts
byte-for-byte:

| Artifact | SHA-256 |
|---|---|
| `metrics.csv` | `62284DAF2F42A2EF2EA875BFAB4BF953B599FDB04B90DFB9D675D7B3F665B12A` |
| `summary.json` | `A99D811EE8AAF7D6DAA3FD97D305617CB5A0E724F65F6CDEAC147776C3A55976` |
| `metadata.json` | `55EFA3EFB5526495E3D0EAF7FC80126814B8B5E3428A996F32253AFBB94F4A08` |

The final full repository experiment suite completed with `134 passed in
5.87 s`. Captured stdout has SHA-256
`884B35F90BB3DB2E3531AA704758826B0DA9CC3DFAF46313E1340AF0C0AB1475`.

Primary output:

```text
experiments/dependence_delay_linear/results/exp015a_pilot_20260731
```

The 15.8 MB metrics file is ignored by Git and retained locally. The
machine-readable novelty matrix hash before the validation commit is
`D27CB9DD488D087F1DF8E72CC101F0BAEAD9319F8455DB77B8DA5E95668620A6`.

## Minimum next gap

Do not tune the 0.80 gate or launch formal seeds. The smallest defensible
next theory/algorithm task is:

1. prove AC-7, an adaptive chain-rule lower bound for history-dependent
   dimension-changing probes;
2. close AC-8 by replacing public mixing with a composite or separately
   certified mixing likelihood;
3. derive an explicit safety-slack rule whose short-budget fallback is a
   theorem consequence rather than a reserve heuristic.

Until these are closed, larger CPU grids and nonlinear MARL/GPU execution are
not scientifically justified.
